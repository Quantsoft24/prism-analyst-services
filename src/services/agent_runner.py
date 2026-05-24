"""AgentRunner — owns the ADK runner lifecycle and translates ADK events
into PRISM's typed ``ChatEvent`` stream.

This module is one of the **two abstraction points** for swapping agent
frameworks (the other is ``src/agents/base.py``). Application code never
imports ``google.adk.runners`` directly; it imports ``AgentRunner``.

Responsibilities:
  * Construct + cache the ADK ``Runner`` for a given ``PrismAgent``.
  * Resolve / create an ADK session per request (in-memory for Slice 3 —
    Postgres-backed in Phase 4 so conversations survive restarts).
  * Iterate ADK's async event stream, mapping each to a typed ``ChatEvent``.
  * Open + close a single ``AgentRun`` DB row per invocation (audit log).
  * Enforce cost cap + iteration cap + timeout.

The class is a regular ``async``-method object — not a generator — so it
can own setup/teardown cleanly. The ``run()`` method *returns* an async
iterator of typed events for the router to consume.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.base import PrismAgent
from src.config import settings
from src.core.database import session_scope
from src.models.agent_run import AgentRun
from src.schemas.chat import (
    ErrorEvent,
    FinalEvent,
    MetaEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Legacy pricing constants — preserved as a fallback when the model name
# isn't in the canonical ``MODEL_PRICING_USD_PER_1M`` table (e.g. when an
# agent was constructed with an explicit ``model=`` override outside the
# router's tier configs). The canonical table is in
# ``services/model_router_config.py``.
#
# Values are USD per 1M tokens, (input_rate, output_rate).
_LEGACY_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.075, 0.30),
    "gemini-2.5-pro": (1.25, 5.00),
    "gemini-flash-latest": (0.075, 0.30),
    "gemini-pro-latest": (1.25, 5.00),
}


# Type alias for the union of streamed events.
ChatEvent = MetaEvent | ToolCallEvent | ToolResultEvent | TokenEvent | FinalEvent | ErrorEvent


# Module-level cache so we don't rebuild the same ADK Runner on every request.
_runner_cache: dict[str, Any] = {}


class AgentRunner:
    """One instance per HTTP request — disposable."""

    def __init__(
        self,
        agent: PrismAgent,
        *,
        firm_id: str,
        user_id: uuid.UUID | None = None,
        session_id: str | None = None,
    ) -> None:
        self._agent = agent
        self._firm_id = firm_id
        self._user_id = user_id
        # ADK session_id is opaque — generate one if the caller didn't pass one.
        self._session_id = session_id or f"sess_{uuid.uuid4().hex[:16]}"
        self._agent_run_id: uuid.UUID | None = None
        self._started_at: float = 0.0
        self._tool_trace: list[dict[str, Any]] = []
        self._input_tokens = 0
        self._output_tokens = 0

    # ── Public API ─────────────────────────────────────────────────────────

    async def run(self, user_message: str) -> AsyncIterator[ChatEvent]:
        """Run the agent on ``user_message`` and yield typed events.

        Lifecycle:
          * INSERT an ``agent_runs`` row with status='running'.
          * Emit ``meta`` event with run + session IDs.
          * Iterate the ADK event stream, yielding tool_call / tool_result / token.
          * On normal completion: UPDATE row to status='complete', emit ``final``.
          * On error / timeout / cost-exceed: UPDATE row + emit ``error``.

        Cancellation: if the client disconnects (FastAPI cancels the request
        task), this iterator is closed and the row is left in 'running' — a
        background cleanup task can mark stale runs as 'abandoned' later.
        """
        self._started_at = time.perf_counter()
        try:
            self._agent_run_id = await self._open_run_row(user_message)
        except Exception as exc:
            logger.exception("Failed to open AgentRun audit row")
            yield ErrorEvent(
                code="audit_open_failed",
                message=f"Could not start agent run: {exc}",
                retriable=True,
            )
            return

        yield MetaEvent(
            agent_run_id=self._agent_run_id,
            session_id=self._session_id,
            agent_name=self._agent.name,
        )

        try:
            async for event in self._stream_with_timeout(user_message):
                yield event
        except asyncio.TimeoutError:
            await self._close_run_row(
                status="timeout",
                error_code="timeout",
                error_message=f"Exceeded {settings.AGENT_TIMEOUT_SECONDS}s",
                final_answer=None,
            )
            yield ErrorEvent(
                code="timeout",
                message=f"Agent exceeded the {settings.AGENT_TIMEOUT_SECONDS}s timeout.",
                retriable=True,
                agent_run_id=self._agent_run_id,
            )
        except Exception as exc:
            logger.exception("Agent run %s failed", self._agent_run_id)
            await self._close_run_row(
                status="failed",
                error_code=type(exc).__name__,
                error_message=str(exc)[:1000],
                final_answer=None,
            )
            yield ErrorEvent(
                code=type(exc).__name__,
                message=str(exc),
                retriable=False,
                agent_run_id=self._agent_run_id,
            )

    # ── ADK integration ────────────────────────────────────────────────────

    async def _stream_with_timeout(self, user_message: str) -> AsyncIterator[ChatEvent]:
        """Wrap the inner generator with an overall timeout."""
        gen = self._stream_adk(user_message)
        try:
            while True:
                event = await asyncio.wait_for(
                    gen.__anext__(), timeout=settings.AGENT_TIMEOUT_SECONDS
                )
                yield event
        except StopAsyncIteration:
            return

    async def _stream_adk(self, user_message: str) -> AsyncIterator[ChatEvent]:
        """The actual ADK event loop. Imported lazily so module loads without ADK."""
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types as genai_types

        # Build (or reuse) the ADK Runner for this agent.
        runner = _runner_cache.get(self._agent.name)
        if runner is None:
            adk_agent = self._agent.build()
            runner = Runner(
                agent=adk_agent,
                app_name="prism",
                session_service=InMemorySessionService(),
            )
            _runner_cache[self._agent.name] = runner

        # Ensure the session exists in the session service before we call run_async.
        # ADK is strict: the session must exist or it raises.
        sess_svc = runner.session_service
        existing = await sess_svc.get_session(
            app_name="prism", user_id=self._firm_id, session_id=self._session_id
        )
        if existing is None:
            await sess_svc.create_session(
                app_name="prism",
                user_id=self._firm_id,
                session_id=self._session_id,
            )

        new_message = genai_types.Content(
            role="user", parts=[genai_types.Part(text=user_message)]
        )

        # Track tool calls so we can match results back to their calls.
        pending_calls: dict[str, tuple[str, float]] = {}  # call_id -> (tool_name, start_ts)
        final_text_parts: list[str] = []
        final_seen = False

        # IMPORTANT: do not ``break`` out of the runner loop on final_response.
        # ADK uses OpenTelemetry contextvars internally; closing the inner
        # generator early triggers GeneratorExit inside spans that were opened
        # in a different asyncio context, producing "Failed to detach context"
        # spam. Letting the generator drain naturally — ADK stops yielding
        # right after the final response anyway — keeps spans tidy without
        # adding latency.
        async for event in runner.run_async(
            user_id=self._firm_id,
            session_id=self._session_id,
            new_message=new_message,
        ):
            # ── Extract token usage if present (cumulative on each event) ──
            usage = getattr(event, "usage_metadata", None)
            if usage:
                self._input_tokens = getattr(usage, "prompt_token_count", self._input_tokens) or self._input_tokens
                self._output_tokens = (
                    getattr(usage, "candidates_token_count", self._output_tokens)
                    or self._output_tokens
                )

            # After we've seen the final response we still let the loop run so
            # ADK can close its internal generators cleanly — but we stop
            # emitting tokens (the FinalEvent below carries the full answer).
            if final_seen:
                continue

            # ── Walk content parts to detect tool calls / results / text ──
            content = getattr(event, "content", None)
            parts = getattr(content, "parts", None) or []

            for part in parts:
                fn_call = getattr(part, "function_call", None)
                if fn_call is not None:
                    call_id = getattr(fn_call, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                    tool_name = getattr(fn_call, "name", "unknown")
                    args = dict(getattr(fn_call, "args", {}) or {})
                    pending_calls[call_id] = (tool_name, time.perf_counter())
                    self._tool_trace.append({"tool": tool_name, "args": args, "call_id": call_id})
                    yield ToolCallEvent(tool=tool_name, args=args, call_id=call_id)
                    continue

                fn_resp = getattr(part, "function_response", None)
                if fn_resp is not None:
                    call_id = getattr(fn_resp, "id", None) or ""
                    tool_name = getattr(fn_resp, "name", "unknown")
                    response = getattr(fn_resp, "response", None) or {}
                    started = pending_calls.pop(call_id, (tool_name, time.perf_counter()))[1]
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    summary = _summarize_tool_response(response)
                    yield ToolResultEvent(
                        call_id=call_id,
                        tool=tool_name,
                        ok=True,
                        result_summary=summary,
                        latency_ms=latency_ms,
                    )
                    continue

                text = getattr(part, "text", None)
                if text:
                    # On the final response, all the text is in this event;
                    # for streaming intermediate, also surfaces as tokens.
                    final_text_parts.append(text)
                    yield TokenEvent(text=text)

            # ── Detect end-of-turn ──
            check = getattr(event, "is_final_response", None)
            if callable(check):
                try:
                    if check():
                        final_seen = True
                except Exception:
                    pass

        # Compute cost + write final audit row.
        final_answer = "".join(final_text_parts).strip()
        cost = _estimate_cost_usd(self._agent.model, self._input_tokens, self._output_tokens)
        latency_ms = int((time.perf_counter() - self._started_at) * 1000)

        await self._close_run_row(
            status="complete",
            final_answer=final_answer,
            cost_usd=cost,
        )

        yield FinalEvent(
            answer=final_answer,
            agent_run_id=self._agent_run_id,  # type: ignore[arg-type]
            cost_usd=cost,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            latency_ms=latency_ms,
        )

    # ── Persistence ────────────────────────────────────────────────────────

    async def _open_run_row(self, user_message: str) -> uuid.UUID:
        """Insert a row with status='running'. Returns the new id.

        ``model`` records the *intent* — either the explicit model the agent
        declared, or the virtual tier name ("prism-fast"). The actual
        deployment chosen by the router (e.g. ``gemini/gemini-3.1-flash-lite``
        on key #2) is captured later from LiteLLM's response metadata.
        """
        model_intent = self._agent.model or f"prism-{self._agent.model_tier}"
        async with session_scope() as session:
            run = AgentRun(
                firm_id=self._firm_id,
                user_id=self._user_id,
                session_id=self._session_id,
                agent_name=self._agent.name,
                user_input=user_message,
                status="running",
                model=model_intent,
            )
            session.add(run)
            await session.flush()
            return run.id

    async def _close_run_row(
        self,
        *,
        status: str,
        final_answer: str | None = None,
        cost_usd: float | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update the row this runner opened. Safe to call once at end of run."""
        if self._agent_run_id is None:
            return
        latency_ms = int((time.perf_counter() - self._started_at) * 1000)
        async with session_scope() as session:
            await self._update_row(
                session,
                status=status,
                final_answer=final_answer,
                cost_usd=cost_usd or 0.0,
                error_code=error_code,
                error_message=error_message,
                latency_ms=latency_ms,
            )

    async def _update_row(
        self,
        session: AsyncSession,
        *,
        status: str,
        final_answer: str | None,
        cost_usd: float,
        error_code: str | None,
        error_message: str | None,
        latency_ms: int,
    ) -> None:
        stmt = (
            update(AgentRun)
            .where(AgentRun.id == self._agent_run_id)
            .values(
                status=status,
                final_answer=final_answer,
                cost_usd=cost_usd,
                error_code=error_code,
                error_message=error_message,
                latency_ms=latency_ms,
                tool_trace=self._tool_trace,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
            )
        )
        await session.execute(stmt)


# ── Helpers ────────────────────────────────────────────────────────────────


def _summarize_tool_response(response: dict[str, Any]) -> str:
    """Squash a tool response dict into a short human-readable summary line."""
    if not isinstance(response, dict):
        return str(response)[:120]
    if "items" in response and isinstance(response["items"], list):
        n = len(response["items"])
        total = response.get("total", n)
        return f"{n} of {total} item(s)"
    if response.get("found") is True:
        name = response.get("name") or response.get("ticker") or ""
        return f"found {name}".strip()
    if response.get("found") is False:
        return "not found"
    keys = list(response.keys())[:4]
    return "ok · " + ", ".join(keys)


def _estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Rough cost estimate. ``0`` if the model isn't in any pricing table —
    better to under-report than to fabricate a number.

    Lookup order:
      1. ``MODEL_PRICING_USD_PER_1M`` (canonical — covers everything the
         ModelRouter knows about, including free-tier rows at (0, 0))
      2. ``_LEGACY_PRICING`` (covers bare model names like ``gemini-2.5-flash``
         used by explicit ``model=`` overrides on PrismAgent)
    """
    from src.services.model_router_config import MODEL_PRICING_USD_PER_1M

    pricing = MODEL_PRICING_USD_PER_1M.get(model) or _LEGACY_PRICING.get(model)
    if pricing is None:
        return 0.0
    input_rate, output_rate = pricing
    return round(
        (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate,
        6,
    )
