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
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.base import PrismAgent
from src.config import settings
from src.core.database import session_scope
from src.integrations.tools._errors import (
    extract_error_message,
    is_error,
)
from src.models.agent_run import AgentRun
from src.schemas.chat import (
    AgentThoughtEvent,
    ChartEvent,
    ChartPoint,
    DataFreshnessEvent,
    ErrorEvent,
    FinalAnswer,
    FinalEvent,
    MetaEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolRetryEvent,
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
ChatEvent = (
    MetaEvent
    | ToolCallEvent
    | ToolResultEvent
    | TokenEvent
    | AgentThoughtEvent
    | ToolRetryEvent
    | DataFreshnessEvent
    | ChartEvent
    | FinalEvent
    | ErrorEvent
)


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
        client_key: str | None = None,
    ) -> None:
        self._agent = agent
        self._firm_id = firm_id
        self._user_id = user_id
        self._client_key = client_key
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

        # Synthetic opening plan thought — gives the UI an immediate "Thinking…"
        # card so the user sees agentic motion in the 1-3 seconds before the
        # first tool result arrives. Phrasing is intentionally generic so we
        # never lie about what tools the LLM will choose. See `_initial_plan_thought`.
        yield AgentThoughtEvent(
            text=_initial_plan_thought(user_message),
            kind="plan",
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
        # All `data_freshness` values surfaced by tools THIS turn. Used as
        # the allow-list when validating the agent's structured
        # ``data_freshness`` claim — see ``_validate_structured_freshness``.
        # Gemini occasionally fabricates freshness dates from training data;
        # this is the defence-in-depth that drops them silently before they
        # ever reach the UI's "as of …" chip.
        observed_freshness: set[str] = set()
        # Identical-call circuit breaker. Wire-log analysis showed Gemini
        # Flash occasionally fires the SAME (tool, args) call 3+ times in a
        # row when the tool keeps returning data the model finds inadequate.
        # ADK's iteration cap doesn't catch this — those are legitimately
        # different iterations as far as ADK is concerned. We bail at 3
        # consecutive identical signatures to prevent the 6-tool-card wall.
        _last_call_sig: str | None = None
        _identical_run: int = 0
        _IDENTICAL_RUN_LIMIT = 3
        _circuit_broken = False
        # Token de-duplication. Some ADK / Gemini configurations re-emit the
        # same text part across consecutive events (observed in production
        # as 3x repeated chunks during streaming). We dedupe at the runner
        # so the UI never sees the typing animation flicker on duplicates.
        _last_text_emitted: str | None = None

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
                    # Circuit-breaker for repeated-same-call loops. Compute a
                    # stable signature on tool + sorted args. Same signature
                    # 3 times in a row → break: emit an error, stop iterating.
                    try:
                        sig = f"{tool_name}::{json.dumps(args, sort_keys=True, default=str)}"
                    except (TypeError, ValueError):
                        sig = f"{tool_name}::<unserialisable>"
                    if sig == _last_call_sig:
                        _identical_run += 1
                    else:
                        _identical_run = 1
                        _last_call_sig = sig
                    if _identical_run >= _IDENTICAL_RUN_LIMIT:
                        logger.warning(
                            "Agent run %s hit identical-call circuit breaker: "
                            "tool=%r fired %d times in a row with the same args. "
                            "Stopping the loop.",
                            self._agent_run_id, tool_name, _identical_run,
                        )
                        yield ErrorEvent(
                            code="identical_call_loop",
                            message=(
                                f"The agent called `{tool_name}` with the same "
                                f"arguments {_identical_run} times in a row without "
                                "making progress. Stopping to avoid wasting your "
                                "tokens. Try rephrasing the question with more "
                                "specifics (a single ticker, an explicit fiscal "
                                "period, or a different metric)."
                            ),
                            retriable=True,
                            agent_run_id=self._agent_run_id,
                        )
                        _circuit_broken = True
                        break  # exit the parts loop
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

                    # Distinguish success from failure using the structured
                    # error contract (see src/integrations/tools/_errors.py).
                    if is_error(response):
                        err_msg = extract_error_message(response) or "tool error"
                        err_code = (
                            response.get("error_code") if isinstance(response, dict) else None
                        )
                        next_action = (
                            response.get("next_action") if isinstance(response, dict) else None
                        )
                        yield ToolResultEvent(
                            call_id=call_id,
                            tool=tool_name,
                            ok=False,
                            error=err_msg,
                            error_code=err_code,
                            next_action=next_action,
                            latency_ms=latency_ms,
                        )
                        continue

                    # Success path. Before the result event, emit a
                    # ToolRetryEvent if the tool helper had to silently
                    # retry a transient blip (see stock_chat._post /
                    # bmc._request). The frontend shows ↻ on the tool
                    # card so users know we recovered from a hiccup.
                    if isinstance(response, dict):
                        rc = response.get("retry_count")
                        if isinstance(rc, int) and rc > 0:
                            yield ToolRetryEvent(
                                call_id=call_id,
                                tool=tool_name,
                                attempt=rc + 1,  # 1-indexed; rc=1 means we're on attempt 2
                                reason="transient transport blip (auto-retried)",
                            )

                    summary = _summarize_tool_response(response)
                    # Attach a trimmed copy of the response to the matching
                    # tool_trace entry. The rescue path
                    # (_rescue_empty_synthesis) needs the actual returned
                    # data — not just the name/args — to compose a prose
                    # answer when the orchestrator skips synthesis.
                    for entry in self._tool_trace:
                        if entry.get("call_id") == call_id:
                            entry["response"] = _trim_response_for_rescue(response)
                            break
                    yield ToolResultEvent(
                        call_id=call_id,
                        tool=tool_name,
                        ok=True,
                        result_summary=summary,
                        latency_ms=latency_ms,
                    )
                    # Surface data freshness when the tool gave us one (filings
                    # tools return the latest announcement_dt; technicals
                    # returns "live"). Frontend renders an "as of …" chip on
                    # the corresponding tool card + the final answer.
                    if isinstance(response, dict):
                        freshness = response.get("data_freshness")
                        if freshness:
                            as_of_str = str(freshness)
                            observed_freshness.add(as_of_str)
                            yield DataFreshnessEvent(
                                call_id=call_id,
                                source=_freshness_source_label(tool_name),
                                as_of=as_of_str,
                            )
                    # Auto-chart from time-series rows when the tool emits
                    # a recognizable shape (financials_query with period_end +
                    # a numeric series across >=3 rows). The helper returns
                    # None when conditions aren't met — never a fabricated chart.
                    chart_evt = _try_emit_chart(tool_name, call_id, response)
                    if chart_evt is not None:
                        yield chart_evt
                    continue

                # Text part — either a "thought" (Gemini thinking mode, if ever
                # enabled) or the actual answer text. Thought parts MUST NOT
                # accumulate into the final answer prose.
                text = getattr(part, "text", None)
                if text:
                    if bool(getattr(part, "thought", False)):
                        yield AgentThoughtEvent(text=text, kind="reflect")
                        continue
                    # Token dedup: production wire logs showed ADK / Gemini
                    # occasionally re-emit byte-identical text parts across
                    # consecutive events (observed as 3x duplicate streaming
                    # chunks). Drop exact duplicates — the final answer is
                    # built from the SAME source so it stays clean.
                    if text == _last_text_emitted:
                        logger.debug(
                            "Agent run %s: skipping duplicate text part "
                            "(%d chars).",
                            self._agent_run_id, len(text),
                        )
                        continue
                    _last_text_emitted = text
                    final_text_parts.append(text)
                    # Re-chunk large text parts so the UI sees a smooth typing
                    # cadence instead of a single 500+ char drop. ADK's Gemini
                    # adapter typically yields the entire final answer in one
                    # part. Small parts pass through with no delay so naturally-
                    # streamed mid-turn text stays snappy. Cancellation
                    # propagates through `asyncio.sleep` cleanly.
                    async for chunk in _TokenChunker.stream(text):
                        yield TokenEvent(text=chunk)

            # Circuit-breaker fired inside the parts loop — stop iterating
            # ADK events entirely. Without this break we'd keep streaming
            # downstream events from the same broken turn.
            if _circuit_broken:
                final_seen = True
                break

            # ── Detect end-of-turn ──
            check = getattr(event, "is_final_response", None)
            if callable(check):
                try:
                    if check():
                        final_seen = True
                except Exception:
                    pass

        # Compute cost + write final audit row.
        raw_final = "".join(final_text_parts).strip()
        prose, structured = _split_structured_answer(raw_final)

        # Defence-in-depth: silently drop a fabricated `data_freshness`
        # from the structured payload. Gemini sometimes writes a date
        # from training data into the meta block even when no tool
        # returned one this turn. Without this guard the UI shows a
        # misleading "as of <date>" chip that doesn't trace back to any
        # real source. See the prompt's <answer_meta> rules — this is
        # the runtime check that backs the rule.
        structured = _validate_structured_freshness(structured, observed_freshness)

        # Stall detection: the model wrote prose like "I will re-run the
        # query..." but the turn already ended without firing another
        # tool. ADK accepted the narration as the final answer; the user
        # sees a useless stall message. We treat this the same as empty
        # prose — blank `prose` so the rescue path below fires. The
        # detector is conservative (substantive-content guard) to avoid
        # false-positives on real answers that mention a retry. See
        # `_is_stall_response` for the exact logic.
        if prose and _is_stall_response(prose):
            logger.warning(
                "Agent run %s emitted stall prose (%d chars, no "
                "substantive content); routing to synthesis rescue.",
                self._agent_run_id,
                len(prose),
            )
            prose = ""  # zero out so the empty-prose rescue path runs

        # Safety net for the "empty prose" failure mode. Gemini sometimes
        # terminates a turn after a tool call without writing the prose answer
        # — either zero output, or (after the 2026-05-28 prompt change) ONLY
        # the <answer_meta> block. The system prompt forbids this (Rule 0)
        # but Flash is non-deterministic. Layered rescue (most-to-least useful):
        #
        #   1. structured.sections[0].body  → promote it to prose. Gemini
        #      put the answer in the section body; just surface it.
        #   2. structured.citations exist   → emit a short "data retrieved,
        #      see Report tab" pointer. The right pane already has the data.
        #   3. otherwise                    → existing generic fallback
        #      that names the tools so users know something happened.
        if not prose:
            if structured is not None and structured.sections:
                first_body = (structured.sections[0].body or "").strip()
                if first_body:
                    prose = first_body
                    # Sync structured.text so the Report tab + chat thread
                    # render the same content.
                    structured = structured.model_copy(update={"text": prose})
                    logger.warning(
                        "Agent run %s emitted only meta block; promoted "
                        "sections[0].body to prose.",
                        self._agent_run_id,
                    )
            if not prose and structured is not None and structured.citations:
                prose = (
                    "I retrieved the data but didn't compose a written "
                    "summary this turn — see the **Report** and **Sources** "
                    "tabs on the right for the full breakdown."
                )
                structured = structured.model_copy(update={"text": prose})
                logger.warning(
                    "Agent run %s emitted only meta block with citations; "
                    "surfacing pointer-to-Report-tab fallback.",
                    self._agent_run_id,
                )
            if not prose:
                # Industry-grade rescue: when the orchestrator (Flash)
                # ran tools but skipped synthesis, ask a higher-tier
                # model (Pro) to compose the answer in a single targeted
                # call. The retry is single-shot (no tools, no agent
                # loop) so it can't hit the same multi-turn failure
                # mode. ~1-2s latency on failure turns only; zero impact
                # on the happy path. Matches the Claude / Cursor / Devin
                # pattern for synthesis rescue.
                rescued = await _rescue_empty_synthesis(
                    user_message=user_message,
                    tool_trace=self._tool_trace,
                )
                if rescued:
                    prose = rescued
                    # Sync the prose into structured.text so the Report
                    # tab + chat thread render the same content.
                    if structured is not None:
                        structured = structured.model_copy(update={"text": prose})
                    logger.warning(
                        "Agent run %s rescued synthesis via single-shot "
                        "Pro retry after orchestrator skipped prose.",
                        self._agent_run_id,
                    )
                else:
                    # Last-resort deterministic message. The rescue path
                    # already covers every realistic failure (network,
                    # empty Pro response, exception); this is here to
                    # guarantee the user always sees SOMETHING.
                    prose = _synthesize_empty_answer_fallback(self._tool_trace)
                    structured = None
                    logger.warning(
                        "Agent run %s: rescue returned empty too; "
                        "surfacing generic fallback. %d tools called.",
                        self._agent_run_id,
                        len(self._tool_trace),
                    )

        cost = _estimate_cost_usd(self._agent.model, self._input_tokens, self._output_tokens)
        latency_ms = int((time.perf_counter() - self._started_at) * 1000)

        await self._close_run_row(
            status="complete",
            final_answer=prose,
            cost_usd=cost,
        )

        yield FinalEvent(
            answer=prose,
            structured=structured,
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
                client_key=self._client_key,
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
    """Squash a tool response dict into a short human-readable summary line.

    Never inspects the error path — the runner branches on ``is_error()``
    before calling this. Designed for the UI tool-call card: ~80 chars,
    no trailing punctuation, no markdown.
    """
    if not isinstance(response, dict):
        return str(response)[:120]

    # ── financials_query shapes — check FIRST so they don't fall through to
    # the generic "ok · keys" path. The financials envelope always carries
    # rows + sql + needs_clarification + clarification, so the bare-key
    # fallback below would render an unreadable "ok · rows, sql, needs_…"
    # for every call. Explicit branches:
    if (
        "rows" in response and "sql" in response
        and "needs_clarification" in response
    ):
        rows = response.get("rows") or []
        clarif = response.get("needs_clarification")
        # NOT IN DATABASE refusal — the upstream returns a one-row stub.
        if (
            isinstance(rows, list) and len(rows) == 1
            and isinstance(rows[0], dict)
            and isinstance(rows[0].get("note"), str)
            and rows[0]["note"].startswith("NOT IN DATABASE")
        ):
            return "no data · NOT IN DATABASE"
        # Clarification gate fired (and the wrapper's auto-disambig couldn't
        # resolve, so the user has to pick).
        if clarif:
            # Count candidates in the clarification text — gives the UI a
            # quick sense of how much ambiguity there is.
            text = response.get("clarification") or ""
            n_candidates = sum(
                1 for line in text.splitlines()
                if line.strip().startswith(("1.", "2.", "3.", "4.", "5.",
                                            "1)", "2)", "3)", "4)", "5)"))
            )
            n_label = f"{n_candidates} candidates" if n_candidates else "ambiguous"
            return f"needs clarification · {n_label}"
        # Happy path — non-empty rows.
        if rows:
            n = len(rows)
            chip = ""
            picked = response.get("auto_disambiguated_to")
            if picked:
                # Trim to keep the chip readable in the tool card.
                picked_short = (picked[:24] + "…") if len(picked) > 25 else picked
                chip = f" · auto-resolved: {picked_short}"
            return f"{n} row{'s' if n != 1 else ''}{chip}"
        # Empty rows, no clarification, no NOT IN DATABASE — odd but possible.
        return "no data"

    # Company list / filings list / generic items array
    if "items" in response and isinstance(response["items"], list):
        n = len(response["items"])
        total = response.get("total", n)
        suggestions = response.get("suggestions") or []
        suffix = f" · {len(suggestions)} near-match(es)" if suggestions else ""
        return f"{n} of {total} item(s){suffix}"
    if "filings" in response and isinstance(response["filings"], list):
        n = len(response["filings"])
        total = response.get("total", n)
        return f"{n} of {total} filing(s)"
    if "blocks" in response and isinstance(response["blocks"], list):
        return f"{len(response['blocks'])}-block canvas"
    if response.get("found") is True:
        name = response.get("name") or response.get("ticker") or ""
        return f"found {name}".strip()
    if response.get("found") is False:
        suggestions = response.get("suggestions") or []
        if suggestions:
            return f"not found · {len(suggestions)} suggestion(s)"
        return "not found"
    if "answer" in response and response["answer"]:
        # Filings-read / block-chat — show a hint of the answer
        snippet = str(response["answer"]).strip().split("\n", 1)[0][:60]
        return f"answer: {snippet}…" if len(snippet) >= 60 else f"answer: {snippet}"
    if "result" in response:
        unit = response.get("unit", "")
        return f"= {response['result']}{unit}".strip()
    keys = [k for k in response.keys() if not k.startswith("_")][:4]
    return "ok · " + ", ".join(keys)


# Regex anchored to the end of the response. Uses GREEDY ``.*`` for the JSON
# capture (not ``\{.*?\}``) so it handles nested braces / arrays without
# truncating mid-payload — critical when the LLM emits a citations array of
# objects: a non-greedy ``\{.*?\}`` would stop at the FIRST inner ``}``,
# capture invalid JSON, and the whole block would leak into the visible prose.
# The ``\s*$`` anchor ensures we still only match a block at the very end of
# the response, not a stray inline ``<answer_meta>`` tag.
_ANSWER_META_RE = re.compile(
    r"<answer_meta>\s*(?P<json>.*)</answer_meta>\s*$",
    re.DOTALL,
)

# Defensive fallback: if the main regex fails to parse cleanly for any reason
# (truncated block, unbalanced JSON, LLM put text after the closing tag),
# we still strip any ``<answer_meta>...</answer_meta>`` pair from the tail so
# users NEVER see the raw block as visible prose. Same end-anchor.
_ANSWER_META_STRIP_RE = re.compile(
    r"<answer_meta>.*?</answer_meta>\s*$",
    re.DOTALL,
)

# Backstop: strip a trailing "Sources: ..." line if the LLM appended one
# despite the prompt rule against it. The UI's right-pane "Sources" tab is
# the canonical place for source attribution; a prose line is duplicate noise.
# Matches the most common shapes Gemini Flash produces:
#   "\nSources: financials_query"
#   "\n**Sources:** [...]"
#   "\n\n- Sources: foo, bar"
_TRAILING_SOURCES_RE = re.compile(
    r"\n+\s*[\*\-•]*\s*\**\s*Sources?\s*:\s*[^\n]+\s*$",
    re.IGNORECASE,
)

# Valid Literal values lifted from src/schemas/chat.py — kept here so we can
# coerce LLM output BEFORE Pydantic strict-validates and rejects the whole
# structured payload. Used by ``_coerce_citation`` / ``_coerce_section``.
_VALID_SOURCE_KIND = frozenset({"filing", "web", "bmc", "tool"})
_VALID_CONFIDENCE = frozenset({"high", "medium", "low"})
_VALID_SECTION_KIND = frozenset({"summary", "anomaly", "note"})


def _coerce_citation(c: Any) -> dict | None:
    """Soften LLM-emitted citation fields so Pydantic accepts them.

    Gemini sometimes writes ``source_kind: "financials"`` (or anything else);
    rather than reject the whole structured payload over a vocabulary mismatch,
    we coerce unknown values to ``"tool"`` (the safe default the schema
    already uses). Non-dict entries are dropped.
    """
    if not isinstance(c, dict):
        return None
    out = dict(c)
    if out.get("source_kind") not in _VALID_SOURCE_KIND:
        out["source_kind"] = "tool"
    return out


def _coerce_section(s: Any) -> dict | None:
    """Same idea as ``_coerce_citation`` for ``FinalSection``: unknown
    ``kind`` values get mapped to ``"summary"``. Missing title/body → drop."""
    if not isinstance(s, dict):
        return None
    if not s.get("title") or not s.get("body"):
        return None
    out = dict(s)
    if out.get("kind") not in _VALID_SECTION_KIND:
        out["kind"] = "summary"
    return out


def _clean_prose_tail(prose: str) -> str:
    """Strip artefacts the LLM commonly appends despite the prompt rules:
    (1) a raw ``<answer_meta>...</answer_meta>`` block, (2) a trailing
    ``Sources:`` line. Run AFTER the meta block has been extracted (or
    after a parse-failure), so it's always a defensive last pass."""
    if not prose:
        return prose
    cleaned = _ANSWER_META_STRIP_RE.sub("", prose).rstrip()
    cleaned = _TRAILING_SOURCES_RE.sub("", cleaned).rstrip()
    return cleaned


def _split_structured_answer(raw: str) -> tuple[str, FinalAnswer | None]:
    """Extract an optional structured FinalAnswer block from the prose tail.

    Contract: the agent SHOULD end its response with::

        <answer_meta>{"citations":[...],"confidence":"high",
          "data_freshness":"2025-03-31","kpis":[...],"sections":[...]}</answer_meta>

    Returns ``(prose, FinalAnswer | None)``. Soft contract: NEVER errors on
    a malformed block, NEVER leaks the raw meta block into the visible
    prose, NEVER preserves a trailing "Sources:" line.

    Layered fallback (most-to-least preferred):
      1. Regex matches + JSON parses + FinalAnswer validates → full structured.
      2. Regex matches + JSON parses + validation fails per-field → coerced
         and partial FinalAnswer (citations w/ unknown source_kind → "tool",
         unknown confidence → "medium", invalid sections dropped).
      3. Regex matches + JSON fails → strip the block from prose, return prose.
      4. Regex misses → strip any raw meta tags from prose defensively.
    """
    if not raw:
        return raw, None
    match = _ANSWER_META_RE.search(raw)
    if not match:
        # No structured block — still defensively clean any artefacts so
        # the user never sees raw meta tags or a duplicate Sources line.
        return _clean_prose_tail(raw), None
    prose = _clean_prose_tail(raw[: match.start()])
    payload = match.group("json").strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        logger.debug("answer_meta JSON parse failed; returning cleaned prose")
        return prose, None
    if not isinstance(data, dict):
        return prose, None

    # Coerce LLM output to schema-valid shapes BEFORE Pydantic strict-validates.
    confidence = data.get("confidence")
    if confidence not in _VALID_CONFIDENCE:
        confidence = "medium"
    citations = [
        c for c in (_coerce_citation(x) for x in (data.get("citations") or []))
        if c is not None
    ]
    sections = [
        s for s in (_coerce_section(x) for x in (data.get("sections") or []))
        if s is not None
    ]
    kpis_raw = data.get("kpis") or []
    kpis = [k for k in kpis_raw if isinstance(k, dict) and k.get("label") and k.get("value")]

    try:
        structured = FinalAnswer(
            text=prose,
            citations=citations,
            confidence=confidence,
            data_freshness=data.get("data_freshness"),
            kpis=kpis,
            sections=sections,
        )
    except Exception as exc:
        logger.debug("answer_meta validation failed after coercion: %s", exc)
        return prose, None
    return prose, structured


# Stall phrases — fragments that signal the model wrote prose PROMISING
# another tool call instead of answering. Matched case-insensitively as
# substrings. Curated to be unambiguous: each phrase only legitimately
# appears in mid-process narration, not in a real analyst answer.
_STALL_PHRASES: tuple[str, ...] = (
    # "I will re-run" family — model promises a tool call it never fires
    "i will re-run", "i'll re-run", "let me re-run",
    "i will re-query", "let me re-query",
    "i am still retrieving", "i'm still retrieving", "still retrieving",
    "still investigating", "still gathering",
    "i will try again", "let me try again", "i'll try again",
    "initial query did not return all",
    "initial query didn't return all",
    "did not return all the necessary",
    "didn't return all the necessary",
    "let me gather more", "i will gather more",
    "i'll check again", "let me check again",
    "i need to re-query", "i need to retry",
    # "Give up" family — model writes a long refusal despite having data
    # (added 2026-05-29 after the wire log showed this failure shape on
    # "Compare TCS, Infosys, Wipro, HCLTech…": tool returned data, model
    # composed a "I do not have access to comparative metrics" refusal).
    "i do not have access to",
    "i don't have access to",
    "i am unable to provide",
    "i'm unable to provide",
    "currently not available",
    "could not be fulfilled",
    "my current database does not",
    "the available data does not support",
    "the available financial query tools",
    "is not available through the available",
)

# Regex patterns that indicate substantive financial content in prose.
# Used as the false-positive guard for _is_stall_response — if prose
# carries actual VALUES (numbers with thousands-separators, percentages,
# currency symbols, unit words like "crore"), we trust it even if it
# also mentions a retry ("we tried twice; here's TCS at ₹2.46L cr...").
#
# Critical: FY labels (FY25, FY24) alone are NOT substantive. A stall
# like "I will fetch revenue for FY25" mentions FY25 but delivers no
# value — so FY-only matches must not exempt prose from being a stall.
_SUBSTANTIVE_NUMBER_RE = re.compile(
    r"\d{1,3}(?:[,.]\d{3})+|\d+\.?\d*\s*%",  # 80,787  or  24.6%
    re.IGNORECASE,
)
_SUBSTANTIVE_UNITS = (
    "₹", "$ ", "crore", "lakh", "billion", "million", " bps", "ebitda",
)


def _is_stall_response(prose: str) -> bool:
    """True when the model wrote text that promises a future tool call
    but never fires one — the "I will re-run the query" failure mode.

    Two conditions must hold:
      1. At least one ``_STALL_PHRASES`` fragment is present.
      2. The prose has NO substantive VALUE content — no ₹/$/% symbol,
         no big formatted number (80,787), no unit word (crore / lakh).
         FY labels alone (FY25) are NOT enough; they often appear inside
         stall phrases ("I will fetch ... for FY25").
    Both → it's pure stalling. Either alone → leave the prose intact.

    This is the runner-side guard that pairs with the prompt-level
    "NO STALLING" rule in `agents/company_intel.py`. Gemini Flash
    occasionally writes a stalling final answer despite the rule; the
    runner detects it and triggers the same Pro-rescue path that
    handles empty prose.
    """
    if not prose:
        return False
    lower = prose.lower()
    if not any(phrase in lower for phrase in _STALL_PHRASES):
        return False
    if any(unit in lower for unit in _SUBSTANTIVE_UNITS):
        return False
    if _SUBSTANTIVE_NUMBER_RE.search(prose):
        return False
    return True


def _trim_response_for_rescue(response: Any, max_rows: int = 20, max_str: int = 2000) -> Any:
    """Trim a tool response to a size suitable for (a) the audit log and
    (b) the synthesis-rescue context window.

    Tool responses can be large — financials_query can return 200 rows,
    stock_filings_read returns multi-page PDF excerpts. Storing those raw
    on every tool_trace entry blows up the audit row and the rescue
    prompt. We keep the shape but bound the size:
      * lists longer than ``max_rows`` → first ``max_rows`` + a marker
      * strings longer than ``max_str`` → cropped + ellipsis
      * non-dict responses pass through unchanged
    """
    if not isinstance(response, dict):
        return response
    out: dict = {}
    for k, v in response.items():
        if isinstance(v, list) and len(v) > max_rows:
            out[k] = list(v[:max_rows]) + [f"... (+{len(v) - max_rows} more)"]
        elif isinstance(v, str) and len(v) > max_str:
            out[k] = v[:max_str] + "..."
        else:
            out[k] = v
    return out


# System prompt for the synthesis-rescue single-shot call. Deliberately
# narrow: write the answer, cite inline, no tool calls, no meta block.
_RESCUE_SYSTEM_PROMPT = """You are a senior research analyst's writing assistant. \
The research agent ran tools successfully but failed to compose a written \
answer for the user. Your job is to write that answer using ONLY the tool \
data provided below — do not invent numbers, dates, or companies. Use the \
inline citation format [Source | period] sparingly. Write 2-4 sentences of \
prose; include 2-4 supporting bullets if the data warrants them. Do NOT \
call any tools, do NOT write a <answer_meta> block, do NOT apologise for \
the agent's earlier silence. Prose only."""


async def _rescue_empty_synthesis(
    user_message: str,
    tool_trace: list[dict[str, Any]],
) -> str | None:
    """Single-shot Pro call that composes prose when the orchestrator skips it.

    Returns the rescued prose, or ``None`` on any failure (no API key,
    network/timeout, empty response, exception). Caller falls back to the
    deterministic message on ``None``.

    Why a separate call instead of just looping back through the agent:
      * Single-shot has NO tools attached → can't hit the same "model
        decides to call another tool and skips synthesis" failure mode.
      * Pro (not Flash) is far more reliable about following an explicit
        "write the answer" directive — Flash is exactly what failed.
      * No agent loop overhead; ~1-2s vs 5-10s for a full agent re-run.

    Triggered only on the empty-prose path so steady-state cost is zero.
    On failures (~5-10% of turns in early measurement) it adds one Pro
    call (~$0.001) and ~1-2s of latency, in exchange for a real answer
    instead of the generic "I ran N tool(s)…" message.
    """
    if not tool_trace:
        return None

    # Pick the first non-empty Gemini key — we share the same pool as the
    # main router. If no key is set we can't rescue; caller falls back.
    api_key = next((k for k in settings.gemini_api_keys if k), None)
    if not api_key:
        logger.warning("Rescue skipped: no GEMINI_API_KEY available.")
        return None

    # Build a compact summary of the tools and their (already-trimmed)
    # responses. Last 3 tools max to keep the prompt lean.
    parts: list[str] = []
    for entry in tool_trace[-3:]:
        tool = entry.get("tool", "?")
        args = entry.get("args", {})
        response = entry.get("response")
        if response is None:
            continue
        try:
            response_str = json.dumps(response, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            response_str = repr(response)
        # Defensive: even after _trim_response_for_rescue, cap one final time.
        if len(response_str) > 2500:
            response_str = response_str[:2500] + "..."
        parts.append(f"{tool}(args={args}) →\n{response_str}")

    if not parts:
        return None

    user_prompt = (
        f"ORIGINAL USER QUESTION:\n{user_message}\n\n"
        "TOOL CALLS AND RESPONSES (most recent last):\n\n"
        + "\n\n---\n\n".join(parts)
        + "\n\nNow write the analyst's answer using the data above."
    )

    try:
        # Lazy import — keeps module load fast on the happy path.
        import litellm

        resp = await litellm.acompletion(
            model="gemini/gemini-2.5-pro",
            api_key=api_key,
            messages=[
                {"role": "system", "content": _RESCUE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=600,
            temperature=0.2,  # low temp — composing from given data, not creating
            timeout=20,
        )
        text_obj = resp.choices[0].message.content if resp.choices else None
        text = (text_obj or "").strip()
        # Strip any accidental <answer_meta> tail Pro might still emit.
        meta_at = text.find("<answer_meta>")
        if meta_at >= 0:
            text = text[:meta_at].rstrip()
        return text or None
    except Exception as exc:  # noqa: BLE001 — any failure → fall back
        logger.warning("Synthesis rescue failed (%s): %s", type(exc).__name__, exc)
        return None


def _validate_structured_freshness(
    structured: FinalAnswer | None, observed: set[str],
) -> FinalAnswer | None:
    """Drop ``data_freshness`` from a structured payload when it doesn't
    trace back to a date the runner actually saw a tool emit this turn.

    Why this exists: Gemini sometimes writes a plausible-looking date
    (typically last quarter or "today") into ``<answer_meta>.data_freshness``
    even when the question didn't invoke any date-returning tool. The
    UI faithfully renders that as an "as of <date>" chip, which misleads
    analysts into believing data exists where none does. The prompt tells
    the model not to do this; this validator is the runtime guarantee.

    Validation is conservative: we only drop on a *positive mismatch* —
    the model wrote a string, and that exact string isn't in the observed
    set. ``observed`` is the set of `data_freshness` values surfaced by
    tools this turn (financials_query rows' `period_end`, stock_filings_*'s
    latest `announcement_dt`, technicals' literal "live"). Empty/None
    `data_freshness` is left alone (the model correctly omitted it).
    """
    if structured is None or not structured.data_freshness:
        return structured
    if structured.data_freshness in observed:
        return structured
    logger.warning(
        "Dropping fabricated data_freshness %r from structured payload "
        "(no tool emitted that date; observed=%s)",
        structured.data_freshness,
        sorted(observed),
    )
    return structured.model_copy(update={"data_freshness": None})


def _synthesize_empty_answer_fallback(tool_trace: list[dict[str, Any]]) -> str:
    """Produce a user-facing message when Gemini terminated a turn without
    emitting any final text.

    Heuristics tuned to the failure modes we've observed in prod:

      • Single ``search_companies`` call → user query was probably ambiguous
        ("Adani", "Tata"); LLM saw multiple matches and bailed. Tell the
        user to pick one.
      • Any other shape → generic fallback citing the tools that ran so the
        user knows their question wasn't dropped.

    The message stays short and points the user at a concrete next action.
    Never refer to the LLM model name or internal failure — that's our
    bug, not theirs.
    """
    if not tool_trace:
        return (
            "I couldn't put together an answer for that question. "
            "Try a more specific query — e.g. a ticker like RELIANCE or "
            "a company name like 'HDFC Bank'."
        )

    tool_names = [t.get("tool", "") for t in tool_trace]
    only_search = (
        len(tool_trace) == 1 and tool_names[0] == "search_companies"
    )
    if only_search:
        args = tool_trace[0].get("args", {})
        query = args.get("query", "your query") if isinstance(args, dict) else "your query"
        return (
            f"The search for **{query}** returned multiple companies — "
            "I need you to pick which one to research. "
            "Try again with a specific ticker (e.g. ADANIENT, ADANIPORTS, "
            "ADANIGREEN) or a more complete name."
        )

    called = ", ".join(f"`{n}`" for n in tool_names)
    return (
        f"I ran {len(tool_trace)} tool(s) ({called}) but didn't have enough "
        "to produce a confident answer. Try a more specific question — "
        "for example a single ticker, a specific time period, or what you "
        "want to know (filings vs. price vs. business model)."
    )


def _freshness_source_label(tool_name: str) -> str:
    """Map a tool name to the short UI label shown in the freshness chip."""
    if tool_name.startswith("stock_filings"):
        return "filings catalog"
    if tool_name == "stock_technicals":
        return "market data"
    if tool_name.startswith("bmc_"):
        return "business model canvas"
    if tool_name == "web_search":
        return "web search"
    return tool_name


class _TokenChunker:
    """Chunk long text into word-boundary beats with short async sleeps.

    Why this exists: ADK's Gemini adapter typically emits the full final
    answer as ONE big text part. The frontend then sees a single TokenEvent
    of 200-2000 chars and the prose "pops" into view all at once — a long
    wait followed by a sudden drop. Re-chunking on the server gives the UI
    the same smooth typing cadence the mock-mode scenarios produce
    (``prism-analyst-platform/src/lib/api/chat.mock.ts``), without changing
    the wire shape (still ``TokenEvent`` objects; just more of them).

    Pass-through for short parts: when ADK is streaming naturally (small
    chunks at 50-100ms intervals), adding our own delay would compound and
    feel sluggish. ``_PASSTHROUGH`` is the boundary above which re-chunking
    is worth the slight extra latency.

    Cancellation: ``asyncio.sleep`` propagates ``CancelledError`` cleanly,
    so a user clicking Stop interrupts mid-chunk just like before.
    """

    _PASSTHROUGH = 120   # parts this size or smaller emit as-is, no delay
    _TARGET_LEN = 80     # split larger parts at roughly this width
    _SLACK = 20          # accept a word boundary within ±SLACK of the target
    _DELAY_S = 0.035     # 35 ms between chunks — perceptible motion, not slow

    @classmethod
    async def stream(cls, text: str) -> AsyncIterator[str]:
        """Yield text in chunks suitable for ``TokenEvent.text``. Short
        text passes through with zero delay; long text is split at word
        boundaries with a 35 ms sleep between yields."""
        if not text:
            return
        if len(text) <= cls._PASSTHROUGH:
            yield text
            return
        remaining = text
        first = True
        while remaining:
            chunk, remaining = cls._split_one(remaining)
            if not first:
                await asyncio.sleep(cls._DELAY_S)
            first = False
            yield chunk

    @classmethod
    def _split_one(cls, text: str) -> tuple[str, str]:
        """Take ~``_TARGET_LEN`` chars at the nearest word boundary.
        Returns ``(chunk, remainder)``. Never breaks inside a word when
        a sensible boundary exists within ±``_SLACK``."""
        if len(text) <= cls._TARGET_LEN:
            return text, ""
        # Look for a space within target ± SLACK
        lo = max(1, cls._TARGET_LEN - cls._SLACK)
        hi = min(len(text), cls._TARGET_LEN + cls._SLACK)
        cut = text.rfind(" ", lo, hi)
        if cut == -1:
            # No space in window — fall back to a hard cut at TARGET_LEN
            cut = cls._TARGET_LEN
        else:
            cut += 1  # include the space in the chunk
        return text[:cut], text[cut:]


def _initial_plan_thought(user_msg: str) -> str:
    """Honest, safe opening thought emitted before any tool fires.

    Phrasings never name a specific tool or commit to a path — the LLM
    might pick differently. They're tuned to match the cadence of the
    mock's plan beats while remaining accurate regardless of which tool
    the agent actually chooses.
    """
    if not user_msg:
        return "Let me work on this."
    # Pad with spaces so leading-position keywords (e.g. "RSI for X" → " rsi ")
    # match the same way they do in mid-sentence position. Cheap, robust.
    m = " " + user_msg.lower() + " "
    # Multi-company comparison
    if any(k in m for k in (" vs ", " vs. ", " compare ", " versus ")):
        return "Let me pull data on each company so I can compare them side by side."
    # Filings narrative
    if any(k in m for k in (
        "filing", "annual report", "disclos", " agm ", "board meeting",
        "md&a", "concall", "transcript",
    )):
        return "Let me check the latest filings for that."
    # Live market data
    if any(k in m for k in (
        "price", " rsi ", "moving average", " macd ", "trading at", "52-week", "52 week",
    )):
        return "Let me pull the live market data."
    # Numbers / financials
    if any(k in m for k in (
        "profit", "revenue", "ebitda", "margin", "ratio", "cagr",
        " yoy ", "growth", "earnings", " pat ", " pbt ", "balance sheet",
        "cash flow", "debt", "shareholding", "holding",
    )):
        return "Let me pull the financial data for that."
    # Business model
    if any(k in m for k in (" bmc ", "business model", "canvas")):
        return "Let me load the business model canvas."
    # Sector
    if any(k in m for k in ("sector", "industry", " top ", " bottom ", "ranking", "peers")):
        return "Let me check what's in that sector."
    return "Let me work on this."


# Columns that are metadata, not chartable values, on financials_query rows.
# Used by ``_try_emit_chart`` to filter down to a single numeric series.
_NON_CHART_COLUMNS = frozenset({
    "period_end", "period_type", "company_id", "company_name", "ticker",
    "exchange", "industry", "industry_group", "sector",
    "sid", "line_code", "line_path", "statement_type", "taxonomy",
    "view", "fiscal_period", "as_of", "source", "id",
})


def _try_emit_chart(
    tool_name: str, call_id: str, response: Any,
) -> ChartEvent | None:
    """Auto-build a ChartEvent from a tool response that holds a time-series.

    Conservative on purpose — fires ONLY when every signal lines up:
      * tool is ``financials_query`` (the only tool that reliably returns
        date-anchored rows today)
      * response has ``rows`` (list, >=3 entries)
      * every row has a string ``period_end`` field
      * exactly one numeric column appears in every row and isn't a meta
        column (period_end / company_id / sid / etc.)

    Returns ``None`` when any check fails. Rather a dropped chart than a
    wrong/ugly one. Multi-series and bar-style charts could be added later
    without changing the call site.
    """
    if tool_name != "financials_query":
        return None
    if not isinstance(response, dict):
        return None
    rows = response.get("rows")
    if not isinstance(rows, list) or len(rows) < 3:
        return None
    if not all(
        isinstance(r, dict) and isinstance(r.get("period_end"), str)
        for r in rows
    ):
        return None

    # Find numeric columns present in EVERY row, excluding meta fields.
    first_row = rows[0]
    candidate_cols: list[str] = []
    for col, val in first_row.items():
        if col in _NON_CHART_COLUMNS:
            continue
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        if all(
            isinstance(r.get(col), (int, float)) and not isinstance(r.get(col), bool)
            for r in rows
        ):
            candidate_cols.append(col)

    if not candidate_cols:
        return None
    # Multiple series → pick deterministically (alphabetical) for now.
    # Future: detect "value" / "amount" preferred columns, or emit one chart per series.
    column = sorted(candidate_cols)[0]

    sorted_rows = sorted(rows, key=lambda r: r["period_end"])
    points = [
        ChartPoint(x=str(r["period_end"]), y=float(r[column]))
        for r in sorted_rows
    ]
    first_y = points[0].y
    last_y = points[-1].y
    if first_y != 0:
        delta_pct = (last_y - first_y) / abs(first_y) * 100
    else:
        delta_pct = 0.0
    if delta_pct > 0.5:
        delta_kind: Any = "pos"
        delta_str = f"+{delta_pct:.1f}% over period"
    elif delta_pct < -0.5:
        delta_kind = "neg"
        delta_str = f"{delta_pct:.1f}% over period"
    else:
        delta_kind = "neutral"
        delta_str = "flat over period"

    title = column.replace("_", " ").strip().title()
    # Compact display formatting: don't strip zeros from values like "0.44".
    if abs(last_y) >= 100:
        current_value = f"{last_y:,.0f}"
    else:
        current_value = f"{last_y:,.2f}"

    return ChartEvent(
        call_id=call_id,
        chart_id=f"financials_{column}",
        title=title,
        unit="",
        current_value=current_value,
        current_delta=delta_str,
        delta_kind=delta_kind,
        points=points,
        kind="line",
    )


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
