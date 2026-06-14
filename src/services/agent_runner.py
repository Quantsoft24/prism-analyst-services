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

from sqlalchemy import false, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.base import PrismAgent
from src.auth.principal import ANONYMOUS_FIRM
from src.config import settings
from src.core.agent_context import current_firm_id
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
    Citation,
    ClarificationEvent,
    ClarificationOption,
    ClarificationQuestion,
    DataFreshnessEvent,
    ErrorEvent,
    FinalAnswer,
    FinalEvent,
    MetaEvent,
    PlanEvent,
    PlanStep,
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
    | ClarificationEvent
    | PlanEvent
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
        # Task checklist. The agent declares the step TITLES once via update_plan;
        # the RUNNER owns the statuses and advances them deterministically as each
        # tool actually completes (``_gather_count`` = successful tool results so
        # far → how many steps are done). This keeps the checklist in lockstep with
        # real work instead of the model's unreliable, laggy update_plan calls.
        self._last_plan_steps: list[dict[str, Any]] | None = None
        # Runner-synthesized checklist (the fallback when the agent doesn't declare
        # one via update_plan) — built from the tool sequence so a multi-step turn
        # ALWAYS shows a synced checklist. Agent-declared titles take precedence.
        self._auto_plan_steps: list[dict[str, Any]] = []
        self._gather_count: int = 0
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
        # Make this run's firm_id available to per-tenant tools (e.g. BMC, which
        # persists by firm_id) without exposing it to the LLM. Async tools awaited
        # inline in this run inherit the ContextVar. See core/agent_context.py.
        current_firm_id.set(self._firm_id)
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
            session = await sess_svc.create_session(
                app_name="prism",
                user_id=self._firm_id,
                session_id=self._session_id,
            )
            # A resumed conversation whose in-memory ADK session was lost (server
            # restart / a different worker / cache eviction): seed the fresh
            # session with this conversation's persisted transcript so the agent
            # continues with full prior context — nothing lost. No-op for a
            # genuinely new chat (no prior turns). See `_rehydrate_session`.
            await self._rehydrate_session(sess_svc, session)

        new_message = genai_types.Content(
            role="user", parts=[genai_types.Part(text=user_message)]
        )

        # Track tool calls so we can match results back to their calls.
        pending_calls: dict[str, tuple[str, float]] = {}  # call_id -> (tool_name, start_ts)
        final_text_parts: list[str] = []
        final_seen = False
        # Set if the model stops on its output-token cap (finish_reason=MAX_TOKENS)
        # → the answer is cut off and the UI offers "Continue generating".
        truncated = False
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
        # Agentic clarification: when a tool (request_clarification) asks the user
        # to disambiguate, we terminate the turn with a ClarificationEvent and
        # await their selection on the next turn. Holds the pending payload.
        clarification_payload: dict[str, Any] | None = None
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
                    # `update_plan` is a META tool (the visible task checklist) —
                    # no tool card, no evidence; its PlanEvent is emitted on the
                    # response below.
                    if tool_name == "update_plan":
                        pending_calls[call_id] = (tool_name, time.perf_counter())
                        continue
                    pending_calls[call_id] = (tool_name, time.perf_counter())
                    self._tool_trace.append({"tool": tool_name, "args": args, "call_id": call_id})
                    yield ToolCallEvent(tool=tool_name, args=args, call_id=call_id)
                    # Fallback checklist: if the agent didn't declare a plan, build
                    # one from the tool sequence (deduped friendly titles) so a
                    # multi-step turn always shows a synced checklist. The 2nd
                    # distinct phase is when it first renders.
                    if not self._last_plan_steps:
                        title = _plan_step_title(tool_name)
                        if not self._auto_plan_steps or self._auto_plan_steps[-1]["title"] != title:
                            self._auto_plan_steps.append(
                                {"id": f"a{len(self._auto_plan_steps)}", "title": title}
                            )
                        evt = self._build_plan_event(self._gather_count)
                        if evt is not None:
                            yield evt
                    continue

                fn_resp = getattr(part, "function_response", None)
                if fn_resp is not None:
                    call_id = getattr(fn_resp, "id", None) or ""
                    tool_name = getattr(fn_resp, "name", "unknown")
                    response = getattr(fn_resp, "response", None) or {}
                    started = pending_calls.pop(call_id, (tool_name, time.perf_counter()))[1]
                    latency_ms = int((time.perf_counter() - started) * 1000)

                    # `update_plan` → the agent DECLARES the step titles; the
                    # runner renders statuses at the current progress (so even a
                    # late declaration reflects tools already finished). No card.
                    if tool_name == "update_plan":
                        if isinstance(response, dict) and "_plan" in response:
                            raw_steps = response["_plan"].get("steps") or []
                            # Accept the agent's declared plan ONLY if it's a real
                            # multi-step plan (>=2) AND the runner's fallback hasn't
                            # already started — otherwise a lazy, late 1-step
                            # "plan" would replace the better synced fallback.
                            if len(raw_steps) >= 2 and len(self._auto_plan_steps) < 2:
                                self._last_plan_steps = raw_steps
                                evt = self._build_plan_event(self._gather_count)
                                if evt is not None:
                                    yield evt
                        continue

                    # Agentic clarification → terminate the turn with a
                    # ClarificationEvent and await the user's pick next turn (no
                    # synthesis/rescue — there's no answer to compose). Two
                    # sources, both deterministic (no LLM re-narration of ids):
                    #   1. ``request_clarification`` returns a ``_clarification``
                    #      payload (agent-composed, any format).
                    #   2. ``resolve_company`` returns ``needs_clarification`` +
                    #      a structured ``clarification`` block (options dict) —
                    #      we surface it even if the model forgets to ask. (A
                    #      prose ``clarification`` STRING, e.g. financials_query,
                    #      is NOT a dict → handled by the agent, not here.)
                    _clar = None
                    if isinstance(response, dict):
                        if "_clarification" in response:
                            _clar = response["_clarification"]
                        elif response.get("needs_clarification") and isinstance(
                            response.get("clarification"), dict
                        ):
                            _clar = response["clarification"]
                    if _clar is not None:
                        yield ToolResultEvent(
                            call_id=call_id, tool=tool_name, ok=True,
                            result_summary="awaiting your selection", latency_ms=latency_ms,
                        )
                        clarification_payload = _clar
                        final_seen = True
                        break  # exit the parts loop

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
                    # Runner-driven checklist: a tool just finished → advance the
                    # plan one step, IN SYNC with the real work (not the model's
                    # discretionary, laggy update_plan calls).
                    self._gather_count += 1
                    evt = self._build_plan_event(self._gather_count)
                    if evt is not None:
                        yield evt
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

            # Circuit-breaker or clarification fired inside the parts loop —
            # stop iterating ADK events entirely.
            if _circuit_broken or clarification_payload is not None:
                final_seen = True
                break

            # ── Detect end-of-turn ──
            check = getattr(event, "is_final_response", None)
            if callable(check):
                try:
                    if check():
                        final_seen = True
                        # finish_reason=MAX_TOKENS → the answer was cut off at the
                        # output cap. The enum stringifies as "FinishReason.MAX_TOKENS"
                        # (name/value "MAX_TOKENS"); match defensively across forms.
                        fr = getattr(event, "finish_reason", None)
                        if fr is not None and "MAX_TOKENS" in str(getattr(fr, "name", fr)):
                            truncated = True
                except Exception:
                    pass

        # Agentic clarification terminated the turn — emit the structured
        # question(s) and stop. No prose/synthesis (the user must answer first);
        # their selection arrives as the next message in this session.
        if clarification_payload is not None:
            event = _build_clarification_event(self._agent_run_id, clarification_payload)
            yield event
            await self._close_run_row(
                status="awaiting_clarification",
                final_answer=event.question,
                result_payload={
                    "structured": None,
                    "plan": [
                        s.model_dump()
                        for s in (self._build_plan_event(self._gather_count) or PlanEvent(steps=[])).steps
                    ],
                    "clarification": event.model_dump(mode="json"),
                },
            )
            return

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
        # query..." but the turn already ended without firing another tool.
        # Treat it as empty so the composer / fallback writes a real answer.
        if prose and _is_stall_response(prose):
            logger.warning(
                "Agent run %s emitted stall prose (%d chars, no substantive "
                "content); routing to the quality-tier composer.",
                self._agent_run_id,
                len(prose),
            )
            prose = ""

        # BMC COLD MISS — a bmc_get found no saved canvas and none was generated.
        # OVERRIDE the answer (incl. the agent's own freeform "could not be found —
        # want me to generate?" prose, which dead-ends) with a deterministic honest
        # handoff; the UI's "Open full canvas" card routes the user to /bmc to
        # generate it there. Fires ONLY on a true cold miss — `_bmc_cold_miss_message`
        # returns None when a canvas exists or other real evidence was gathered, so
        # successful/cache-hit turns are untouched.
        bmc_miss = _bmc_cold_miss_message(self._tool_trace)
        if bmc_miss:
            prose = bmc_miss
            structured = FinalAnswer(text=prose)

        # GATHER-RESCUE (deterministic) — a non-clarification turn that ended with
        # NO data gathered AND no answer means the orchestrator stopped after a
        # routing tool without finishing (free-tier Flash flakiness). Re-invoke
        # ONCE with an explicit gather nudge, then re-read the prose/structured.
        # This does NOT depend on the first pass getting it right; worst case is
        # the same generic fallback as before. See `_gather_rescue_pass`.
        if (
            clarification_payload is None
            and not _circuit_broken
            and not prose
            and not _has_substantive_evidence(self._tool_trace)
        ):
            async for ev in self._gather_rescue_pass(
                runner, genai_types, final_text_parts, observed_freshness
            ):
                yield ev
            raw_final = "".join(final_text_parts).strip()
            prose, structured = _split_structured_answer(raw_final)
            structured = _validate_structured_freshness(structured, observed_freshness)
            if prose and _is_stall_response(prose):
                prose = ""

        # BUSINESS MODEL CANVAS — render node-by-node DETERMINISTICALLY (bold block
        # title + cited bullets, all nine blocks), bypassing the composer. The
        # canvas is a structured artifact; letting the LLM "summarize" it is what
        # caused dropped blocks + mangled citations. This is faithful + reliable.
        bmc_canvas = _first_bmc_canvas(self._tool_trace)
        if bmc_canvas is not None:
            prose = _render_bmc_answer(bmc_canvas)
            structured = FinalAnswer(text=prose, suggestions=_bmc_followups(bmc_canvas))

        # PRIMARY answer path — when the turn gathered substantive evidence, the
        # AUTHORITATIVE answer is composed on the quality tier from that evidence
        # (the fast orchestrator is only a planner/gatherer; flash-lite is too
        # weak to reliably write the final answer, and thinking-mode often ends
        # the turn without it). Trivial turns (no data tools) keep the fast
        # model's prose. Clarification turns already returned above.
        elif _has_substantive_evidence(self._tool_trace):
            composed = await _compose_final_answer(user_message, self._tool_trace)
            if composed:
                composed, suggestions = _extract_follow_ups(composed)
                prose = composed
                base = (
                    structured.model_copy(update={"text": composed})
                    if structured is not None
                    else FinalAnswer(text=composed)
                )
                structured = (
                    base.model_copy(update={"suggestions": suggestions})
                    if suggestions
                    else base
                )
                logger.info(
                    "Agent run %s: composed final answer on the quality tier "
                    "(%d chars, %d follow-ups).",
                    self._agent_run_id, len(composed), len(suggestions),
                )

        # Deterministic filing citations: stock_filings_read evidence carries the
        # exact ``pdf_url`` + ``page`` per cited passage. Parse the (now composed)
        # prose for `[Company | p.N]` and attach trustworthy, page-exact Citations
        # the UI deep-links to — rather than trusting the LLM to transcribe URLs.
        structured = _merge_filing_citations(structured, self._tool_trace)

        # Safety net for the "still no prose" case — the quality composer ran
        # above for substantive turns; this catches (a) trivial turns where the
        # fast model emitted nothing, and (b) the rare case the composer itself
        # failed (router off + no key, total provider outage). Layered, most-to-
        # least useful:
        #   1. structured.sections[0].body  → the answer was in the meta body.
        #   2. structured.citations exist   → point to the Report/Sources tabs.
        #   3. one more composer attempt    → in case it wasn't tried (no
        #      substantive evidence flagged but tools did run).
        #   4. generic deterministic message → guarantee the user sees SOMETHING.
        if not prose:
            if structured is not None and structured.sections:
                first_body = (structured.sections[0].body or "").strip()
                if first_body:
                    prose = first_body
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
            if not prose and not _has_substantive_evidence(self._tool_trace):
                # Tools ran but none were flagged substantive (edge case) — try
                # composing once before the generic message.
                composed = await _compose_final_answer(user_message, self._tool_trace)
                if composed:
                    prose = composed
                    structured = (
                        structured.model_copy(update={"text": composed})
                        if structured is not None
                        else FinalAnswer(text=composed)
                    )
            if not prose:
                # Last-resort deterministic message — guarantees the user always
                # sees SOMETHING when every compose path failed.
                prose = _synthesize_empty_answer_fallback(self._tool_trace)
                structured = None
                logger.warning(
                    "Agent run %s: composer returned empty too; surfacing "
                    "generic fallback. %d tools called.",
                    self._agent_run_id,
                    len(self._tool_trace),
                )

        cost = _estimate_cost_usd(self._agent.model, self._input_tokens, self._output_tokens)
        latency_ms = int((time.perf_counter() - self._started_at) * 1000)

        # Final checklist state (every step done — including any trailing "compose"
        # step that had no tool of its own). Computed once for both the live
        # PlanEvent and the persisted replay payload (agent OR runner-built plan).
        done_plan = self._build_plan_event(len(self._active_plan() or []))
        done_steps = [s.model_dump() for s in done_plan.steps] if done_plan is not None else []

        await self._close_run_row(
            status="complete",
            final_answer=prose,
            cost_usd=cost,
            result_payload={
                # Persist the SAME structured payload the UI rendered live so a
                # reopened conversation replays citations / confidence / freshness
                # / sources / follow-ups — not just the prose.
                "structured": (
                    structured.model_dump(mode="json") if structured is not None else None
                ),
                "plan": done_steps,
                "clarification": None,
            },
        )

        if done_plan is not None:
            yield done_plan

        yield FinalEvent(
            answer=prose,
            structured=structured,
            agent_run_id=self._agent_run_id,  # type: ignore[arg-type]
            cost_usd=cost,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            latency_ms=latency_ms,
            truncated=truncated,
        )

    # ── Runner-driven task checklist ─────────────────────────────────────────

    def _active_plan(self) -> list[dict[str, Any]] | None:
        """The checklist steps to render: the agent's declared titles if it gave
        any, else the runner-synthesized fallback (only once it has ≥2 distinct
        phases — a single tool isn't worth a checklist)."""
        if self._last_plan_steps:
            return self._last_plan_steps
        if len(self._auto_plan_steps) >= 2:
            return self._auto_plan_steps
        return None

    def _build_plan_event(self, progress: int) -> "PlanEvent | None":
        """Render the checklist at a given progress (# of completed steps).

        Step TITLES come from the agent (update_plan) or, as a fallback, the
        runner's tool-derived titles. The runner owns the STATUSES — steps
        ``[0, progress)`` are ``done``, step ``progress`` is ``in_progress`` (the
        one being worked), the rest ``pending``. ``progress`` is driven by real
        tool completion (``_gather_count``), so the checklist ticks off in sync
        with the work, not the model's whims.
        """
        steps = self._active_plan()
        if not steps:
            return None
        n = len(steps)
        p = max(0, min(progress, n))
        rendered = []
        for i, s in enumerate(steps):
            status = "done" if i < p else ("in_progress" if i == p else "pending")
            rendered.append(PlanStep(id=str(s.get("id") or f"s{i}"), title=str(s.get("title") or ""), status=status))
        return PlanEvent(steps=rendered)

    # ── Deterministic gather-rescue ──────────────────────────────────────────

    async def _gather_rescue_pass(
        self,
        runner: Any,
        genai_types: Any,
        text_sink: list[str],
        observed_freshness: set[str],
    ) -> AsyncIterator[ChatEvent]:
        """Recover a turn the orchestrator dropped before gathering any data.

        The free-tier Flash orchestrator intermittently ends a turn right after
        a *routing* tool (``resolve_company`` / ``update_plan``) without calling
        the data tool that actually answers the question — so the turn reaches
        the post-loop with no substantive evidence and no prose, and the user
        gets the generic "couldn't put together an answer" fallback. This is
        model flakiness, not a missing capability (the same query succeeds on a
        retry), so we fix it deterministically instead of via the prompt: when
        that exact state is detected, re-invoke the agent ONCE in the SAME
        session with an explicit nudge to finish the job. The session already
        holds the user's question + any resolved ``security_id``(s); tool
        calls/results flow back into ``self._tool_trace`` so the existing
        quality-tier composer writes the answer from the freshly-gathered
        evidence. Worst case (the nudge also gathers nothing) is identical to
        today's generic fallback — so this is strictly an improvement.
        """
        nudge = (
            "[system] Your previous step ended WITHOUT calling any data tool, so "
            "there is no answer yet. You ALREADY have everything you need — the "
            "user's question and the resolved security_id(s) are in the "
            "conversation above. Do NOT ask for clarification again and do NOT "
            "re-resolve the companies. Call the right data tool now "
            "(stock_filings_read / financials_query / stock_technicals / news_* / "
            "bmc_*) with those id(s), then write the complete answer."
        )
        msg = genai_types.Content(role="user", parts=[genai_types.Part(text=nudge)])
        pending: dict[str, float] = {}
        logger.warning(
            "Agent run %s: gather-rescue — orchestrator stopped before gathering; "
            "re-invoking with an explicit gather nudge.",
            self._agent_run_id,
        )
        try:
            async for event in runner.run_async(
                user_id=self._firm_id,
                session_id=self._session_id,
                new_message=msg,
            ):
                usage = getattr(event, "usage_metadata", None)
                if usage:
                    self._input_tokens = (
                        getattr(usage, "prompt_token_count", self._input_tokens)
                        or self._input_tokens
                    )
                    self._output_tokens = (
                        getattr(usage, "candidates_token_count", self._output_tokens)
                        or self._output_tokens
                    )
                content = getattr(event, "content", None)
                for part in getattr(content, "parts", None) or []:
                    fn_call = getattr(part, "function_call", None)
                    if fn_call is not None:
                        call_id = getattr(fn_call, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                        tool_name = getattr(fn_call, "name", "unknown")
                        args = dict(getattr(fn_call, "args", {}) or {})
                        if tool_name == "update_plan":
                            pending[call_id] = time.perf_counter()
                            continue
                        pending[call_id] = time.perf_counter()
                        self._tool_trace.append(
                            {"tool": tool_name, "args": args, "call_id": call_id}
                        )
                        yield ToolCallEvent(tool=tool_name, args=args, call_id=call_id)
                        continue

                    fn_resp = getattr(part, "function_response", None)
                    if fn_resp is not None:
                        call_id = getattr(fn_resp, "id", None) or ""
                        tool_name = getattr(fn_resp, "name", "unknown")
                        response = getattr(fn_resp, "response", None) or {}
                        started = pending.pop(call_id, time.perf_counter())
                        latency_ms = int((time.perf_counter() - started) * 1000)
                        if tool_name == "update_plan":
                            if isinstance(response, dict) and "_plan" in response:
                                raw_steps = response["_plan"].get("steps") or []
                                if len(raw_steps) >= 2 and len(self._auto_plan_steps) < 2:
                                    self._last_plan_steps = raw_steps
                                    evt = self._build_plan_event(self._gather_count)
                                    if evt is not None:
                                        yield evt
                            continue
                        # A re-fired clarification in the rescue is ignored (the
                        # nudge forbids it); treat it as a normal result so the
                        # turn still terminates with the generic fallback if the
                        # model insists rather than looping.
                        if is_error(response):
                            yield ToolResultEvent(
                                call_id=call_id, tool=tool_name, ok=False,
                                error=extract_error_message(response) or "tool error",
                                error_code=(
                                    response.get("error_code")
                                    if isinstance(response, dict) else None
                                ),
                                next_action=(
                                    response.get("next_action")
                                    if isinstance(response, dict) else None
                                ),
                                latency_ms=latency_ms,
                            )
                            continue
                        for entry in self._tool_trace:
                            if entry.get("call_id") == call_id:
                                entry["response"] = _trim_response_for_rescue(response)
                                break
                        yield ToolResultEvent(
                            call_id=call_id, tool=tool_name, ok=True,
                            result_summary=_summarize_tool_response(response),
                            latency_ms=latency_ms,
                        )
                        if isinstance(response, dict):
                            freshness = response.get("data_freshness")
                            if freshness:
                                observed_freshness.add(str(freshness))
                                yield DataFreshnessEvent(
                                    call_id=call_id,
                                    source=_freshness_source_label(tool_name),
                                    as_of=str(freshness),
                                )
                        # Keep the checklist advancing during the rescue pass too.
                        self._gather_count += 1
                        evt = self._build_plan_event(self._gather_count)
                        if evt is not None:
                            yield evt
                        continue

                    text = getattr(part, "text", None)
                    if text and not bool(getattr(part, "thought", False)):
                        text_sink.append(text)
                        async for chunk in _TokenChunker.stream(text):
                            yield TokenEvent(text=chunk)
        except Exception:
            logger.exception(
                "Agent run %s: gather-rescue pass errored; falling through to the "
                "existing fallback chain.",
                self._agent_run_id,
            )

    # ── Session rehydration (context continuity on resume) ──────────────────

    def _scope_clause(self):
        """Principal scope, mirroring ``conversation_repo._scope``: signed-in by
        user_id, guest by client_key, dev/no-auth by firm_id. Guards a resumed
        session from ever reading another principal's history."""
        if self._user_id is not None:
            return AgentRun.user_id == self._user_id
        if self._firm_id == ANONYMOUS_FIRM:
            return AgentRun.client_key == self._client_key if self._client_key else false()
        return AgentRun.firm_id == self._firm_id

    async def _load_prior_turns(self) -> list[tuple[str, str]]:
        """Completed ``(user_input, final_answer)`` turns already persisted for
        this session_id — oldest → newest, scoped to the caller, excluding the
        current in-flight run (which has no final_answer yet)."""
        stmt = (
            select(AgentRun.user_input, AgentRun.final_answer)
            .where(
                AgentRun.session_id == self._session_id,
                AgentRun.hidden_at.is_(None),
                AgentRun.final_answer.is_not(None),
                self._scope_clause(),
            )
            .order_by(AgentRun.created_at.asc())
        )
        if self._agent_run_id is not None:
            stmt = stmt.where(AgentRun.id != self._agent_run_id)
        async with session_scope() as session:
            rows = (await session.execute(stmt)).all()
        return [(r.user_input, r.final_answer) for r in rows if r.user_input and r.final_answer]

    async def _rehydrate_session(self, sess_svc: Any, session: Any) -> None:
        """Seed a freshly-created ADK session with this conversation's persisted
        transcript so a resumed chat continues with full prior context even when
        the in-memory session was lost (restart / different worker). No-op for a
        genuinely new conversation (no prior completed turns)."""
        from google.adk.events import Event
        from google.genai import types as genai_types

        turns = await self._load_prior_turns()
        if not turns:
            return
        for user_input, final_answer in turns:
            await sess_svc.append_event(
                session,
                Event(
                    author="user",
                    content=genai_types.Content(
                        role="user", parts=[genai_types.Part(text=user_input)]
                    ),
                ),
            )
            await sess_svc.append_event(
                session,
                Event(
                    author=self._agent.name,
                    content=genai_types.Content(
                        role="model", parts=[genai_types.Part(text=final_answer)]
                    ),
                ),
            )
        logger.info(
            "prism: rehydrated session %s with %d prior turn(s) of context",
            self._session_id,
            len(turns),
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
        result_payload: dict[str, Any] | None = None,
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
                result_payload=result_payload,
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
        result_payload: dict[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "status": status,
            "final_answer": final_answer,
            "cost_usd": cost_usd,
            "error_code": error_code,
            "error_message": error_message,
            "latency_ms": latency_ms,
            "tool_trace": self._tool_trace,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
        }
        # Only overwrite result_payload when we have one — never clobber a stored
        # payload with null on an error/intermediate close.
        if result_payload is not None:
            values["result_payload"] = result_payload
        stmt = update(AgentRun).where(AgentRun.id == self._agent_run_id).values(**values)
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


# System prompt for the quality-tier final-answer composer. The fast
# orchestrator gathers evidence; THIS writes the user-facing answer.
_COMPOSER_SYSTEM_PROMPT = """You are PRISM, an expert equity-research analyst writing \
the FINAL answer for the user, using ONLY the tool evidence provided. Write the best, \
most COMPLETE, decision-useful answer to the user's underlying question.

How to write it:
- LEAD with the direct answer — the key finding, number, or takeaway — in the first \
sentence. No preamble ("Based on the evidence…"), no restating the question.
- Be COMPLETE: address EVERY part of the question (each period, metric, topic, and \
company). Include every material fact, figure, and date the evidence supports — do \
NOT omit or over-compress. Do NOT invent anything not in the evidence.
- STRUCTURE like an analyst note: a tight lead paragraph, then bullets for the \
specifics. Scannable, concrete, no fluff. The chat renders Markdown (incl. tables).
- HONOR the user's requested format: "in a table" → a Markdown table (NOT bullets); \
"in short" → 1-2 lines; etc. For a multi-company / multi-metric COMPARISON, default \
to a Markdown table (one row per company, one column per metric) — clearer than \
parallel bullet lists — led by a one-sentence takeaway.
- CITE filings, don't fabricate pages: when the evidence provides a \
'[Company | p.N]' citation string (filing passages do), preserve it VERBATIM next to \
the fact — these are clickable PDF deep-links, so never alter, merge, drop, or INVENT \
them. For data that has NO such string (e.g. financial figures from the database), \
just state the figure with its period/date — do NOT make up a '[… | p.N]' citation.
- PARTIAL / THIN evidence: give everything useful the evidence DOES contain, state \
plainly what is missing, and suggest the most useful next step (e.g. "I can pull the \
full annual report", or ask for a specific quarter/period). NEVER dead-end with only \
"content not available" when ANY useful detail exists.
- If the user's latest message is only a company selection (e.g. "<Company> — \
security_id N"), answer the QUESTION implied by the tool calls — never treat the \
selection text as the question.

After the answer, on its OWN LAST LINE, emit 2-3 short, specific follow-up \
questions the user might naturally ask next that OUR tools can answer (a metric, \
a period, a peer, the filing detail), formatted EXACTLY as:
FOLLOW_UPS: <question 1> | <question 2> | <question 3>
Put nothing after that line. Do NOT call tools and do NOT emit any <answer_meta> block."""


# Tools that GATHER substantive evidence (vs. routing/disambiguation helpers).
# A turn that ran any of these gets the quality-tier composer; trivial turns
# (acknowledgements, a lone resolve/clarify) keep the fast model's text.
_NON_SUBSTANTIVE_TOOLS = frozenset(
    {"resolve_company", "resolve_companies", "search_companies", "list_sectors",
     "request_clarification", "update_plan"}
)


def _has_substantive_evidence(tool_trace: list[dict[str, Any]]) -> bool:
    """True if the turn gathered real data (a non-routing tool returned)."""
    return any(
        e.get("tool") not in _NON_SUBSTANTIVE_TOOLS and e.get("response") is not None
        for e in tool_trace
    )


def _bmc_cold_miss_message(tool_trace: list[dict[str, Any]]) -> str | None:
    """Honest message for a Business Model Canvas COLD MISS: a ``bmc_get`` found
    no saved canvas and none was generated this turn (the free-tier orchestrator
    sometimes stops after the 404 instead of calling ``bmc_generate``).

    Rather than block the chat turn on a 30-60s in-chat generate (slow + brushes
    the agent timeout), we return a clear message; the UI's "Open full canvas"
    card routes the user to ``/bmc`` to generate it there (proper progress UI).

    Returns ``None`` when a canvas EXISTS (the agent did generate / it was cached)
    or the turn ALSO gathered real non-BMC evidence — those go to the composer."""
    bmc_ticker: str | None = None
    for t in tool_trace:
        tool = str(t.get("tool", ""))
        if not tool.startswith("bmc_"):
            continue
        resp = t.get("response")
        if isinstance(resp, dict) and resp.get("blocks"):
            return None  # a real canvas came back — not a cold miss
        if tool == "bmc_get" and isinstance(t.get("args"), dict):
            bmc_ticker = t["args"].get("ticker") or bmc_ticker
    if not bmc_ticker:
        return None
    # Don't override a turn that gathered real non-BMC data.
    if _has_substantive_evidence(
        [t for t in tool_trace if not str(t.get("tool", "")).startswith("bmc_")]
    ):
        return None
    return (
        f"There's no saved Business Model Canvas for **{bmc_ticker}** yet. "
        "Click **Open full canvas** below to generate one — it builds a "
        "filing-grounded 9-block canvas (~30–60s) with clickable citations."
    )


def _strip_answer_meta(text: str) -> str:
    """Drop any accidental ``<answer_meta>`` tail + surrounding whitespace."""
    text = (text or "").strip()
    meta_at = text.find("<answer_meta>")
    return text[:meta_at].rstrip() if meta_at >= 0 else text


def _extract_follow_ups(text: str) -> tuple[str, list[str]]:
    """Split a trailing ``FOLLOW_UPS: a | b | c`` line off the composed answer →
    (clean prose, up to 3 suggestions)."""
    m = re.search(r"(?im)^\s*FOLLOW[_ ]?UPS:\s*(.+?)\s*$", text or "")
    if not m:
        return (text or "").strip(), []
    sugg = [s.strip(" -•*") for s in m.group(1).split("|") if s.strip(" -•*")][:3]
    return text[: m.start()].rstrip(), sugg


def _clar_question(payload: dict[str, Any], default_id: str) -> ClarificationQuestion:
    """Build one ClarificationQuestion from a payload dict."""
    return ClarificationQuestion(
        id=str(payload.get("id") or default_id),
        question=payload.get("question", "Could you clarify?"),
        mode=payload.get("mode", "single_select"),
        options=[ClarificationOption(**o) for o in (payload.get("options") or [])],
        allow_search=payload.get("allow_search", True),
    )


def _build_clarification_event(
    agent_run_id: Any, payload: dict[str, Any],
) -> ClarificationEvent:
    """Normalize a clarification payload into a ClarificationEvent. Accepts either
    a single-question payload (``{question, mode, options, allow_search}``) or a
    multi-question one (``{questions: [...]}`` — e.g. from ``resolve_companies``,
    so "Reliance"/"Adani"/"Tata" are disambiguated together in one card). The
    back-compat single fields mirror ``questions[0]``."""
    raw = payload.get("questions")
    if isinstance(raw, list) and raw:
        questions = [_clar_question(q, f"q{i}") for i, q in enumerate(raw)]
    else:
        questions = [_clar_question(payload, "q0")]
    first = questions[0]
    return ClarificationEvent(
        agent_run_id=agent_run_id,
        questions=questions,
        question=first.question,
        mode=first.mode,
        options=first.options,
        allow_search=first.allow_search,
    )


def _looks_like_clarification_pick(message: str) -> bool:
    """The turn message is just a company selection from a clarification MCQ
    (e.g. "Reliance Industries Ltd. — security_id 2228"), not a real question."""
    return bool(re.search(r"security[_ ]?id\s*:?\s*\d", message or "", re.IGNORECASE))


def _tool_questions(tool_trace: list[dict[str, Any]]) -> list[str]:
    """The natural-language research questions the agent issued to data tools this
    turn (the ``question`` arg of stock_filings_read / financials_query / …) — the
    best signal of the user's underlying intent, especially on a pick-reply turn
    where the message is just the company choice. Company-lookup ``query`` args
    (resolve_company / search_companies) are intentionally excluded. Deduped,
    order-preserving."""
    seen: set[str] = set()
    out: list[str] = []
    for e in tool_trace:
        args = e.get("args") or {}
        q = args.get("question")
        if isinstance(q, str) and q.strip() and q not in seen:
            seen.add(q)
            out.append(q.strip())
    return out


async def _compose_final_answer(
    user_message: str,
    tool_trace: list[dict[str, Any]],
) -> str | None:
    """Compose the user-facing answer from gathered tool evidence on the
    **quality tier** (`gemini-2.5-flash`→`gemini-2.5-pro`, or an `openai/*` model
    if configured). This is the PRIMARY answer path for substantive turns — the
    fast orchestrator only plans + calls tools; this writes the authentic,
    complete answer, reliably (a plain no-tools generation, immune to the
    thinking-mode "think-then-stop" that drops the fast model's final text).

    Returns the prose, or ``None`` on any failure (caller falls back). Routes via
    the ModelRouter (multi-key 429 fallback); if the router is unavailable, falls
    back to a direct single-shot call with a raw Gemini key.
    """
    if not tool_trace:
        return None

    # Compact the evidence: last 6 tool results, generous per-result cap so the
    # filing `evidence[].quote` + `[Company | p.N]` survive (the source material
    # the composer must use). `_trim_response_for_rescue` already bounded these.
    parts: list[str] = []
    for entry in tool_trace[-6:]:
        tool = entry.get("tool", "?")
        args = entry.get("args", {})
        response = entry.get("response")
        if response is None:
            continue
        # BMC: the raw 9-block JSON (with every evidence excerpt) blows past the
        # 8000-char cap and truncates to ~5 blocks — so the brief silently drops
        # Revenue Streams / Cost Structure / etc. Feed a COMPACT all-9-block view
        # (titles + bullets + per-bullet page hints, no excerpts) so the composer
        # sees the WHOLE model and can cite single pages.
        if str(tool).startswith("bmc_") and isinstance(response, dict) and response.get("blocks"):
            response_str = _compact_bmc(response)
        else:
            try:
                response_str = json.dumps(response, default=str, ensure_ascii=False)
            except (TypeError, ValueError):
                response_str = repr(response)
            if len(response_str) > 8000:
                response_str = response_str[:8000] + "…(truncated)"
        parts.append(f"{tool}(args={args}) →\n{response_str}")

    if not parts:
        return None

    # Surface the user's underlying question. On a clarification-pick turn the
    # message is just the company choice, so the real intent lives in the tool
    # questions — lead with those so the composer answers the right thing.
    tool_qs = _tool_questions(tool_trace)
    if _looks_like_clarification_pick(user_message) and tool_qs:
        intent_block = (
            f"USER PICKED: {user_message}\n"
            "THE QUESTION TO ANSWER (from the tool calls): "
            + " | ".join(tool_qs)
        )
    else:
        intent_block = f"USER QUESTION: {user_message}"
        if tool_qs:
            intent_block += "\n(tools were asked: " + " | ".join(tool_qs) + ")"

    user_prompt = (
        intent_block
        + "\n\nTOOL EVIDENCE (most recent last):\n\n"
        + "\n\n---\n\n".join(parts)
        + "\n\nWrite the complete analyst answer now, citing each fact with its "
        "[Company | p.N] string verbatim."
    )
    # Business Model Canvas: the full 9-block canvas is rendered separately (the
    # /bmc surface + a handoff card), and the composer's evidence is capped (so a
    # full inline reproduction truncates anyway). Ask for a crisp overview instead.
    if any(str(e.get("tool", "")).startswith("bmc_") for e in tool_trace):
        user_prompt += (
            "\n\nBUSINESS MODEL CANVAS — this OVERRIDES the 'be complete' rule. Do "
            "NOT reproduce all nine blocks (the full interactive canvas is shown "
            "separately). Write a TIGHT, scannable analyst brief in this EXACT shape:\n"
            "  • A one-sentence LEAD: what the company does + how it makes money.\n"
            "  • Then 5-6 Markdown bullets that SYNTHESIZE across ALL the canvas "
            "blocks above (do not just take the first few). COVER the essentials: "
            "customer segments, HOW IT MAKES MONEY (revenue streams), cost drivers, "
            "key resources/moat, and key partnerships/channels. One crisp fact per "
            "bullet, CONCRETE and QUANTIFIED wherever the evidence allows. No filler.\n"
            "MANDATORY CITATIONS — every bullet AND every quantified lead fact MUST "
            "end with a citation marker `[<company_name> | p.<page>]`:\n"
            "  - use the COMPANY NAME from the evidence's evidence[] rows (e.g. "
            "'Reliance Industries Ltd'), NEVER the newsid / filing id;\n"
            "  - exactly ONE page per marker — never combine pages (write two markers "
            "`[X | p.5] [X | p.10]`, never `[X | p.5, p.10]`);\n"
            "  - take the page from the SAME evidence row the fact came from. These "
            "deep-link to the cited PDF page, so an uncited claim is not acceptable here."
        )
    messages = [
        {"role": "system", "content": _COMPOSER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # PRIMARY — quality tier via the router (load-balanced, 429 fallback chain).
    try:
        from src.services.model_router import get_router

        text = _strip_answer_meta(await get_router().acomplete(
            tier="quality", messages=messages, temperature=0.2,
        ))
        if text:
            return text
    except Exception as exc:  # noqa: BLE001 — router off / 429 exhausted / etc.
        logger.warning("Composer via router quality tier failed (%s): %s",
                       type(exc).__name__, exc)

    # FALLBACK — direct single-shot with a raw Gemini key (router disabled in
    # some deploys/tests). Best-effort; ``None`` lets the caller degrade.
    api_key = next((k for k in settings.gemini_api_keys if k), None)
    if not api_key:
        return None
    try:
        import litellm

        resp = await litellm.acompletion(
            model="gemini/gemini-2.5-pro",
            api_key=api_key,
            messages=messages,
            temperature=0.2,
            timeout=25,
        )
        text = _strip_answer_meta(
            (resp.choices[0].message.content or "") if resp.choices else ""
        )
        return text or None
    except Exception as exc:  # noqa: BLE001 — any failure → fall back
        logger.warning("Composer direct fallback failed (%s): %s",
                       type(exc).__name__, exc)
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


def _norm_cite_label(label: str | None) -> str:
    """Comparable key for a citation label — alnum only, lowercased. So
    ``"[ITC Ltd | p.5]"`` and the LLM's ``"ITC Ltd | p.5"`` collide."""
    return re.sub(r"[^a-z0-9]+", "", (label or "").lower())


# stock-chat's inline citation format in the prose answer: ``[Company | p.N]``
# (tolerant of spaced initials like "I T C Ltd." and "p. 5" / "p.5" / "pp. 5").
_INLINE_CITE_RE = re.compile(r"\[([^\]|]+?)\s*\|\s*pp?\.?\s*(\d{1,4})\b", re.IGNORECASE)
# A page number embedded in a citation LABEL (e.g. "ITC Ltd p. 7", "… | p.5").
_PAGE_IN_LABEL_RE = re.compile(r"\bpp?\.?\s*(\d{1,4})\b", re.IGNORECASE)


def _filing_pdf_index(tool_trace: list[dict[str, Any]]) -> dict[str, str]:
    """Map normalized company name → filing ``pdf_link`` from every
    ``stock_filings_read`` result. ``selected_filings`` carries the PDF url in
    BOTH synthesise modes (``evidence`` is empty when ``synthesise=true``, which
    is what the agent uses), so it's the reliable source; ``evidence[].pdf_url``
    is a fallback for the ``synthesise=false`` path."""
    out: dict[str, str] = {}
    for entry in tool_trace:
        if entry.get("tool") != "stock_filings_read":
            continue
        resp = entry.get("response")
        if not isinstance(resp, dict):
            continue
        for f in resp.get("selected_filings") or []:
            if isinstance(f, dict) and f.get("pdf_link") and f.get("company_name"):
                out.setdefault(_norm_cite_label(f["company_name"]), str(f["pdf_link"]))
        for ev in resp.get("evidence") or []:
            if isinstance(ev, dict) and ev.get("pdf_url") and ev.get("company_name"):
                out.setdefault(_norm_cite_label(ev["company_name"]), str(ev["pdf_url"]))
    return out


def _evidence_citations(tool_trace: list[dict[str, Any]]) -> list[Citation]:
    """Build filing citations DIRECTLY from ``stock_filings_read`` evidence — the
    most reliable source. With ``synthesise=false`` (the default) each evidence
    item carries its own ``citation`` string, ``page``, AND ``pdf_url``, so no
    parsing or company-name join is needed. Deduped by (url, page)."""
    out: list[Citation] = []
    seen: set[tuple[str, int | None]] = set()
    for entry in tool_trace:
        if entry.get("tool") != "stock_filings_read":
            continue
        resp = entry.get("response")
        if not isinstance(resp, dict):
            continue
        for ev in resp.get("evidence") or []:
            if not isinstance(ev, dict) or not ev.get("pdf_url"):
                continue
            try:
                page = int(ev.get("page")) if ev.get("page") is not None else None
            except (TypeError, ValueError):
                page = None
            url = str(ev["pdf_url"])
            key = (url, page)
            if key in seen:
                continue
            seen.add(key)
            label = ev.get("citation") or f"{ev.get('company_name', '')} p.{page}".strip()
            out.append(Citation(label=str(label), url=url, source_kind="filing", page=page))
    return out


def _bmc_citations(tool_trace: list[dict[str, Any]]) -> list[Citation]:
    """Build filing citations from BMC block evidence (``bmc_get`` / ``bmc_generate``
    → ``blocks[].evidence[]``), so the composer's ``[Company | p.N]`` markers
    deep-link to the cited PDF page (same UX as stock-filings). Each evidence row
    carries ``company_name``, ``page``, and ``pdf_url`` (with ``#page=N``). The
    label leads with the company name so the inline-marker matcher resolves it.
    Deduped by (url, page)."""
    out: list[Citation] = []
    seen: set[tuple[str, int | None]] = set()
    for entry in tool_trace:
        if not str(entry.get("tool", "")).startswith("bmc_"):
            continue
        resp = entry.get("response")
        if not isinstance(resp, dict):
            continue
        company_top = resp.get("company_name") or ""
        for block in resp.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            for ev in block.get("evidence") or []:
                if not isinstance(ev, dict) or not ev.get("pdf_url"):
                    continue
                try:
                    page = int(ev.get("page")) if ev.get("page") is not None else None
                except (TypeError, ValueError):
                    page = None
                url = str(ev["pdf_url"])
                key = (url, page)
                if key in seen:
                    continue
                seen.add(key)
                company = ev.get("company_name") or company_top or ""
                label = (f"{company} — p.{page}" if page is not None else str(company or "filing")).strip(" —")
                out.append(Citation(label=label, url=url, source_kind="filing", page=page))
    return out


def _compact_bmc(resp: dict[str, Any]) -> str:
    """Compact ALL nine BMC blocks for the composer: company + each block's title,
    summary bullets (with their `[N]` markers rewritten to the cited page), and
    key_insights — but NOT the bulky verbatim excerpts. This keeps the full 9-block
    canvas well under the evidence cap (the raw JSON truncates to ~5 blocks), so
    the brief reflects the WHOLE model (incl. revenue streams + cost structure)."""
    company = resp.get("company_name") or resp.get("ticker") or "the company"
    blocks = resp.get("blocks") or []
    lines = [
        f"Business Model Canvas for {company} — all {len(blocks)} blocks. "
        "Cite each fact as [" + str(company) + " | p.<page>] using the page shown "
        "next to it (one page per marker):"
    ]
    for b in blocks:
        if not isinstance(b, dict):
            continue
        marker_to_page = {
            ev.get("marker"): ev.get("page")
            for ev in (b.get("evidence") or [])
            if isinstance(ev, dict) and ev.get("marker")
        }

        def _pageize(text: str) -> str:
            return re.sub(
                r"\[(\d+)\]",
                lambda m: (f"(p.{marker_to_page['[' + m.group(1) + ']']})"
                           if marker_to_page.get("[" + m.group(1) + "]") is not None
                           else ""),
                text,
            )

        lines.append(f"\n## {b.get('title') or b.get('block_id')}")
        for bl in b.get("summary_bullets") or []:
            lines.append(f"- {_pageize(str(bl))}")
        for ki in b.get("key_insights") or []:
            lines.append(f"  (insight) {ki}")
    out = "\n".join(lines)
    return out[:12000] + "…(truncated)" if len(out) > 12000 else out


def _first_bmc_canvas(tool_trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The first BMC tool response that actually carries a 9-block canvas."""
    for entry in tool_trace:
        if str(entry.get("tool", "")).startswith("bmc_"):
            resp = entry.get("response")
            if isinstance(resp, dict) and resp.get("blocks"):
                return resp
    return None


def _render_bmc_answer(resp: dict[str, Any]) -> str:
    """Render the Business Model Canvas node-by-node, DETERMINISTICALLY: each block
    as a **bold title** followed by its details as bullets, with every bullet's
    ``[N]`` marker rewritten to a clickable ``[Company | p.<page>]`` citation. This
    is faithful to the canvas (all nine blocks, exact facts, exact pages) — the
    composer is bypassed for BMC so it can't summarize, drop blocks, or mangle
    citations."""
    company = resp.get("company_name") or resp.get("ticker") or "Company"
    parts = [
        f"Here's **{company}**'s business model across the nine building blocks, "
        f"built from its own filings — every fact links to its source page.",
    ]
    for b in resp.get("blocks") or []:
        if not isinstance(b, dict):
            continue
        title = b.get("title") or str(b.get("block_id", "")).replace("_", " ").title()
        marker_to_page = {
            ev.get("marker"): ev.get("page")
            for ev in (b.get("evidence") or [])
            if isinstance(ev, dict) and ev.get("marker")
        }

        def _fmt(text: str, _m2p: dict = marker_to_page) -> str:
            # [N] -> clickable [Company | p.<page>]
            text = re.sub(
                r"\[(\d+)\]",
                lambda m: (f"[{company} | p.{_m2p['[' + m.group(1) + ']']}]"
                           if _m2p.get("[" + m.group(1) + "]") is not None else ""),
                text,
            )
            # Make the figures an analyst scans for pop (₹/INR/Rs crore·lakh, %).
            text = re.sub(
                r"(?:₹|INR|Rs\.?)\s?[\d,]+(?:\.\d+)?\s*(?:crore|cr|lakh|billion|million|bn|mn)?",
                lambda m: f"**{m.group(0).strip()}**",
                text,
            )
            text = re.sub(r"(?<!\d)(\d+(?:\.\d+)?%)", r"**\1**", text)
            return text.strip()

        parts.append(f"\n**{title}**")
        bullets = b.get("summary_bullets") or []
        if bullets:
            parts.extend(f"- {_fmt(str(bl))}" for bl in bullets)
        else:
            parts.append("- _No filing evidence disclosed for this block._")
    return "\n".join(parts)


def _bmc_followups(resp: dict[str, Any]) -> list[str]:
    """Three useful next questions our tools can answer for a BMC turn."""
    c = resp.get("company_name") or resp.get("ticker") or "this company"
    return [
        f"How has {c}'s business model changed over recent years?",
        f"What are the main risks to {c}'s business model?",
        f"Which segment drives most of {c}'s revenue?",
    ]


def _merge_filing_citations(
    structured: FinalAnswer | None, tool_trace: list[dict[str, Any]],
) -> FinalAnswer | None:
    """Give filing citations a clickable ``url`` + exact ``page`` so the UI can
    deep-link to the cited PDF page. PRIMARY path: build them straight from the
    tool's ``evidence`` (synthesise=false default — each item has citation+page+
    pdf_url). FALLBACK (synthesise=true, no evidence): parse the agent's prose /
    citation labels and join to ``selected_filings`` pdf_link. All deterministic.
    Keeps non-filing citations; replaces filing ones. No-op when no filings."""
    if structured is None:
        return structured

    # PRIMARY — evidence carries everything; no parsing/guesswork. Includes BMC
    # block evidence so a Business Model Canvas answer's [Company | p.N] markers
    # deep-link to the cited PDF page too.
    ev_cites = _evidence_citations(tool_trace)
    bmc_cites = _bmc_citations(tool_trace)
    if bmc_cites:
        # The chat brief cites a handful of facts; the full 15-source set lives on
        # /bmc. Scope the Sources chip to the pages actually referenced inline so
        # "N sources" matches the brief (keep all if the model emitted no markers).
        referenced = {
            int(m.group(2)) for m in _INLINE_CITE_RE.finditer(structured.text or "")
        }
        if referenced:
            bmc_cites = [c for c in bmc_cites if c.page in referenced]
    ev_cites = ev_cites + bmc_cites
    if ev_cites:
        kept = [c for c in structured.citations if c.source_kind != "filing"]
        return structured.model_copy(update={"citations": kept + ev_cites})

    # FALLBACK — synthesise=true left evidence empty; derive from prose + filings.
    pdf_index = _filing_pdf_index(tool_trace)
    if not pdf_index:
        return structured
    # Single-company answers: every page-cite maps to the one PDF read.
    sole_pdf = next(iter(pdf_index.values())) if len(pdf_index) == 1 else None

    def _url_for(company_text: str) -> str | None:
        norm = _norm_cite_label(company_text)
        if norm in pdf_index:
            return pdf_index[norm]
        # Citation labels often pad the company ("ITC Ltd Annual Report 2025") —
        # match when a filing's company key appears within (works for multi-
        # company answers where there's no single-PDF fallback).
        for key, url in pdf_index.items():
            if key and key in norm:
                return url
        return sole_pdf

    out: list[Citation] = []
    seen: set[tuple[str, int]] = set()
    # 1) Enrich the agent's own citations whose label carries a page number.
    for c in structured.citations:
        m = _PAGE_IN_LABEL_RE.search(c.label or "")
        if m:
            page = int(m.group(1))
            company = re.sub(r"[\[\]|]", " ", _PAGE_IN_LABEL_RE.sub("", c.label or ""))
            url = _url_for(company)
            if url:
                c = c.model_copy(update={"url": url, "page": page, "source_kind": "filing"})
                seen.add((url, page))
        out.append(c)
    # 2) Add inline [Company | p.N] cites from the prose that aren't already listed.
    for mm in _INLINE_CITE_RE.finditer(structured.text or ""):
        company = mm.group(1).strip()
        try:
            page = int(mm.group(2))
        except ValueError:
            continue
        url = _url_for(company)
        if not url or (url, page) in seen:
            continue
        seen.add((url, page))
        out.append(
            Citation(label=f"[{company} | p.{page}]", url=url, source_kind="filing", page=page)
        )
    return structured.model_copy(update={"citations": out})


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


_TOOL_STEP_TITLES = {
    "resolve_company": "Identify the company",
    "resolve_companies": "Identify the companies",
    "search_companies": "Search companies",
    "financials_query": "Pull financial data",
    "stock_filings_read": "Read filings",
    "stock_filings_list": "List filings",
    "stock_technicals": "Fetch market data",
    "web_search": "Search the web",
}


def _plan_step_title(tool_name: str) -> str:
    """A short, user-facing checklist title for a tool — used to synthesize the
    runner's fallback checklist when the agent didn't declare one."""
    if tool_name in _TOOL_STEP_TITLES:
        return _TOOL_STEP_TITLES[tool_name]
    if tool_name.startswith("news_"):
        return "Gather recent news"
    if tool_name.startswith("bmc_"):
        return "Analyze the business model"
    if tool_name.startswith(("sebi_", "regulatory")):
        return "Check regulatory data"
    return tool_name.replace("_", " ").capitalize()


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
