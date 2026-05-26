"""Wire types for the chat / agent-run API.

The SSE stream emitted by ``POST /api/v1/chat/run`` consists of typed events.
Frontend code switches on ``event.type`` — keep this union closed and stable.
Add a new event variant only when the new information genuinely cannot fit
into an existing one.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

# ── Request ───────────────────────────────────────────────────────────────


class ChatRunRequest(BaseModel):
    """Body for ``POST /api/v1/chat/run``."""

    message: str = Field(min_length=1, max_length=4_000, description="The analyst's query.")
    session_id: str | None = Field(
        default=None,
        description=(
            "Pass an existing ADK session ID to continue a conversation; "
            "omit / null to start a new session (server will allocate one and "
            "return it in the first ``meta`` event)."
        ),
    )
    agent: Literal["company_intel"] = Field(
        default="company_intel",
        description="Which agent to invoke. Only one available in Slice 3.",
    )


# ── Streamed event variants ───────────────────────────────────────────────
# Each event is JSON-encoded into the SSE ``data:`` field.


class MetaEvent(BaseModel):
    """First event emitted — gives the client an ID to reference the run."""

    type: Literal["meta"] = "meta"
    agent_run_id: uuid.UUID
    session_id: str
    agent_name: str


class ToolCallEvent(BaseModel):
    """Agent invoked a tool. Args are JSON-safe; large values truncated server-side."""

    type: Literal["tool_call"] = "tool_call"
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    # An opaque ID linking this call to its matching ``tool_result`` event.
    call_id: str


class ToolResultEvent(BaseModel):
    """Tool returned. ``ok=false`` carries an error message instead of a result.

    ``error_code`` and ``next_action`` mirror the structured-error contract
    (see ``src/integrations/tools/_errors.py``) — the UI uses them to render
    icons / chips / hints. Both are optional for back-compat with legacy
    tools that haven't been migrated.
    """

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    tool: str
    ok: bool = True
    result_summary: str | None = None  # short human-readable, e.g. "found TCS · 8 fields"
    error: str | None = None
    error_code: str | None = None
    next_action: str | None = None  # one of NextAction (see _errors.py)
    latency_ms: int


class TokenEvent(BaseModel):
    """Streamed chunk of the final LLM response. Concatenate ``text`` in order."""

    type: Literal["token"] = "token"
    text: str


class AgentThoughtEvent(BaseModel):
    """The agent surfaced a piece of reasoning the user can inspect.

    Sent when ADK exposes thought / planning content parts. The UI renders
    these as collapsible "Thinking…" cards above the eventual answer.

    ``kind`` lets the UI default-expand the most useful kind:
      • ``plan``     — initial step-list ("I'll first look up X, then …")
      • ``reflect``  — mid-run reconsideration ("That tool returned …, so …")
      • ``decision`` — branch chosen ("Using stock_filings_read because …")
    """

    type: Literal["agent_thought"] = "agent_thought"
    text: str
    kind: Literal["plan", "reflect", "decision"] = "decision"


class ToolRetryEvent(BaseModel):
    """The runner is re-invoking a tool after a transient failure.

    Emitted between the failing ``tool_result`` and the next ``tool_call``
    for the same ``call_id``. The frontend renders a ↻ retry indicator on
    that tool's card.
    """

    type: Literal["tool_retry"] = "tool_retry"
    call_id: str
    tool: str
    attempt: int  # 1-indexed: 2 means "second try"
    reason: str  # short human reason ("upstream timeout", "503")


class DataFreshnessEvent(BaseModel):
    """A tool result carries a known data-freshness signal.

    Emitted immediately after the corresponding ``tool_result`` when the
    tool's response dict includes a ``data_freshness`` field (e.g. the
    latest filing date in a ``stock_filings_*`` result, or ``"live"`` for
    technicals). The UI shows a "data as of …" chip on the answer block
    and the relevant tool card.
    """

    type: Literal["data_freshness"] = "data_freshness"
    call_id: str
    source: str  # short label e.g. "stock-chat filings"
    as_of: str | None = None  # ISO date / "live" / null


# ── Structured final-answer payload ───────────────────────────────────────


class Citation(BaseModel):
    """A single citation backing a fact in the final answer."""

    label: str  # e.g. "Reliance Industries Q4 FY24 filing, p. 12"
    url: str | None = None
    source_kind: Literal["filing", "web", "bmc", "tool"] = "tool"
    as_of: str | None = None  # ISO date or null
    tool_call_id: str | None = None  # links to a ToolCallEvent.call_id


class FinalAnswer(BaseModel):
    """Structured answer payload. Replaces the bare string in ``FinalEvent.answer``.

    The agent is instructed (see ``company_intel.py`` system prompt) to
    return this shape on every successful turn. Older clients that consume
    ``answer`` as a string still work via ``str(FinalAnswer)`` — but the
    proper rendering path is to parse the structured fields.
    """

    text: str  # the prose answer (markdown allowed)
    citations: list[Citation] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    data_freshness: str | None = None  # earliest source date present in the answer


class FinalEvent(BaseModel):
    """Terminal success event. ``cost_usd`` may be 0 on free-tier.

    ``answer`` is the prose text (kept for back-compat); ``structured``
    carries the FinalAnswer object when the agent emitted one — preferred
    rendering path for the frontend.
    """

    type: Literal["final"] = "final"
    answer: str
    structured: FinalAnswer | None = None
    agent_run_id: uuid.UUID
    cost_usd: float
    input_tokens: int
    output_tokens: int
    latency_ms: int


class ErrorEvent(BaseModel):
    """Terminal failure event. ``retriable`` hints whether the UI should
    offer a retry button."""

    type: Literal["error"] = "error"
    code: str
    message: str
    retriable: bool = False
    agent_run_id: uuid.UUID | None = None
