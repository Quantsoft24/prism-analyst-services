"""Wire types for the chat / agent-run API.

The SSE stream emitted by ``POST /api/v1/chat/run`` consists of typed events.
Frontend code switches on ``event.type`` — keep this union closed and stable.
Add a new event variant only when the new information genuinely cannot fit
into an existing one.
"""

from __future__ import annotations

import uuid
from datetime import datetime
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


class ChartPoint(BaseModel):
    """One labeled (x, y) data point on a chart series."""

    x: str  # label / period — e.g. "Q4'25", "2024-04-30", "TCS"
    y: float


class ChartEvent(BaseModel):
    """Structured chart data a tool surfaced — drives Workspace → Charts tab.

    No tool emits these today; the schema lives here so when a chart-producing
    tool ships (e.g. ``compute_trend`` over a NRE series, or a ``stock_technicals``
    enrichment), the runner can emit ChartEvent and the frontend renders it
    without further changes. Frontend mock-mode already validates the end-to-end
    pipeline (see ``chat.mock.ts``).
    """

    type: Literal["chart"] = "chart"
    call_id: str | None = None  # which tool emitted it; null if from final
    chart_id: str  # stable id so the UI can dedup
    title: str  # e.g. "Jio segment ARPU · trailing 5 quarters"
    unit: str = ""  # display prefix/suffix — "₹" / "%" / "x" / ""
    current_value: str  # latest value as a display string ("202", "47,628")
    current_delta: str | None = None  # "+3.1% q/q" / "−0.42% YTD"
    delta_kind: Literal["pos", "neg", "neutral"] | None = None
    points: list[ChartPoint] = Field(default_factory=list)
    kind: Literal["line", "area", "bar"] = "line"


# ── Structured final-answer payload ───────────────────────────────────────


class Citation(BaseModel):
    """A single citation backing a fact in the final answer."""

    label: str  # e.g. "Reliance Industries Q4 FY24 filing, p. 12"
    url: str | None = None
    source_kind: Literal["filing", "web", "bmc", "tool"] = "tool"
    as_of: str | None = None  # ISO date or null
    tool_call_id: str | None = None  # links to a ToolCallEvent.call_id


class FinalKpi(BaseModel):
    """A headline KPI surfaced in the workspace Report tab.

    Renders as one card in the KPI grid (mockup pattern: Revenue ·
    ₹2.74L cr · cite 1 · pg 4). No tool emits these today; the schema
    lives here so when a tool ships that extracts headline numbers
    from a filing (or the agent fills it from a structured answer
    block), the Report tab renders without further frontend changes.
    """

    label: str  # e.g. "Revenue"
    value: str  # e.g. "₹2.74L cr" or "47,628"
    unit: str | None = None  # optional secondary unit chip ("cr", "%")
    cite_label: str | None = None  # short "cite 1 · pg 4" reference


class FinalSection(BaseModel):
    """A named section in the final research note (Executive summary,
    Anomaly flags, etc.). Body is markdown so citations / bold / lists
    survive. ``kind`` lets the UI accent anomaly callouts in warn-yellow.
    """

    title: str
    body: str  # markdown
    kind: Literal["summary", "anomaly", "note"] = "summary"


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
    kpis: list[FinalKpi] = Field(default_factory=list)
    sections: list[FinalSection] = Field(default_factory=list)


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


# ── Conversation history (derived from agent_runs by session_id) ───────────


class ConversationSummary(BaseModel):
    """One row in the user's conversation list (sidebar / history)."""

    session_id: str
    title: str  # first user message, truncated
    turns: int
    last_activity: datetime
    preview: str = ""  # latest answer, truncated
    agent_name: str | None = None


class ConversationTurn(BaseModel):
    """One turn (one agent_run) inside a conversation, for replay."""

    agent_run_id: uuid.UUID
    user_input: str
    final_answer: str | None = None
    status: str
    created_at: datetime
    tool_trace: list[dict[str, Any]] | None = None


class ConversationDetail(BaseModel):
    session_id: str
    turns: list[ConversationTurn] = Field(default_factory=list)


class ConversationTitleUpdate(BaseModel):
    """PATCH body to rename a conversation."""

    title: str = Field(min_length=1, max_length=200)
