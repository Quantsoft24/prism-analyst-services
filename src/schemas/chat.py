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
    # Page in the source PDF this citation points at (filings). Lets the UI deep
    # link to the exact page in the Report-tab viewer. Populated in Phase 6.
    page: int | None = None


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
    # 2-3 suggested next questions our tools can answer — rendered as clickable
    # chips. Composed in the same pass (no extra LLM call).
    suggestions: list[str] = Field(default_factory=list)


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


class ClarificationOption(BaseModel):
    """One selectable answer in a clarification question (single/multi select)."""

    id: str                       # stable client key
    label: str                    # e.g. "Reliance Industries Ltd."
    hint: str | None = None       # e.g. "RELIANCE · NSE/BSE · Oil, Gas …"
    value: Any                    # what the client sends back (e.g. a security_id)


class ClarificationQuestion(BaseModel):
    """One question in a clarification form. Several can be asked together (e.g.
    disambiguating "Reliance", "Adani", and "Tata" in one comparison) so the user
    answers them all at once — Claude-Code-style multi-question prompts."""

    id: str                       # stable key — e.g. the term being disambiguated ("Reliance")
    question: str
    mode: Literal["single_select", "multi_select", "open_text"] = "single_select"
    options: list[ClarificationOption] = Field(default_factory=list)
    # The UI offers a securities search box (the "none of these" path).
    allow_search: bool = True


class ClarificationEvent(BaseModel):
    """Terminal event: the agent needs the user to disambiguate before it can
    proceed. The UI renders an interactive form with ONE OR MORE questions (each a
    radio/MCQ, checkboxes, or free-text, plus a master_securities search box for
    company picks). The user answers them all and the combined selection is sent
    as the next ``/chat/run`` message in the same session; the agent resumes.

    Replaces a prose "which one did you mean?" with a structured, clickable
    picker — the core of the agentic clarification flow.
    """

    type: Literal["clarification"] = "clarification"
    agent_run_id: uuid.UUID | None = None
    # The form's questions (one or many). Clients should render `questions`.
    questions: list[ClarificationQuestion] = Field(default_factory=list)
    # ── Back-compat single-question mirror (= questions[0]); prefer `questions`. ──
    question: str = ""
    mode: Literal["single_select", "multi_select", "open_text"] = "single_select"
    options: list[ClarificationOption] = Field(default_factory=list)
    allow_search: bool = True


class PlanStep(BaseModel):
    """One task in the agent's visible plan/checklist."""

    id: str = ""
    title: str
    status: Literal["pending", "in_progress", "done"] = "pending"


class PlanEvent(BaseModel):
    """The agent's task list (Claude-Code-style checklist). Emitted whenever the
    agent declares or updates its plan via ``update_plan``; the UI renders the
    latest ``steps`` as checkboxes that tick off as work progresses."""

    type: Literal["plan"] = "plan"
    steps: list[PlanStep] = Field(default_factory=list)


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
    """One turn (one agent_run) inside a conversation, for replay.

    ``structured`` / ``plan`` / ``clarification`` are restored from the stored
    ``result_payload`` so reopening a past conversation renders the SAME rich
    view the user saw live (citations, confidence, freshness, sources, follow-up
    chips, task checklist, and a resumable pending clarification) — not a
    degraded prose-only replay. Null/empty on legacy rows saved before this was
    persisted.
    """

    agent_run_id: uuid.UUID
    user_input: str
    final_answer: str | None = None
    status: str
    created_at: datetime
    tool_trace: list[dict[str, Any]] | None = None
    structured: FinalAnswer | None = None
    plan: list[PlanStep] = Field(default_factory=list)
    clarification: ClarificationEvent | None = None


class ConversationDetail(BaseModel):
    session_id: str
    turns: list[ConversationTurn] = Field(default_factory=list)


class ConversationTitleUpdate(BaseModel):
    """PATCH body to rename a conversation."""

    title: str = Field(min_length=1, max_length=200)


class QuotaRead(BaseModel):
    """Today's message quota for the caller (guest or signed-in)."""

    limit: int
    used: int
    remaining: int
    is_anonymous: bool
    enabled: bool
