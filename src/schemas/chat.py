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
    """Tool returned. ``ok=false`` carries an error message instead of a result."""

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    tool: str
    ok: bool = True
    result_summary: str | None = None  # short human-readable, e.g. "found TCS · 8 fields"
    error: str | None = None
    latency_ms: int


class TokenEvent(BaseModel):
    """Streamed chunk of the final LLM response. Concatenate ``text`` in order."""

    type: Literal["token"] = "token"
    text: str


class FinalEvent(BaseModel):
    """Terminal success event. ``cost_usd`` may be 0 on free-tier."""

    type: Literal["final"] = "final"
    answer: str
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
