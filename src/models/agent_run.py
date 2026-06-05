"""AgentRun — the audit log row for every agent invocation.

One row per ``POST /api/v1/chat/run`` request. Captures input, the agent's
tool trace, final answer, latency, token usage, and cost — for:
  * **compliance** (every decision traceable for SEBI-style review),
  * **replay** (re-run a failed agent with the same inputs),
  * **cost analytics** (per-firm, per-user, per-agent),
  * **quality eval** (which agent runs got 👎 from users).

Tenant-scoped via ``firm_id``. Append-only (no deletes) — soft delete only.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin, UUIDPKMixin


class AgentRun(UUIDPKMixin, TimestampMixin, Base):
    """One agent invocation."""

    __tablename__ = "agent_runs"

    # Tenancy — every row carries firm_id from Day 1.
    firm_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True
    )

    # ADK session — multiple agent_runs may share a session_id (multi-turn chat).
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    # Which agent ran. Free-form string for now ("company_intel", "bmc_orchestrator", ...).
    agent_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # User-facing input + final answer.
    user_input: Mapped[str] = mapped_column(Text, nullable=False)
    final_answer: Mapped[str | None] = mapped_column(Text)

    # Status lifecycle:
    #   'running' → 'complete' | 'failed' | 'timeout' | 'cost_exceeded'
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="running")
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)

    # Full tool-call trace: list of {tool, args, result_summary, latency_ms, ts}.
    # JSONB so we can query later (e.g. "all runs that called google_search").
    tool_trace: Mapped[list | None] = mapped_column(JSONB, server_default="[]")

    # Cost + token telemetry — populated from ADK event usage_metadata.
    model: Mapped[str | None] = mapped_column(String(64))
    input_tokens: Mapped[int] = mapped_column(default=0, server_default="0", nullable=False)
    output_tokens: Mapped[int] = mapped_column(default=0, server_default="0", nullable=False)
    cost_usd: Mapped[float] = mapped_column(
        Numeric(10, 6), default=0, server_default="0", nullable=False
    )
    latency_ms: Mapped[int | None] = mapped_column()

    # Soft-delete: set when a user hides this conversation. The audit row is
    # PRESERVED (cost/tokens/trace intact); it's just excluded from the user's
    # conversation history. Never hard-deleted — append-only for compliance.
    hidden_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    def __repr__(self) -> str:
        return f"<AgentRun id={self.id} agent={self.agent_name!r} status={self.status!r}>"
