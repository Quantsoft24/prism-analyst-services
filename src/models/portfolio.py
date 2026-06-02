"""Persistence for the Systematic Portfolio Builder (PRISM primary DB).

``pb_backtests`` is both the **durable job row** and the **result store** for a
backtest: a worker process claims queued rows (``FOR UPDATE SKIP LOCKED``), runs
the engine, streams progress back, and persists the result JSONB. Firm-scoped by
``firm_id`` (the slug, like ``bmc_analyses`` / ``agent_runs``) with a nullable
``created_by`` ready for per-user filtering once auth populates it.

Saved strategies + custom factors (Phase 4) get their own tables later.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin, UUIDPKMixin

# status values
BT_QUEUED = "queued"
BT_RUNNING = "running"
BT_SUCCEEDED = "succeeded"
BT_FAILED = "failed"
BT_CANCELLED = "cancelled"


class PortfolioBacktest(UUIDPKMixin, TimestampMixin, Base):
    """A backtest job + its result."""

    __tablename__ = "pb_backtests"

    firm_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)  # firm slug
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    name: Mapped[str | None] = mapped_column(String(200))

    # The full backtest spec (universe, filters, frequency, weighting, dates,
    # basis) — replayable and the basis of the result cache.
    spec: Mapped[dict] = mapped_column(JSONB, nullable=False)
    strategy_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default=BT_QUEUED, index=True)
    progress: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    stage: Mapped[str | None] = mapped_column(String(120))
    error: Mapped[str | None] = mapped_column(Text)

    # NAV / benchmark / drawdown / metrics / rebalances — present when succeeded.
    result: Mapped[dict | None] = mapped_column(JSONB)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    def __repr__(self) -> str:
        return f"<PortfolioBacktest {self.id} {self.status} {self.progress:.0%}>"


class PortfolioCustomFactor(UUIDPKMixin, TimestampMixin, Base):
    """A saved user-composed factor (expression over base factor ids)."""

    __tablename__ = "pb_custom_factors"
    __table_args__ = (
        UniqueConstraint("firm_id", "name", name="uq_pb_custom_factors_firm_name"),
    )

    firm_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    expression: Mapped[str] = mapped_column(Text, nullable=False)
    direction: Mapped[str] = mapped_column(String(16), nullable=False, server_default="higher_better")
    normalization: Mapped[str] = mapped_column(String(16), nullable=False, server_default="none")

    def __repr__(self) -> str:
        return f"<PortfolioCustomFactor {self.name!r}={self.expression!r}>"


class PortfolioStrategy(UUIDPKMixin, TimestampMixin, Base):
    """A saved builder configuration (universe + filters + rules + weighting +
    custom factors) — the basis for the Saved-Results / Edit flow."""

    __tablename__ = "pb_strategies"
    __table_args__ = (
        UniqueConstraint("firm_id", "name", name="uq_pb_strategies_firm_name"),
    )

    firm_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)

    def __repr__(self) -> str:
        return f"<PortfolioStrategy {self.name!r}>"
