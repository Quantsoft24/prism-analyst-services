"""ORM models for the Business Model Canvas.

Three tables (Phase 2 Lite):
  * ``bmc_analyses``  — one row per (firm, ticker, version). The canvas header.
  * ``bmc_blocks``    — 9 rows per analysis (one per Osterwalder block).
  * ``bmc_evidence``  — citation links: which filing chunk supports which block.

Per-block drill-down chat (``bmc_chats``) and temporal diffs (``bmc_diffs``)
are Phase 3/4 — deliberately not created yet (YAGNI; they get their own
migration when built).

Tenant-scoped via ``firm_id`` from Day 1. Append-only semantics: re-generating
a BMC creates a NEW version rather than mutating the old one, so a firm can
diff a company's business model over time (the Phase 4 differentiator).
"""

from __future__ import annotations

import uuid

from sqlalchemy import Float, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin, UUIDPKMixin

# Per-block status:
#   'ok'               — generated with evidence
#   'evidence_missing' — no filing evidence found; block left honest-empty
#   'failed'           — generation errored for this block
BMC_BLOCK_STATUSES = {"ok", "evidence_missing", "failed"}

# Analysis status lifecycle.
BMC_STATUSES = {"running", "complete", "partial", "failed"}


class BMCAnalysis(UUIDPKMixin, TimestampMixin, Base):
    """One generated canvas for a company, at a point in time."""

    __tablename__ = "bmc_analyses"
    __table_args__ = (
        UniqueConstraint("firm_id", "ticker", "version", name="uq_bmc_firm_ticker_version"),
    )

    firm_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")

    # Anchors temporal diffing (Phase 4) — which fiscal period the underlying
    # filings represent, e.g. "Q4-FY26".
    fiscal_period: Mapped[str | None] = mapped_column(String(32))

    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="running")
    # 0..1 average of per-block confidences — a quick canvas-quality signal.
    overall_confidence: Mapped[float | None] = mapped_column(Float)

    # Cost + provenance.
    cost_usd: Mapped[float] = mapped_column(Numeric(10, 6), server_default="0", nullable=False)
    model: Mapped[str | None] = mapped_column(String(64))

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )

    blocks: Mapped[list[BMCBlock]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<BMCAnalysis {self.ticker} v{self.version} status={self.status!r}>"


class BMCBlock(UUIDPKMixin, TimestampMixin, Base):
    """One Osterwalder block within an analysis."""

    __tablename__ = "bmc_blocks"
    __table_args__ = (
        UniqueConstraint("bmc_id", "block_id", name="uq_bmc_blocks_bmc_block"),
    )

    bmc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("bmc_analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    block_id: Mapped[str] = mapped_column(String(32), nullable=False)  # e.g. 'revenue_streams'
    title: Mapped[str] = mapped_column(String(64), nullable=False)
    order: Mapped[int] = mapped_column(Integer, nullable=False)

    # The block content — a list of bullet strings with inline [n] citation markers.
    summary_bullets: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="[]")
    key_insights: Mapped[list | None] = mapped_column(JSONB)

    confidence: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="ok")

    analysis: Mapped[BMCAnalysis] = relationship(back_populates="blocks")
    evidence: Mapped[list[BMCEvidence]] = relationship(
        back_populates="block", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<BMCBlock {self.block_id} status={self.status!r}>"


class BMCEvidence(UUIDPKMixin, TimestampMixin, Base):
    """A citation: the filing chunk backing a block's claims.

    ``marker`` is the inline reference (e.g. '[1]') used in ``summary_bullets``.
    Links to the source ``filing_chunks`` row so the UI can deep-link to the
    exact filing + page — the "show your work" contract.
    """

    __tablename__ = "bmc_evidence"

    bmc_block_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("bmc_blocks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    marker: Mapped[str] = mapped_column(String(8), nullable=False)  # '[1]', '[2]', ...
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("filing_chunks.id", ondelete="SET NULL")
    )
    filing_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("filings.id", ondelete="SET NULL")
    )
    page_number: Mapped[int | None] = mapped_column(Integer)
    excerpt: Mapped[str] = mapped_column(Text, nullable=False)  # quoted text shown in UI

    block: Mapped[BMCBlock] = relationship(back_populates="evidence")
