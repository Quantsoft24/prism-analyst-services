"""bmc_analyses + bmc_blocks + bmc_evidence — Business Model Canvas (Phase 2 Lite)

Revision ID: 0005_bmc_tables
Revises: 0004_filings_and_chunks
Create Date: 2026-05-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_bmc_tables"
down_revision: str | None = "0004_filings_and_chunks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── bmc_analyses ──────────────────────────────────────────────────────
    op.create_table(
        "bmc_analyses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("firm_id", sa.String(64), nullable=False),
        sa.Column("ticker", sa.String(32), nullable=False),
        sa.Column("company_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("companies.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("fiscal_period", sa.String(32)),
        sa.Column("status", sa.String(32), nullable=False, server_default="running"),
        sa.Column("overall_confidence", sa.Float()),
        sa.Column("cost_usd", sa.Numeric(10, 6), nullable=False, server_default="0"),
        sa.Column("model", sa.String(64)),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("firm_id", "ticker", "version", name="uq_bmc_firm_ticker_version"),
    )
    op.create_index("ix_bmc_analyses_firm_id", "bmc_analyses", ["firm_id"])
    op.create_index("ix_bmc_analyses_ticker", "bmc_analyses", ["ticker"])
    op.create_index("ix_bmc_analyses_firm_ticker", "bmc_analyses", ["firm_id", "ticker", sa.text("version DESC")])

    # ── bmc_blocks ────────────────────────────────────────────────────────
    op.create_table(
        "bmc_blocks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("bmc_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("bmc_analyses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("block_id", sa.String(32), nullable=False),
        sa.Column("title", sa.String(64), nullable=False),
        sa.Column("order", sa.Integer(), nullable=False),
        sa.Column("summary_bullets", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("key_insights", postgresql.JSONB()),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(32), nullable=False, server_default="ok"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("bmc_id", "block_id", name="uq_bmc_blocks_bmc_block"),
    )
    op.create_index("ix_bmc_blocks_bmc_id", "bmc_blocks", ["bmc_id"])

    # ── bmc_evidence ──────────────────────────────────────────────────────
    op.create_table(
        "bmc_evidence",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("bmc_block_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("bmc_blocks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("marker", sa.String(8), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("filing_chunks.id", ondelete="SET NULL")),
        sa.Column("filing_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("filings.id", ondelete="SET NULL")),
        sa.Column("page_number", sa.Integer()),
        sa.Column("excerpt", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_bmc_evidence_block", "bmc_evidence", ["bmc_block_id"])


def downgrade() -> None:
    op.drop_table("bmc_evidence")
    op.drop_table("bmc_blocks")
    op.drop_table("bmc_analyses")
