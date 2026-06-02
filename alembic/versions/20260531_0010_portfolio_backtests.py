"""pb_backtests — Systematic Portfolio Builder backtest jobs + results

Revision ID: 0010_portfolio_backtests
Revises: 0009_drop_companies_and_filings
Create Date: 2026-05-31
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0010_portfolio_backtests"
down_revision: str | None = "0009_drop_companies_and_filings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pb_backtests",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("firm_id", sa.String(64), nullable=False),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("name", sa.String(200)),
        sa.Column("spec", postgresql.JSONB(), nullable=False),
        sa.Column("strategy_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0"),
        sa.Column("stage", sa.String(120)),
        sa.Column("error", sa.Text()),
        sa.Column("result", postgresql.JSONB()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_pb_backtests_firm_id", "pb_backtests", ["firm_id"])
    op.create_index("ix_pb_backtests_strategy_hash", "pb_backtests", ["strategy_hash"])
    op.create_index("ix_pb_backtests_status", "pb_backtests", ["status"])
    # Worker claim path: cheapest "oldest queued" scan.
    op.create_index(
        "ix_pb_backtests_status_created", "pb_backtests", ["status", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_pb_backtests_status_created", table_name="pb_backtests")
    op.drop_index("ix_pb_backtests_status", table_name="pb_backtests")
    op.drop_index("ix_pb_backtests_strategy_hash", table_name="pb_backtests")
    op.drop_index("ix_pb_backtests_firm_id", table_name="pb_backtests")
    op.drop_table("pb_backtests")
