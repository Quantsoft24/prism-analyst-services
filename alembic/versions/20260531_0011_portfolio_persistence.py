"""pb_custom_factors + pb_strategies — saved custom factors and strategies

Revision ID: 0011_portfolio_persistence
Revises: 0010_portfolio_backtests
Create Date: 2026-05-31
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0011_portfolio_persistence"
down_revision: str | None = "0010_portfolio_backtests"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pb_custom_factors",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("firm_id", sa.String(64), nullable=False),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("expression", sa.Text(), nullable=False),
        sa.Column("direction", sa.String(16), nullable=False, server_default="higher_better"),
        sa.Column("normalization", sa.String(16), nullable=False, server_default="none"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("firm_id", "name", name="uq_pb_custom_factors_firm_name"),
    )
    op.create_index("ix_pb_custom_factors_firm_id", "pb_custom_factors", ["firm_id"])

    op.create_table(
        "pb_strategies",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("firm_id", sa.String(64), nullable=False),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("firm_id", "name", name="uq_pb_strategies_firm_name"),
    )
    op.create_index("ix_pb_strategies_firm_id", "pb_strategies", ["firm_id"])


def downgrade() -> None:
    op.drop_index("ix_pb_strategies_firm_id", table_name="pb_strategies")
    op.drop_table("pb_strategies")
    op.drop_index("ix_pb_custom_factors_firm_id", table_name="pb_custom_factors")
    op.drop_table("pb_custom_factors")
