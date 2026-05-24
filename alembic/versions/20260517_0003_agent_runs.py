"""agent_runs — audit log for every agent invocation

Revision ID: 0003_agent_runs
Revises: 0002_seed_nse_top10
Create Date: 2026-05-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_agent_runs"
down_revision: str | None = "0002_seed_nse_top10"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("firm_id", sa.String(64), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        sa.Column("session_id", sa.String(128), nullable=False),
        sa.Column("agent_name", sa.String(64), nullable=False),
        sa.Column("user_input", sa.Text(), nullable=False),
        sa.Column("final_answer", sa.Text()),
        sa.Column("status", sa.String(32), nullable=False, server_default="running"),
        sa.Column("error_code", sa.String(64)),
        sa.Column("error_message", sa.Text()),
        sa.Column("tool_trace", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb")),
        sa.Column("model", sa.String(64)),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Numeric(10, 6), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Integer()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_agent_runs_firm_id", "agent_runs", ["firm_id"])
    op.create_index("ix_agent_runs_user_id", "agent_runs", ["user_id"])
    op.create_index("ix_agent_runs_session_id", "agent_runs", ["session_id"])
    op.create_index("ix_agent_runs_agent_name", "agent_runs", ["agent_name"])
    # Common queries: "all runs for this firm in date range", "all failed runs"
    op.create_index(
        "ix_agent_runs_firm_created", "agent_runs", ["firm_id", sa.text("created_at DESC")]
    )


def downgrade() -> None:
    op.drop_table("agent_runs")
