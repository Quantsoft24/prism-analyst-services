"""agent_runs.result_payload — persist structured answer / plan / clarification for faithful replay

Revision ID: 0016_agent_runs_result_payload
Revises: 0015_agent_runs_client_key
Create Date: 2026-06-12
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0016_agent_runs_result_payload"
down_revision: str | None = "0015_agent_runs_client_key"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column("result_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_runs", "result_payload")
