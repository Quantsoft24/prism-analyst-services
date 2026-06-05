"""agent_runs.hidden_at — soft-delete (hide) a conversation from a user's history

Append-only audit table: we never hard-delete an agent_run. ``hidden_at`` marks
runs a user chose to hide; the conversation-history queries exclude them.

Revision ID: 0013_agent_runs_hidden_at
Revises: 0012_auth_foundation
Create Date: 2026-06-05
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0013_agent_runs_hidden_at"
down_revision: str | None = "0012_auth_foundation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_runs", sa.Column("hidden_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_agent_runs_hidden_at", "agent_runs", ["hidden_at"])


def downgrade() -> None:
    op.drop_index("ix_agent_runs_hidden_at", table_name="agent_runs")
    op.drop_column("agent_runs", "hidden_at")
