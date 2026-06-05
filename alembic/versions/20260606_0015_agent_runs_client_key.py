"""agent_runs.client_key — per-guest id for anonymous daily message limit

Revision ID: 0015_agent_runs_client_key
Revises: 0014_chat_conversations
Create Date: 2026-06-06
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0015_agent_runs_client_key"
down_revision: str | None = "0014_chat_conversations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_runs", sa.Column("client_key", sa.String(128), nullable=True))
    op.create_index("ix_agent_runs_client_key", "agent_runs", ["client_key"])


def downgrade() -> None:
    op.drop_index("ix_agent_runs_client_key", table_name="agent_runs")
    op.drop_column("agent_runs", "client_key")
