"""chat_conversations — editable overlay (titles) for chat conversations

Per-session overlay so renaming a conversation never touches the immutable
agent_runs audit log.

Revision ID: 0014_chat_conversations
Revises: 0013_agent_runs_hidden_at
Create Date: 2026-06-05
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0014_chat_conversations"
down_revision: str | None = "0013_agent_runs_hidden_at"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_conversations",
        sa.Column("session_id", sa.String(128), primary_key=True),
        sa.Column("firm_id", sa.String(64), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_chat_conversations_firm_id", "chat_conversations", ["firm_id"])
    op.create_index("ix_chat_conversations_user_id", "chat_conversations", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_chat_conversations_user_id", table_name="chat_conversations")
    op.drop_index("ix_chat_conversations_firm_id", table_name="chat_conversations")
    op.drop_table("chat_conversations")
