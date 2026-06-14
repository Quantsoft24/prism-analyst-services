"""chat_conversations share columns — read-only public share link

Revision ID: 0019_chat_share
Revises: 0018_message_feedback
Create Date: 2026-06-15
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0019_chat_share"
down_revision: str | None = "0018_message_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_conversations", sa.Column("share_token", sa.String(length=48), nullable=True))
    op.add_column(
        "chat_conversations",
        sa.Column("shared_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "chat_conversations",
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "chat_conversations",
        sa.Column("shared_run_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index(
        "ix_chat_conversations_share_token",
        "chat_conversations",
        ["share_token"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_chat_conversations_share_token", table_name="chat_conversations")
    op.drop_column("chat_conversations", "shared_run_ids")
    op.drop_column("chat_conversations", "revoked_at")
    op.drop_column("chat_conversations", "shared_at")
    op.drop_column("chat_conversations", "share_token")
