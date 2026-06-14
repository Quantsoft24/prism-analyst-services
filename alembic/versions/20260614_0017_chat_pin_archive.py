"""chat_conversations: pin + archive (and make title nullable)

Adds ``is_pinned`` (sticks to the top of the list) and ``archived_at`` (hidden
from the default list, shown only in an Archived view). ``title`` becomes
nullable so an overlay row can exist for pin/archive alone — the list falls back
to the first-message title when it's NULL.

Revision ID: 0017_chat_pin_archive
Revises: 0016_agent_runs_result_payload
Create Date: 2026-06-14

(Revision id kept ≤32 chars — alembic_version.version_num is varchar(32).)
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision: str = "0017_chat_pin_archive"
down_revision: str | None = "0016_agent_runs_result_payload"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_conversations",
        sa.Column("is_pinned", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "chat_conversations",
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.alter_column(
        "chat_conversations", "title", existing_type=sa.String(length=200), nullable=True
    )


def downgrade() -> None:
    # Backfill any NULL titles so the NOT NULL restore can't fail.
    op.execute("UPDATE chat_conversations SET title = 'Untitled' WHERE title IS NULL")
    op.alter_column(
        "chat_conversations", "title", existing_type=sa.String(length=200), nullable=False
    )
    op.drop_column("chat_conversations", "archived_at")
    op.drop_column("chat_conversations", "is_pinned")
