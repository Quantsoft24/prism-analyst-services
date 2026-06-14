"""Per-conversation overlay (titles, etc.) keyed by ``session_id``.

The audit log (``agent_runs``) is per-run and immutable. User-editable
conversation metadata (a renamed title today; pin/archive later) lives here so
we never mutate the audit trail. One row per session; absent → the title is
derived from the first user message.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin


class ChatConversation(TimestampMixin, Base):
    """Editable overlay for a chat conversation (a set of agent_runs sharing a
    session_id). PK is the session_id. Holds user-set metadata — a renamed
    ``title``, ``is_pinned`` (sticks to the top), and ``archived_at`` (hidden
    from the default list). ``title`` is nullable so a row can exist for pin /
    archive alone (the list falls back to the first-message title)."""

    __tablename__ = "chat_conversations"

    session_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    firm_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    is_pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Read-only public share. ``share_token`` is the unguessable public lookup
    # key; ``shared_run_ids`` freezes the EXACT turns shared (so continuing the
    # conversation privately never leaks into the public link, and a future
    # branched share pins to the shared path). ``revoked_at`` kills the link
    # while keeping the audit row. Re-sharing after a revoke mints a new token.
    share_token: Mapped[str | None] = mapped_column(String(48), unique=True, index=True)
    shared_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    shared_run_ids: Mapped[list | None] = mapped_column(JSONB)

    def __repr__(self) -> str:
        return f"<ChatConversation {self.session_id} title={self.title!r}>"
