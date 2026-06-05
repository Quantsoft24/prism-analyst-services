"""Per-conversation overlay (titles, etc.) keyed by ``session_id``.

The audit log (``agent_runs``) is per-run and immutable. User-editable
conversation metadata (a renamed title today; pin/archive later) lives here so
we never mutate the audit trail. One row per session; absent → the title is
derived from the first user message.
"""

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin


class ChatConversation(TimestampMixin, Base):
    """Editable overlay for a chat conversation (a set of agent_runs sharing a
    session_id). PK is the session_id."""

    __tablename__ = "chat_conversations"

    session_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    firm_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)

    def __repr__(self) -> str:
        return f"<ChatConversation {self.session_id} title={self.title!r}>"
