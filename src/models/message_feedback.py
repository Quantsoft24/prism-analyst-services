"""MessageFeedback — a user's 👍 / 👎 on one agent answer (one ``agent_run``).

One row per rated turn (``agent_run_id`` is unique → re-rating upserts). Powers
the quality loop hinted at in ``AgentRun`` ("which agent runs got 👎"). Scoped to
the principal who owns the run (signed-in user / guest / firm); the FK preserves
the link to the audited turn.
"""

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, SmallInteger, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin, UUIDPKMixin


class MessageFeedback(UUIDPKMixin, TimestampMixin, Base):
    """One user's rating of one agent answer."""

    __tablename__ = "message_feedback"
    __table_args__ = (UniqueConstraint("agent_run_id", name="uq_message_feedback_agent_run"),)

    agent_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Who rated (record-keeping + scoping); tenancy via firm_id, guests via client_key.
    firm_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    client_key: Mapped[str | None] = mapped_column(String(128))

    rating: Mapped[int] = mapped_column(SmallInteger, nullable=False)  # +1 (👍) / -1 (👎)
    reasons: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="[]")  # 👎 reason chips
    comment: Mapped[str | None] = mapped_column(Text)

    def __repr__(self) -> str:
        return f"<MessageFeedback run={self.agent_run_id} rating={self.rating}>"
