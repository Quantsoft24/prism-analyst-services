"""Per-user preferences (PRISM primary DB).

One JSONB row per user holding the things the Settings screens currently mock —
default model, citation policy, watchlist, notification prefs, etc. JSONB (not
typed columns) so the UI can add preference keys without a migration each time.

Keyed by ``user_id`` (PK) — so it only ever exists for a real, provisioned user;
the dev stub (no user) reads defaults and can't persist.
"""

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin


class UserPreference(TimestampMixin, Base):
    """A user's preference blob. PK is the user id (1:1 with ``users``)."""

    __tablename__ = "user_preferences"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    prefs: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")

    def __repr__(self) -> str:
        return f"<UserPreference user_id={self.user_id}>"
