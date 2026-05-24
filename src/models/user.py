"""User and FirmMembership models.

A ``User`` is one human identity (one email, federated via Clerk/OAuth later).
A user can belong to multiple firms via ``FirmMembership`` rows — common case
in equity research (independent analyst consulting for two PMSs).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin, UUIDPKMixin

if TYPE_CHECKING:
    from src.models.firm import Firm


class User(UUIDPKMixin, TimestampMixin, Base):
    """One human identity. Federated via Clerk/OAuth — ``external_id`` is the
    provider's user ID, ``email`` is the canonical lookup key for now."""

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False, server_default="")
    external_id: Mapped[str | None] = mapped_column(String(255), unique=True, index=True)
    is_active: Mapped[bool] = mapped_column(default=True, server_default="true", nullable=False)

    memberships: Mapped[list[FirmMembership]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User email={self.email!r}>"


class FirmMembership(UUIDPKMixin, TimestampMixin, Base):
    """Join table — which user has what role in which firm."""

    __tablename__ = "firm_memberships"
    __table_args__ = (
        UniqueConstraint("firm_id", "user_id", name="uq_firm_memberships_firm_user"),
    )

    firm_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("firms.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False, server_default="member")

    firm: Mapped[Firm] = relationship(back_populates="memberships")
    user: Mapped[User] = relationship(back_populates="memberships")
