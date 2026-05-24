"""Firm = a customer organization (HF, PMS, AMC, broker, family office, ...)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin, UUIDPKMixin

if TYPE_CHECKING:
    from src.models.user import FirmMembership


class Firm(UUIDPKMixin, TimestampMixin, Base):
    """A tenant. Every user-data row in the system carries a ``firm_id``."""

    __tablename__ = "firms"

    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    subscription_tier: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="trial"
    )
    country: Mapped[str] = mapped_column(String(2), nullable=False, server_default="IN")

    memberships: Mapped[list[FirmMembership]] = relationship(
        back_populates="firm", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Firm slug={self.slug!r} name={self.name!r}>"
