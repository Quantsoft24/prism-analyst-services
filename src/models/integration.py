"""ORM model for firm-level integration enable/disable overrides.

The *available* integrations live in ``config/integrations.yml`` (the catalog,
built into tools at startup). This table stores only a firm's **deviations** from
the default — default is ON (Part-A), so a row exists only when a firm has
toggled something OFF (or explicitly back ON).

Phase 2 (firm-level). When auth/user-profiles land, a sibling
``user_integrations`` table (or a ``user_id`` column) extends this to per-user
overrides without reworking the registry — the resolver just layers user over
firm over default. See memory: per-user is deferred.
"""

from __future__ import annotations

from sqlalchemy import Boolean, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin, UUIDPKMixin


class FirmIntegration(UUIDPKMixin, TimestampMixin, Base):
    """A firm's enable/disable override for one named integration."""

    __tablename__ = "firm_integrations"
    __table_args__ = (
        UniqueConstraint("firm_id", "name", name="uq_firm_integration"),
    )

    firm_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # Integration name as it appears in config/integrations.yml (e.g. "stock-chat").
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
