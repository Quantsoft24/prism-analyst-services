"""Billing & subscriptions — SCHEMA ONLY (no payment provider yet).

Provider-agnostic by design: ``Subscription.external_ref`` holds the
Razorpay/Stripe id once we integrate (P4), and ``currency`` is on both the plan
and the subscription so India-first INR can coexist with later USD/global plans.
The access policy reads ``entitlements`` for ``entitlement:<key>`` gates.

These tables are created now so the data model is stable, but nothing writes to
them until billing is wired — `Firm.subscription_tier` remains the live signal
until then. See final_docs/12_AUTH_USER_PROFILES_BILLING.md §5.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import Boolean, DateTime, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin, UUIDPKMixin


class Plan(UUIDPKMixin, TimestampMixin, Base):
    """A purchasable plan (free / trial / pro / …). Currency-aware."""

    __tablename__ = "plans"

    key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, server_default="INR")
    amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))  # null = free
    interval: Mapped[str] = mapped_column(String(16), nullable=False, server_default="month")
    features: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")

    def __repr__(self) -> str:
        return f"<Plan {self.key!r} {self.currency} {self.amount}>"


class Subscription(UUIDPKMixin, TimestampMixin, Base):
    """A firm's subscription to a plan. ``external_ref`` = provider id (later)."""

    __tablename__ = "subscriptions"

    firm_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)  # firm slug
    plan_key: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, server_default="trialing")
    currency: Mapped[str] = mapped_column(String(3), nullable=False, server_default="INR")
    current_period_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    external_ref: Mapped[str | None] = mapped_column(String(255))  # razorpay/stripe id (P4)
    cancel_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    def __repr__(self) -> str:
        return f"<Subscription firm={self.firm_id!r} plan={self.plan_key!r} {self.status}>"


class Entitlement(UUIDPKMixin, TimestampMixin, Base):
    """A firm's allowance for one feature. ``limit_value`` null = unlimited.
    Read by the access policy for ``entitlement:<feature_key>`` gates."""

    __tablename__ = "entitlements"
    __table_args__ = (
        UniqueConstraint("firm_id", "feature_key", name="uq_entitlements_firm_feature"),
    )

    firm_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)  # firm slug
    feature_key: Mapped[str] = mapped_column(String(64), nullable=False)
    limit_value: Mapped[int | None] = mapped_column(Integer)  # null = unlimited
    source: Mapped[str] = mapped_column(String(32), nullable=False, server_default="plan")

    def __repr__(self) -> str:
        return f"<Entitlement firm={self.firm_id!r} {self.feature_key}={self.limit_value}>"
