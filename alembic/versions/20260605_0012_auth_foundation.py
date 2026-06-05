"""auth foundation — user_preferences + billing schema (plans/subscriptions/entitlements)

P0 of the auth/user-profiles feature (final_docs/12). Provider-independent:
adds per-user preferences and the billing/subscription schema. No behaviour
change — these tables are unused until P1 (auth) / P4 (payments) wire them.

Revision ID: 0012_auth_foundation
Revises: 0011_portfolio_persistence
Create Date: 2026-06-05
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0012_auth_foundation"
down_revision: str | None = "0011_portfolio_persistence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Per-user preferences (1:1 with users) ──
    op.create_table(
        "user_preferences",
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("prefs", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    # ── Billing schema (no payment provider yet — schema only) ──
    op.create_table(
        "plans",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("key", sa.String(64), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False, server_default="INR"),
        sa.Column("amount", sa.Numeric(12, 2)),
        sa.Column("interval", sa.String(16), nullable=False, server_default="month"),
        sa.Column("features", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("key", name="uq_plans_key"),
    )
    op.create_index("ix_plans_key", "plans", ["key"])

    op.create_table(
        "subscriptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("firm_id", sa.String(64), nullable=False),
        sa.Column("plan_key", sa.String(64), nullable=False),
        sa.Column("status", sa.String(24), nullable=False, server_default="trialing"),
        sa.Column("currency", sa.String(3), nullable=False, server_default="INR"),
        sa.Column("current_period_start", sa.DateTime(timezone=True)),
        sa.Column("current_period_end", sa.DateTime(timezone=True)),
        sa.Column("external_ref", sa.String(255)),
        sa.Column("cancel_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_subscriptions_firm_id", "subscriptions", ["firm_id"])

    op.create_table(
        "entitlements",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("firm_id", sa.String(64), nullable=False),
        sa.Column("feature_key", sa.String(64), nullable=False),
        sa.Column("limit_value", sa.Integer()),
        sa.Column("source", sa.String(32), nullable=False, server_default="plan"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("firm_id", "feature_key", name="uq_entitlements_firm_feature"),
    )
    op.create_index("ix_entitlements_firm_id", "entitlements", ["firm_id"])


def downgrade() -> None:
    op.drop_index("ix_entitlements_firm_id", table_name="entitlements")
    op.drop_table("entitlements")
    op.drop_index("ix_subscriptions_firm_id", table_name="subscriptions")
    op.drop_table("subscriptions")
    op.drop_index("ix_plans_key", table_name="plans")
    op.drop_table("plans")
    op.drop_table("user_preferences")
