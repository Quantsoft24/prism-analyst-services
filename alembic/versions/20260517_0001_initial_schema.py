"""initial schema — firms, users, firm_memberships, companies, company_aliases

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-17

Phase 1 core schema:
  - ``firms`` (tenants)
  - ``users`` + ``firm_memberships`` (people, with multi-firm support)
  - ``companies`` + ``company_aliases`` (BSE/NSE listed entities, global metadata)

pgvector is *not* enabled here — that lands in the Slice 4 migration when we
add ``chunks`` for retrieval.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: str | None = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Required PG extensions ────────────────────────────────────────────
    # ``pgcrypto`` for ``gen_random_uuid()``. ``vector`` will be added later.
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # ── firms ────────────────────────────────────────────────────────────
    op.create_table(
        "firms",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("slug", sa.String(64), nullable=False, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("subscription_tier", sa.String(32), nullable=False, server_default="trial"),
        sa.Column("country", sa.String(2), nullable=False, server_default="IN"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_firms_slug", "firms", ["slug"], unique=True)

    # ── users ────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("full_name", sa.String(255), nullable=False, server_default=""),
        sa.Column("external_id", sa.String(255), unique=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_external_id", "users", ["external_id"], unique=True)

    # ── firm_memberships ──────────────────────────────────────────────────
    op.create_table(
        "firm_memberships",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "firm_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("firms.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(32), nullable=False, server_default="member"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("firm_id", "user_id", name="uq_firm_memberships_firm_user"),
    )
    op.create_index("ix_firm_memberships_firm_id", "firm_memberships", ["firm_id"])
    op.create_index("ix_firm_memberships_user_id", "firm_memberships", ["user_id"])

    # ── companies ────────────────────────────────────────────────────────
    op.create_table(
        "companies",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("ticker", sa.String(32), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("legal_name", sa.String(512)),
        sa.Column("exchange", sa.String(16), nullable=False, server_default="NSE"),
        sa.Column("isin", sa.String(12), unique=True),
        sa.Column("cin", sa.String(32)),
        sa.Column("pan", sa.String(10)),
        sa.Column("sector", sa.String(128)),
        sa.Column("industry", sa.String(128)),
        sa.Column("country", sa.String(2), nullable=False, server_default="IN"),
        sa.Column("website", sa.String(255)),
        sa.Column("description", sa.Text()),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("exchange", "ticker", name="uq_companies_exchange_ticker"),
    )
    op.create_index("ix_companies_ticker", "companies", ["ticker"])
    op.create_index("ix_companies_name", "companies", ["name"])
    op.create_index("ix_companies_isin", "companies", ["isin"], unique=True)
    op.create_index("ix_companies_cin", "companies", ["cin"])
    op.create_index("ix_companies_sector", "companies", ["sector"])

    # ── company_aliases ───────────────────────────────────────────────────
    op.create_table(
        "company_aliases",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "company_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("companies.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("value", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("kind", "value", name="uq_company_aliases_kind_value"),
    )
    op.create_index("ix_company_aliases_company_id", "company_aliases", ["company_id"])
    op.create_index("ix_company_aliases_value", "company_aliases", ["value"])


def downgrade() -> None:
    op.drop_table("company_aliases")
    op.drop_table("companies")
    op.drop_table("firm_memberships")
    op.drop_table("users")
    op.drop_table("firms")
    # We intentionally do NOT drop the pgcrypto extension on downgrade —
    # other databases on the same Postgres instance may use it.
