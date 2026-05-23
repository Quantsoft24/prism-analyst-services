"""firm_integrations — per-firm enable/disable overrides for agent integrations

Revision ID: 0007_firm_integrations
Revises: 0006_bmc_contradictions
Create Date: 2026-05-23
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007_firm_integrations"
down_revision: str | None = "0006_bmc_contradictions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "firm_integrations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("firm_id", sa.String(64), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("firm_id", "name", name="uq_firm_integration"),
    )
    op.create_index("ix_firm_integrations_firm_id", "firm_integrations", ["firm_id"])


def downgrade() -> None:
    op.drop_index("ix_firm_integrations_firm_id", table_name="firm_integrations")
    op.drop_table("firm_integrations")
