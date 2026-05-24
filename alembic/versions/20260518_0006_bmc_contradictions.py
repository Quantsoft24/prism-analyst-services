"""add contradictions column to bmc_analyses (Phase 3 CrossBlockReconciler)

Revision ID: 0006_bmc_contradictions
Revises: 0005_bmc_tables
Create Date: 2026-05-18
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006_bmc_contradictions"
down_revision: str | None = "0005_bmc_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "bmc_analyses",
        sa.Column(
            "contradictions",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("bmc_analyses", "contradictions")
