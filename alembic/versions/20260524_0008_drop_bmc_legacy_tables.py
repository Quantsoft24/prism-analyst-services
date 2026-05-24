"""drop the legacy BMC tables (PRISM's RAG-based implementation is retired)

Revision ID: 0008_drop_bmc_legacy_tables
Revises: 0007_firm_integrations
Create Date: 2026-05-24

PRISM's BMC is now an external service (``src/routers/bmc.py`` thin proxies
to ``BMC_URL``). The external service owns its own ``bmc_*`` tables in the
shared Postgres under a separate alembic version. PRISM's old tables here
are dead weight — drop them to keep the schema clean.

The chain stays continuous: 0005 + 0006 + 0007 remain in history; this
revision unwinds 0005 and 0006's table contents (0006 only added a column
to ``bmc_analyses``, dropping the table removes it implicitly).
"""

from __future__ import annotations

from alembic import op

revision: str = "0008_drop_bmc_legacy_tables"
down_revision: str | None = "0007_firm_integrations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # FK chain: bmc_evidence → bmc_blocks → bmc_analyses. Use ``if_exists``
    # so the migration is idempotent (safe to re-run on a DB where the
    # external service has already taken over these table names — though we
    # don't expect that on PRISM's current Neon DB).
    op.execute("DROP TABLE IF EXISTS bmc_evidence CASCADE")
    op.execute("DROP TABLE IF EXISTS bmc_blocks CASCADE")
    op.execute("DROP TABLE IF EXISTS bmc_analyses CASCADE")


def downgrade() -> None:
    # No going back — the data is gone and the ORM models have been deleted.
    # If you really need the old tables, restore them from a DB backup; the
    # 0005/0006 migration files are still in the repo for reference.
    raise RuntimeError(
        "0008 is irreversible — PRISM's RAG BMC code is deleted. Restore from "
        "DB backup if you need the old bmc_* tables back."
    )
