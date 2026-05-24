"""drop PRISM's legacy `companies` + `filings` stacks (read-on-demand cutover)

Revision ID: 0009_drop_companies_and_filings
Revises: 0008_drop_bmc_legacy_tables
Create Date: 2026-05-24

The company catalog now comes from the catalog DB's ``company_industry`` table
(4,773 rows). Filing narrative Q&A comes from the stock-chat service's
read-on-demand path over ``filings_index`` + ``document_texts``. PRISM no
longer needs its own copies — drop them.

Tables dropped here:
  * filing_chunks       (PRISM's RAG embedding chunks)
  * filings             (PRISM's curated filing rows)
  * company_aliases     (PRISM's alias map)
  * companies           (PRISM's curated company table)

FK order respected: filing_chunks → filings, company_aliases → companies.
Idempotent (``IF EXISTS``) so a re-run on a fresh DB doesn't error.
"""

from __future__ import annotations

from alembic import op

revision: str = "0009_drop_companies_and_filings"
down_revision: str | None = "0008_drop_bmc_legacy_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS filing_chunks CASCADE")
    op.execute("DROP TABLE IF EXISTS filings CASCADE")
    op.execute("DROP TABLE IF EXISTS company_aliases CASCADE")
    op.execute("DROP TABLE IF EXISTS companies CASCADE")


def downgrade() -> None:
    raise RuntimeError(
        "0009 is irreversible — PRISM's RAG/company code was deleted with this "
        "migration. The data lives in the external stock_chat Postgres now. "
        "Restore from DB backup if you need the old tables back."
    )
