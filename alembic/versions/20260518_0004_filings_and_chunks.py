"""filings + filing_chunks — RAG foundation (pgvector + FTS)

Revision ID: 0004_filings_and_chunks
Revises: 0003_agent_runs
Create Date: 2026-05-18

Enables the ``vector`` extension and creates the two tables the RAG layer
operates over:
  * ``filings``        — one row per ingested document
  * ``filing_chunks``  — semantic chunks with a dense ``embedding`` (pgvector)
                         and a stored ``text_tsv`` (Postgres FTS)

Indexes created:
  * HNSW on ``filing_chunks.embedding`` (cosine) — fast approximate NN search
  * GIN  on ``filing_chunks.text_tsv``           — fast full-text (BM25-like)
  * btree on (company_id, section)               — filtered retrieval

Note on HNSW: building the index on an empty table is instant. When we later
bulk-ingest, inserts pay a small index-maintenance cost — acceptable for our
write volume. If bulk loads ever get slow, the standard trick is to drop the
HNSW index, bulk insert, then recreate — but we're nowhere near that scale.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_filings_and_chunks"
down_revision: str | None = "0003_agent_runs"
branch_labels = None
depends_on = None

# Must match settings.EMBEDDING_DIMENSION. Hardcoded here because migrations
# must be deterministic snapshots — they should NOT import runtime settings
# that could drift. If you change the embedding dimension, write a new
# migration that ALTERs the column.
EMBEDDING_DIM = 768


def upgrade() -> None:
    # pgvector extension — provides the ``vector`` column type + distance ops.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── filings ───────────────────────────────────────────────────────────
    op.create_table(
        "filings",
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
        sa.Column("filing_type", sa.String(64), nullable=False),
        sa.Column("fiscal_period", sa.String(32)),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("storage_path", sa.Text()),
        sa.Column("title", sa.String(512)),
        sa.Column("filed_at", sa.Date()),
        sa.Column("parsed_status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("parse_error", sa.Text()),
        sa.Column("fingerprint", sa.String(64), unique=True),
        sa.Column("content_type", sa.String(128)),
        sa.Column("size_bytes", sa.Integer()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_filings_company_id", "filings", ["company_id"])
    op.create_index("ix_filings_filing_type", "filings", ["filing_type"])
    op.create_index("ix_filings_fiscal_period", "filings", ["fiscal_period"])
    op.create_index("ix_filings_filed_at", "filings", ["filed_at"])
    op.create_index("ix_filings_fingerprint", "filings", ["fingerprint"], unique=True)

    # ── filing_chunks ─────────────────────────────────────────────────────
    op.create_table(
        "filing_chunks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "filing_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("filings.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "company_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("companies.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(32), nullable=False, server_default="general"),
        sa.Column("page_number", sa.Integer()),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer()),
        # Stored generated tsvector column — built from ``text`` server-side.
        sa.Column(
            "text_tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', coalesce(text, ''))", persisted=True),
        ),
        # pgvector embedding column.
        sa.Column("embedding", postgresql.ARRAY(sa.Float())),  # placeholder; altered below
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    # Replace the placeholder ARRAY column with a real pgvector column. We do
    # it via raw SQL because Alembic's autogenerate doesn't natively know the
    # ``vector`` type. (Using ARRAY first then ALTER keeps the create_table
    # call dialect-portable for offline SQL generation.)
    op.execute(f"ALTER TABLE filing_chunks DROP COLUMN embedding")
    op.execute(f"ALTER TABLE filing_chunks ADD COLUMN embedding vector({EMBEDDING_DIM})")

    op.create_index("ix_filing_chunks_filing_id", "filing_chunks", ["filing_id"])
    op.create_index("ix_filing_chunks_company_id", "filing_chunks", ["company_id"])
    op.create_index("ix_filing_chunks_section", "filing_chunks", ["section"])
    op.create_index(
        "ix_filing_chunks_company_section", "filing_chunks", ["company_id", "section"]
    )

    # GIN index for full-text (sparse) search.
    op.execute(
        "CREATE INDEX ix_filing_chunks_text_tsv ON filing_chunks USING GIN (text_tsv)"
    )

    # HNSW index for dense (vector) search, cosine distance. m / ef_construction
    # are pgvector defaults that work well up to millions of vectors.
    op.execute(
        "CREATE INDEX ix_filing_chunks_embedding_hnsw "
        "ON filing_chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.drop_table("filing_chunks")
    op.drop_table("filings")
    # Leave the ``vector`` extension installed — other tables may use it and
    # dropping a shared extension on downgrade is destructive.
