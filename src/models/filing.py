"""ORM models for filings and their semantic chunks.

These are the canonical primary-source records every grounded answer in
PRISM resolves to. The plan addendum (BMC) and the data-flow architecture
in ``final_docs/02_ARCHITECTURE_AND_STACK.md`` both anchor on these tables.

Two tables:
  * ``filings``         — one row per ingested document (PDF / HTML / transcript).
                          Global metadata (not firm-scoped) — same filing
                          serves every firm.
  * ``filing_chunks``   — many rows per filing, the semantic chunks that
                          dense/sparse retrieval operates over.

Why ``company_id`` is denormalized onto ``filing_chunks``:
  The hot retrieval query is ``WHERE company_id = ? ORDER BY embedding <=> ?``.
  Joining ``filings`` for every chunk lookup would hurt — denormalizing the
  one column keeps retrieval fast and the index small.

Why ``text_tsv`` is a *generated* column:
  Postgres FTS works on ``tsvector``. Generating it from ``text`` server-side
  means we can't forget to update it on writes; the GIN index on it stays
  consistent automatically.

Idempotency contract:
  ``filings.fingerprint`` is a content hash (SHA-256 of canonical PDF bytes).
  Re-running ingestion on the same source URL → same fingerprint → unique-
  constraint violation → ingestion skips the rewrite. This is how we make
  the pipeline safe to re-run.
"""

from __future__ import annotations

import uuid
from datetime import date

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    Date,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.config import settings
from src.models.base import Base, TimestampMixin, UUIDPKMixin


# Filing types — kept as plain string values for flexibility (we'll add
# 'concall_transcript', 'agm_resolution', 'sebi_disclosure', etc. in later
# slices). Using a string enum table would add migration friction without
# real type-safety benefit in Python.
FILING_TYPES = {
    "quarterly_result",     # standalone + consolidated quarterly numbers
    "annual_report",        # full AR with MD&A, board's report, financials
    "concall_transcript",   # earnings call Q&A — Slice 5B+
    "investor_presentation",
    "agm_resolution",
    "sebi_disclosure",
}

PARSED_STATUSES = {"pending", "parsing", "parsed", "failed"}

# Filing sections — coarse-grained taxonomy used to boost retrieval by
# query intent (e.g., MD&A for narrative questions, P&L for numbers).
# 'general' is the bucket for unclassified chunks; nothing should land here
# once the section classifier in Slice 5B is robust.
FILING_SECTIONS = {
    "general",
    "mda",                 # Management Discussion & Analysis
    "balance_sheet",
    "profit_loss",
    "cash_flow",
    "notes",               # Notes to Accounts
    "auditors_report",
    "directors_report",
    "risk_factors",
    "related_party",
    "segment_reporting",
}


class Filing(UUIDPKMixin, TimestampMixin, Base):
    """One ingested document. Global metadata, not firm-scoped."""

    __tablename__ = "filings"

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Document classification — see FILING_TYPES set above.
    filing_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Indian-fiscal-year period the filing covers, e.g. "Q3-FY24", "FY24".
    # Free-text on purpose — different filing types use different period
    # granularities (annual vs quarterly vs adhoc).
    fiscal_period: Mapped[str | None] = mapped_column(String(32), index=True)

    # Where the canonical source lives — BSE URL, NSE URL, MCA-21 URL, etc.
    source_url: Mapped[str] = mapped_column(Text, nullable=False)

    # Path into our ObjectStore (fsspec). For local dev that's a relative
    # path under FILINGS_STORAGE_URL; for prod it's an S3 key. The store
    # abstraction in ``services/storage/`` resolves it the same way.
    storage_path: Mapped[str | None] = mapped_column(Text)

    title: Mapped[str | None] = mapped_column(String(512))
    filed_at: Mapped[date | None] = mapped_column(Date, index=True)

    # Pipeline lifecycle. Transitions: pending → parsing → parsed | failed.
    # 'parsing' acts as a soft lock — a worker that crashes mid-parse leaves
    # the row in 'parsing' and the next run sweeps it back to 'pending'.
    parsed_status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="pending"
    )
    parse_error: Mapped[str | None] = mapped_column(Text)

    # Content fingerprint (SHA-256 hex). Idempotency anchor — see module
    # docstring. NOT NULL once content has been downloaded; nullable until
    # then to let us register intent (a registry entry) before the bytes
    # arrive.
    fingerprint: Mapped[str | None] = mapped_column(String(64), unique=True, index=True)

    # MIME type as detected at fetch time. Most filings are
    # ``application/pdf`` but BSE serves some HTML announcements too.
    content_type: Mapped[str | None] = mapped_column(String(128))
    size_bytes: Mapped[int | None] = mapped_column(Integer)

    chunks: Mapped[list[FilingChunk]] = relationship(
        back_populates="filing", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Filing id={self.id} type={self.filing_type!r} "
            f"period={self.fiscal_period!r} status={self.parsed_status!r}>"
        )


class FilingChunk(UUIDPKMixin, TimestampMixin, Base):
    """One semantic chunk of a filing — the unit of retrieval."""

    __tablename__ = "filing_chunks"

    filing_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("filings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Denormalized for retrieval-time filter performance — see module docstring.
    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Position within the filing — preserves reading order on display.
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Section classification — see FILING_SECTIONS set above. Used to boost
    # retrieval by query intent ("show me MD&A on margins" → boost section=mda).
    section: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="general", index=True
    )

    # Page number in the source PDF — citation anchor. Nullable because some
    # filings (HTML announcements) don't have a meaningful page number.
    page_number: Mapped[int | None] = mapped_column(Integer)

    # The actual chunk text. Plain UTF-8, no markdown wrapping (the parser
    # normalizes to clean text before chunking).
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # Token count under the same tokenizer the embedder uses — informs
    # downstream context-window budgeting. Set at chunk time, immutable.
    token_count: Mapped[int | None] = mapped_column(Integer)

    # Sparse representation — a STORED generated column built server-side from
    # ``text`` via ``to_tsvector``. ``Computed(..., persisted=True)`` tells
    # SQLAlchemy it's generated (never inserted/updated by the ORM) and lets
    # the GIN index operate without runtime computation. The 'english' config
    # is a safe default; Indian-language analyzers come in a later slice if we
    # ingest Hindi/Marathi/etc. filings.
    text_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', coalesce(text, ''))",
            persisted=True,
        ),
    )

    # Dense representation — embedding from the embedding tier of ModelRouter.
    # Dimension is centralized in ``settings.EMBEDDING_DIMENSION``; updating
    # there requires a migration to alter the column type.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.EMBEDDING_DIMENSION)
    )

    filing: Mapped[Filing] = relationship(back_populates="chunks")

    def __repr__(self) -> str:
        return (
            f"<FilingChunk id={self.id} filing={self.filing_id} "
            f"section={self.section!r} idx={self.chunk_index}>"
        )
