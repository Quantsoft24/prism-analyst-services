"""Data access for filings + chunks.

Two repositories:
  * ``FilingRepository``       — CRUD + status transitions on the ``filings`` table
  * ``FilingChunkRepository``  — bulk chunk insert + the raw retrieval queries
                                 (dense, sparse) the HybridRetrievalService fuses

The repositories own SQL only — no business policy, no HTTP. The hybrid
fusion (RRF) lives in ``services/retrieval/`` so the ranking algorithm is
testable in isolation from the database.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.filing import Filing, FilingChunk


@dataclass(slots=True)
class ChunkHit:
    """A single retrieval hit — chunk id + score + enough metadata to cite."""

    chunk_id: uuid.UUID
    filing_id: uuid.UUID
    company_id: uuid.UUID
    section: str
    page_number: int | None
    text: str
    score: float  # raw score from the search method (distance or ts_rank)


class FilingRepository:
    """CRUD + lifecycle for the ``filings`` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, filing_id: uuid.UUID) -> Filing | None:
        stmt = select(Filing).where(Filing.id == filing_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_by_fingerprint(self, fingerprint: str) -> Filing | None:
        """Idempotency check — has this exact content been ingested already?"""
        stmt = select(Filing).where(Filing.fingerprint == fingerprint)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list_for_company(
        self, company_id: uuid.UUID, *, filing_type: str | None = None
    ) -> list[Filing]:
        stmt = select(Filing).where(Filing.company_id == company_id)
        if filing_type:
            stmt = stmt.where(Filing.filing_type == filing_type)
        stmt = stmt.order_by(Filing.filed_at.desc().nullslast())
        return list((await self._session.execute(stmt)).scalars().all())

    async def add(self, filing: Filing) -> Filing:
        self._session.add(filing)
        await self._session.flush()
        return filing

    async def set_status(
        self, filing_id: uuid.UUID, status: str, *, error: str | None = None
    ) -> None:
        """Move a filing through its parse lifecycle (pending→parsing→parsed|failed)."""
        stmt = (
            update(Filing)
            .where(Filing.id == filing_id)
            .values(parsed_status=status, parse_error=error)
        )
        await self._session.execute(stmt)


class FilingChunkRepository:
    """Bulk insert + raw dense/sparse retrieval queries."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_add(self, chunks: list[FilingChunk]) -> None:
        """Insert many chunks. Caller sets embedding + metadata; text_tsv is
        generated server-side."""
        self._session.add_all(chunks)
        await self._session.flush()

    async def count_for_filing(self, filing_id: uuid.UUID) -> int:
        stmt = select(func.count(FilingChunk.id)).where(FilingChunk.filing_id == filing_id)
        return (await self._session.execute(stmt)).scalar_one()

    # ── Dense (vector) search ────────────────────────────────────────────

    async def search_dense(
        self,
        query_embedding: list[float],
        *,
        company_id: uuid.UUID | None = None,
        section: str | None = None,
        limit: int = 50,
    ) -> list[ChunkHit]:
        """Cosine-distance ANN search over the HNSW index.

        Returns hits ordered by ascending distance (closest first). ``score``
        is ``1 - cosine_distance`` so higher = more similar (consistent with
        the sparse path, which also returns higher = better).
        """
        # pgvector cosine distance operator is ``<=>``. We compute similarity
        # as ``1 - distance`` for an intuitive, fusion-friendly score.
        distance = FilingChunk.embedding.cosine_distance(query_embedding)
        stmt = select(
            FilingChunk.id,
            FilingChunk.filing_id,
            FilingChunk.company_id,
            FilingChunk.section,
            FilingChunk.page_number,
            FilingChunk.text,
            distance.label("distance"),
        ).where(FilingChunk.embedding.isnot(None))
        if company_id is not None:
            stmt = stmt.where(FilingChunk.company_id == company_id)
        if section is not None:
            stmt = stmt.where(FilingChunk.section == section)
        stmt = stmt.order_by(distance.asc()).limit(limit)

        rows = (await self._session.execute(stmt)).all()
        return [
            ChunkHit(
                chunk_id=r.id,
                filing_id=r.filing_id,
                company_id=r.company_id,
                section=r.section,
                page_number=r.page_number,
                text=r.text,
                score=1.0 - float(r.distance),
            )
            for r in rows
        ]

    # ── Sparse (full-text) search ────────────────────────────────────────

    async def search_sparse(
        self,
        query: str,
        *,
        company_id: uuid.UUID | None = None,
        section: str | None = None,
        limit: int = 50,
    ) -> list[ChunkHit]:
        """BM25-like full-text search via Postgres ``ts_rank_cd`` over the GIN
        index. Uses ``websearch_to_tsquery`` so the analyst can type natural
        queries ("revenue growth guidance") without tsquery syntax."""
        tsquery = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank_cd(FilingChunk.text_tsv, tsquery)
        stmt = (
            select(
                FilingChunk.id,
                FilingChunk.filing_id,
                FilingChunk.company_id,
                FilingChunk.section,
                FilingChunk.page_number,
                FilingChunk.text,
                rank.label("rank"),
            )
            .where(FilingChunk.text_tsv.op("@@")(tsquery))
        )
        if company_id is not None:
            stmt = stmt.where(FilingChunk.company_id == company_id)
        if section is not None:
            stmt = stmt.where(FilingChunk.section == section)
        stmt = stmt.order_by(rank.desc()).limit(limit)

        rows = (await self._session.execute(stmt)).all()
        return [
            ChunkHit(
                chunk_id=r.id,
                filing_id=r.filing_id,
                company_id=r.company_id,
                section=r.section,
                page_number=r.page_number,
                text=r.text,
                score=float(r.rank),
            )
            for r in rows
        ]
