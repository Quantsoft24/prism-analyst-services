"""Hybrid retrieval — dense + sparse, fused with Reciprocal Rank Fusion."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.repositories.filing_repo import ChunkHit, FilingChunkRepository
from src.services.embedding import get_embedder


@dataclass(slots=True)
class RetrievedChunk:
    """A fused, ranked retrieval result — what the agent's retrieve() tool returns."""

    chunk_id: uuid.UUID
    filing_id: uuid.UUID
    company_id: uuid.UUID
    section: str
    page_number: int | None
    text: str
    fused_score: float
    # Component scores, for transparency / debugging / the "show your work" UX.
    dense_rank: int | None
    sparse_rank: int | None


def reciprocal_rank_fusion(
    ranked_lists: list[list[ChunkHit]],
    *,
    k: int = 60,
    limit: int = 10,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked lists into one via Reciprocal Rank Fusion.

    RRF score for a document = sum over each list of ``1 / (k + rank)``,
    where ``rank`` is 1-based position in that list. Documents appearing high
    in multiple lists rise to the top. ``k`` (default 60, from the original
    RRF paper) damps the influence of very high ranks so a single list can't
    dominate.

    This is deliberately a PURE function — no DB, no embeddings — so the
    ranking behavior is unit-testable. ``ranked_lists`` is expected as
    ``[dense_hits, sparse_hits]`` (order matters only for tie-break determinism).

    Returns up to ``limit`` fused results, highest score first.
    """
    # chunk_id → accumulator
    scores: dict[uuid.UUID, float] = {}
    meta: dict[uuid.UUID, ChunkHit] = {}
    ranks_per_list: dict[uuid.UUID, list[int | None]] = {}

    num_lists = len(ranked_lists)
    for list_idx, hits in enumerate(ranked_lists):
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            meta.setdefault(hit.chunk_id, hit)
            if hit.chunk_id not in ranks_per_list:
                ranks_per_list[hit.chunk_id] = [None] * num_lists
            ranks_per_list[hit.chunk_id][list_idx] = rank

    # Sort by fused score desc; tie-break by chunk_id for determinism.
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))

    results: list[RetrievedChunk] = []
    for chunk_id, score in ordered[:limit]:
        hit = meta[chunk_id]
        ranks = ranks_per_list[chunk_id]
        results.append(
            RetrievedChunk(
                chunk_id=hit.chunk_id,
                filing_id=hit.filing_id,
                company_id=hit.company_id,
                section=hit.section,
                page_number=hit.page_number,
                text=hit.text,
                fused_score=round(score, 6),
                dense_rank=ranks[0] if num_lists > 0 else None,
                sparse_rank=ranks[1] if num_lists > 1 else None,
            )
        )
    return results


class HybridRetrievalService:
    """Orchestrates dense + sparse search and RRF fusion over filing chunks."""

    def __init__(self, session: AsyncSession) -> None:
        self._chunks = FilingChunkRepository(session)

    async def retrieve(
        self,
        query: str,
        *,
        company_id: uuid.UUID | None = None,
        section: str | None = None,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        """Hybrid retrieve for ``query``, optionally scoped to a company/section.

        Steps:
          1. Embed the query → dense ANN search (pgvector HNSW).
          2. Full-text search (Postgres FTS / BM25-like).
          3. Fuse with RRF → top-K.

        Returns ``[]`` cleanly when nothing matches (e.g. no filings ingested
        yet for the company) — callers must handle the empty case and degrade
        gracefully rather than hallucinate.
        """
        final_limit = limit or settings.RETRIEVAL_TOP_K_FINAL

        # 1. Dense
        query_vec = await get_embedder().embed_query(query)
        dense_hits = await self._chunks.search_dense(
            query_vec,
            company_id=company_id,
            section=section,
            limit=settings.RETRIEVAL_TOP_K_DENSE,
        )

        # 2. Sparse
        sparse_hits = await self._chunks.search_sparse(
            query,
            company_id=company_id,
            section=section,
            limit=settings.RETRIEVAL_TOP_K_SPARSE,
        )

        # 3. Fuse
        return reciprocal_rank_fusion(
            [dense_hits, sparse_hits],
            k=settings.RETRIEVAL_RRF_K,
            limit=final_limit,
        )
