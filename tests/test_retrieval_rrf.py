"""Unit tests for Reciprocal Rank Fusion — pure function, no DB/network."""

from __future__ import annotations

import uuid

from src.repositories.filing_repo import ChunkHit
from src.services.retrieval.hybrid import reciprocal_rank_fusion


def _hit(text: str, score: float) -> ChunkHit:
    return ChunkHit(
        chunk_id=uuid.uuid4(),
        filing_id=uuid.uuid4(),
        company_id=uuid.uuid4(),
        section="general",
        page_number=1,
        text=text,
        score=score,
    )


def test_empty_inputs_return_empty():
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_list_preserves_order():
    a, b, c = _hit("a", 0.9), _hit("b", 0.8), _hit("c", 0.7)
    fused = reciprocal_rank_fusion([[a, b, c]], k=60, limit=10)
    assert [r.chunk_id for r in fused] == [a.chunk_id, b.chunk_id, c.chunk_id]
    # rank-1 item has the highest RRF score (1/(60+1))
    assert fused[0].fused_score > fused[1].fused_score > fused[2].fused_score


def test_document_in_both_lists_ranks_higher():
    """A chunk that appears in BOTH dense and sparse should beat one that
    only appears in a single list, even at a worse individual rank."""
    shared = _hit("shared", 0.5)
    dense_only = _hit("dense_only", 0.99)
    sparse_only = _hit("sparse_only", 0.99)

    dense = [dense_only, shared]      # shared at rank 2 in dense
    sparse = [sparse_only, shared]    # shared at rank 2 in sparse

    fused = reciprocal_rank_fusion([dense, sparse], k=60, limit=10)
    # shared appears in both → score = 1/(60+2) + 1/(60+2) = 2/62
    # dense_only / sparse_only appear once → 1/(60+1) = 1/61
    # 2/62 (0.03226) > 1/61 (0.01639) → shared wins
    assert fused[0].chunk_id == shared.chunk_id
    assert fused[0].dense_rank == 2
    assert fused[0].sparse_rank == 2


def test_limit_truncates_results():
    hits = [_hit(f"h{i}", 1.0 - i * 0.01) for i in range(20)]
    fused = reciprocal_rank_fusion([hits], k=60, limit=5)
    assert len(fused) == 5


def test_component_ranks_recorded():
    a = _hit("a", 0.9)
    b = _hit("b", 0.8)
    # a is rank-1 dense, rank-2 sparse; b is rank-2 dense, rank-1 sparse.
    fused = reciprocal_rank_fusion([[a, b], [b, a]], k=60, limit=10)
    by_id = {r.chunk_id: r for r in fused}
    assert by_id[a.chunk_id].dense_rank == 1
    assert by_id[a.chunk_id].sparse_rank == 2
    assert by_id[b.chunk_id].dense_rank == 2
    assert by_id[b.chunk_id].sparse_rank == 1


def test_deterministic_tiebreak():
    """Equal scores must break ties deterministically (by chunk_id) so results
    are stable across runs — important for cached BMC diffs later."""
    a = _hit("a", 0.5)
    b = _hit("b", 0.5)
    fused1 = reciprocal_rank_fusion([[a], [b]], k=60, limit=10)
    fused2 = reciprocal_rank_fusion([[a], [b]], k=60, limit=10)
    assert [r.chunk_id for r in fused1] == [r.chunk_id for r in fused2]
