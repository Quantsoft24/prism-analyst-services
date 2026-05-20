"""Retrieval — hybrid dense + sparse search with Reciprocal Rank Fusion.

This is the "R" in our RAG. Given an analyst query, it:
  1. embeds the query (dense) and runs ANN search over pgvector
  2. runs full-text (sparse / BM25-like) search over Postgres FTS
  3. fuses the two ranked lists with RRF
  4. returns the top-K fused chunks for the agent to ground its answer

The fusion math (``reciprocal_rank_fusion``) is a pure function with no DB or
network dependency, so it's unit-testable in isolation — see
``tests/test_retrieval_rrf.py``.
"""

from src.services.retrieval.hybrid import (
    HybridRetrievalService,
    RetrievedChunk,
    reciprocal_rank_fusion,
)

__all__ = ["HybridRetrievalService", "RetrievedChunk", "reciprocal_rank_fusion"]
