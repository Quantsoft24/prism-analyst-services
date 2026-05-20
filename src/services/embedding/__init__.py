"""Embedding service — turns text into vectors for storage + retrieval.

Thin layer over ``ModelRouter.aembed`` that adds the concerns a raw embedding
call shouldn't carry:
  * batching within provider input limits
  * consistent dimension (Matryoshka truncation to settings.EMBEDDING_DIMENSION)
  * a single ``embed_query`` / ``embed_documents`` interface shared by both
    the ingestion pipeline (Slice 5B) and the retrieval service.

Ingestion embeds *documents*; retrieval embeds the *query*. They're separated
as methods because some embedding models use different task-type hints for
each (Gemini supports ``RETRIEVAL_DOCUMENT`` vs ``RETRIEVAL_QUERY``) — wiring
that distinction in later is a one-line change here, not across call sites.
"""

from src.services.embedding.embedder import Embedder, get_embedder

__all__ = ["Embedder", "get_embedder"]
