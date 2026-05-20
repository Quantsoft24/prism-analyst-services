"""Embedder — batched, dimension-consistent text embedding."""

from __future__ import annotations

from src.config import settings
from src.services.model_router import get_router

# Conservative batch size. Gemini embedding endpoints accept up to ~100 inputs
# per call and ~2048 tokens each; 32 keeps us well within limits and bounds
# the blast radius of a single failed call during bulk ingestion.
_DEFAULT_BATCH_SIZE = 32


class Embedder:
    """Embeds text via the router's ``embedding`` tier.

    Stateless aside from config. Construct once and share, or use the module
    singleton ``get_embedder()``.
    """

    def __init__(self, dimensions: int | None = None, batch_size: int = _DEFAULT_BATCH_SIZE) -> None:
        # None → use the model's native dimension. We pass our configured
        # dimension so the stored vectors always match the pgvector column
        # width (settings.EMBEDDING_DIMENSION).
        self._dimensions = dimensions if dimensions is not None else settings.EMBEDDING_DIMENSION
        self._batch_size = batch_size

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document chunks (ingestion path).

        Batches internally; preserves input order in the output.
        """
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors = await get_router().aembed(batch, dimensions=self._dimensions)
            out.extend(vectors)
        self._validate(out, expected=len(texts))
        return out

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (retrieval path)."""
        vectors = await get_router().aembed([text], dimensions=self._dimensions)
        self._validate(vectors, expected=1)
        return vectors[0]

    def _validate(self, vectors: list[list[float]], *, expected: int) -> None:
        if len(vectors) != expected:
            raise RuntimeError(
                f"Embedder expected {expected} vectors, got {len(vectors)}. "
                "Embedding provider returned a mismatched batch."
            )
        for v in vectors:
            if len(v) != self._dimensions:
                raise RuntimeError(
                    f"Embedding dimension mismatch: got {len(v)}, "
                    f"expected {self._dimensions}. Check the embedding model + "
                    "settings.EMBEDDING_DIMENSION are in sync with the DB column."
                )


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Module singleton — cheap to construct, but shared for consistency."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
