"""Parser factory — returns the configured backend.

Call ``get_parser()`` rather than constructing a parser directly, so the
choice stays centralized in ``settings.PARSER_BACKEND``.
"""

from __future__ import annotations

from src.config import settings
from src.services.ingestion.parser.base import PdfParser


def get_parser(backend: str | None = None) -> PdfParser:
    """Return a parser instance for ``backend`` (default: settings.PARSER_BACKEND).

    Raises ``ValueError`` for an unknown backend name.
    """
    name = (backend or settings.PARSER_BACKEND).lower()

    if name == "pdfplumber":
        from src.services.ingestion.parser.pdfplumber_parser import PdfPlumberParser

        return PdfPlumberParser()

    if name == "docling":
        from src.services.ingestion.parser.docling_parser import DoclingParser

        return DoclingParser()

    raise ValueError(
        f"Unknown PARSER_BACKEND {name!r}. Valid: 'pdfplumber', 'docling'."
    )
