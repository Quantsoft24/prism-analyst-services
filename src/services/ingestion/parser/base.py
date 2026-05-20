"""Parser interface + the structured document shape every backend returns."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(slots=True)
class ParsedPage:
    """One page of a parsed document.

    ``text`` is clean, reading-order text. ``markdown`` (when the backend
    supports it, e.g. docling) preserves tables + headings so the chunker can
    do section detection; pdfplumber leaves it None and the chunker falls
    back to plain-text heuristics.
    """

    page_number: int
    text: str
    markdown: str | None = None


@dataclass(slots=True)
class ParsedDocument:
    """The parser's output — pages + document-level metadata.

    Intentionally backend-agnostic: the chunker and pipeline consume only
    this shape, so neither knows or cares whether pdfplumber or docling
    produced it.
    """

    pages: list[ParsedPage] = field(default_factory=list)
    # Backend that produced this, for provenance in logs / debugging.
    parser_backend: str = ""
    # True when the backend extracted structured tables (docling). The chunker
    # can use richer section detection when this is set.
    has_structure: bool = False
    # Free-form metadata the backend chose to surface (page count, title, etc.).
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """All pages concatenated, page-break separated. Used for fingerprinting
        and as the chunker's input when page-level chunking isn't needed."""
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def page_count(self) -> int:
        return len(self.pages)


class PdfParser(ABC):
    """Turns raw PDF bytes into a ``ParsedDocument``. Backends are async so a
    slow parse (sidecar HTTP, large doc) doesn't block the event loop."""

    backend_name: str = "base"

    @abstractmethod
    async def parse(self, pdf_bytes: bytes, *, filename: str | None = None) -> ParsedDocument:
        """Parse PDF bytes. Raises ``ParseError`` on unrecoverable failure."""


class ParseError(RuntimeError):
    """Raised when a parser cannot extract usable content from a document."""
