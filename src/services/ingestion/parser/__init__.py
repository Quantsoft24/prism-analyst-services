"""PDF parsing — pluggable backends behind one interface.

Two backends:
  * ``PdfPlumberParser`` — lightweight, in-process, pure Python. Default.
  * ``DoclingParser``    — best-in-class table extraction, runs in a Docker
                           sidecar (heavy ML deps stay out of the backend).

``get_parser()`` returns the configured backend (``settings.PARSER_BACKEND``).
Both produce the same ``ParsedDocument`` shape, so the chunker + pipeline are
parser-agnostic. Swapping backends is a config flag, never a code change.
"""

from src.services.ingestion.parser.base import (
    ParsedDocument,
    ParsedPage,
    PdfParser,
)
from src.services.ingestion.parser.factory import get_parser

__all__ = ["PdfParser", "ParsedDocument", "ParsedPage", "get_parser"]
