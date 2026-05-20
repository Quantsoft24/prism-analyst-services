"""Ingestion — fetch → parse → chunk → embed → store pipeline.

Slice 5A ships only the **declarative source registry** (``FilingsRegistry``).
The actual pipeline stages (docling parser, semantic chunker, orchestrator)
land in Slice 5B. Keeping the registry separate lets us version + review the
"what to ingest" list independently of the "how to ingest" code.

Architecture note: ingestion is deterministic ETL, deliberately NOT wrapped
in an ADK agent — that matches how Bloomberg / AlphaSense / Perplexity
structure ingestion (data engineering) vs. retrieval+synthesis (agentic).
When scheduled multi-source ingestion becomes operationally critical
(Year 2), these functions get wrapped in Prefect flows without changing
their internals.
"""

from src.services.ingestion.chunker import Chunk, Chunker
from src.services.ingestion.fetcher import FetchedDocument, fetch_document
from src.services.ingestion.parser import (
    ParsedDocument,
    ParsedPage,
    PdfParser,
    get_parser,
)
from src.services.ingestion.pipeline import IngestionResult, IngestionService
from src.services.ingestion.registry import (
    FilingsRegistry,
    IngestionSource,
    load_registry,
)

__all__ = [
    "FilingsRegistry",
    "IngestionSource",
    "load_registry",
    "Chunk",
    "Chunker",
    "PdfParser",
    "ParsedDocument",
    "ParsedPage",
    "get_parser",
    "fetch_document",
    "FetchedDocument",
    "IngestionService",
    "IngestionResult",
]
