"""Ingestion pipeline orchestrator.

Deterministic ETL: for one ``IngestionSource``, run
    fetch → store(raw) → parse → chunk → embed → persist
idempotently. NOT an agent (see ``ingestion/__init__.py`` for the rationale —
matches how Bloomberg/AlphaSense/Perplexity separate ingestion from agentic
synthesis).

Idempotency: keyed on the content fingerprint. If a filing with the same
fingerprint already exists and is ``parsed``, we skip. This makes the CLI
safe to re-run (e.g. after adding new companies to the registry).

Resumability: the ``filings.parsed_status`` column is the state machine —
pending → parsing → parsed | failed. A crash mid-parse leaves 'parsing';
a future sweep can reset stale 'parsing' rows to 'pending'.

Each stage is small + replaceable: swap the parser via config, swap the
embedder via the router, swap storage via fsspec URL. The orchestrator only
sequences them.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.models.company import Company
from src.models.filing import Filing, FilingChunk
from src.repositories.filing_repo import FilingChunkRepository, FilingRepository
from src.services.embedding import get_embedder
from src.services.ingestion.chunker import Chunker
from src.services.ingestion.fetcher import fetch_document
from src.services.ingestion.parser import get_parser
from src.services.ingestion.registry import IngestionSource
from src.services.storage import open_storage

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionResult:
    """Outcome of ingesting one source — for CLI reporting + tests."""

    ticker: str
    status: str          # 'ingested' | 'skipped' | 'failed'
    filing_id: uuid.UUID | None = None
    chunk_count: int = 0
    detail: str = ""


class IngestionService:
    """Runs the ingestion pipeline for individual sources.

    Constructed per session (it holds an AsyncSession). The parser, embedder,
    chunker, and store are resolved from config/singletons so the service
    stays thin.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._filings = FilingRepository(session)
        self._chunks = FilingChunkRepository(session)
        self._parser = get_parser()
        self._chunker = Chunker()
        self._embedder = get_embedder()
        self._store = open_storage(settings.FILINGS_STORAGE_URL)

    async def ingest(self, source: IngestionSource, *, force: bool = False) -> IngestionResult:
        """Ingest one source end-to-end. Idempotent unless ``force=True``."""
        # 1. Resolve the company by ticker.
        company = await self._resolve_company(source.ticker, source.exchange)
        if company is None:
            return IngestionResult(
                ticker=source.ticker,
                status="failed",
                detail=f"Company {source.ticker} ({source.exchange}) not in DB — seed it first.",
            )

        # 2. Fetch raw bytes + fingerprint.
        try:
            fetched = await fetch_document(source.source_url)
        except Exception as exc:  # noqa: BLE001 — record + continue with next source
            logger.warning("Fetch failed for %s: %s", source.ticker, exc)
            return IngestionResult(source.ticker, "failed", detail=f"fetch error: {exc}")

        # 3. Idempotency — skip if this exact content is already parsed.
        existing = await self._filings.get_by_fingerprint(fetched.fingerprint)
        if existing is not None and existing.parsed_status == "parsed" and not force:
            return IngestionResult(
                source.ticker,
                "skipped",
                filing_id=existing.id,
                detail="already ingested (matching fingerprint)",
            )

        # 4. Store the raw PDF (provenance + re-parse without re-download).
        storage_key = f"{company.ticker}/{fetched.fingerprint[:16]}.pdf"
        stored = await self._store.put(
            storage_key, fetched.content, content_type=fetched.content_type
        )

        # 5. Create (or reuse) the filing row, mark 'parsing'.
        filing = existing or Filing(
            company_id=company.id,
            filing_type=source.filing_type,
            fiscal_period=source.fiscal_period,
            source_url=source.source_url,
            title=source.title,
            filed_at=source.filed_at,
            fingerprint=fetched.fingerprint,
            content_type=fetched.content_type,
            size_bytes=fetched.size_bytes,
        )
        filing.storage_path = stored.key
        if existing is None:
            await self._filings.add(filing)
        await self._filings.set_status(filing.id, "parsing")

        # 6. Parse → chunk → embed → persist.
        try:
            parsed = await self._parser.parse(fetched.content, filename=storage_key)
            chunks = self._chunker.chunk_document(parsed)
            if not chunks:
                await self._filings.set_status(filing.id, "failed", error="no chunks produced")
                return IngestionResult(source.ticker, "failed", filing.id, detail="no chunks")

            embeddings = await self._embedder.embed_documents([c.text for c in chunks])

            # Clear any prior chunks (re-ingest path) then bulk insert fresh.
            await self._delete_existing_chunks(filing.id)
            rows = [
                FilingChunk(
                    filing_id=filing.id,
                    company_id=company.id,
                    chunk_index=c.chunk_index,
                    section=c.section,
                    page_number=c.page_number,
                    text=c.text,
                    token_count=c.token_count,
                    embedding=emb,
                )
                for c, emb in zip(chunks, embeddings, strict=True)
            ]
            await self._chunks.bulk_add(rows)
            await self._filings.set_status(filing.id, "parsed")

            return IngestionResult(
                source.ticker,
                "ingested",
                filing_id=filing.id,
                chunk_count=len(rows),
                detail=f"parsed via {parsed.parser_backend}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ingestion failed for %s", source.ticker)
            await self._filings.set_status(filing.id, "failed", error=str(exc)[:1000])
            return IngestionResult(source.ticker, "failed", filing.id, detail=str(exc))

    # ── Helpers ──────────────────────────────────────────────────────────

    async def _resolve_company(self, ticker: str, exchange: str) -> Company | None:
        stmt = select(Company).where(
            Company.ticker == ticker.upper(), Company.exchange == exchange.upper()
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def _delete_existing_chunks(self, filing_id: uuid.UUID) -> None:
        """Remove prior chunks for a filing before re-inserting (re-ingest)."""
        from sqlalchemy import delete

        await self._session.execute(
            delete(FilingChunk).where(FilingChunk.filing_id == filing_id)
        )
