"""Tests for IngestionService paths that don't require a live LLM.

The full happy-path (parse → chunk → embed → persist) needs the embedding
provider + parser, so it's validated manually / in the live integration run
when real PDFs are ingested. Here we cover the deterministic guard rails:
unknown company, and fetch failure — both of which short-circuit before any
embedding call.
"""

from __future__ import annotations

import pytest

from src.services.ingestion import IngestionService
from src.services.ingestion.registry import IngestionSource


@pytest.mark.asyncio
async def test_ingest_unknown_company_fails_gracefully(db_session):
    service = IngestionService(db_session)
    src = IngestionSource(
        ticker="NOSUCHCO",
        filing_type="quarterly_result",
        source_url="https://example.com/x.pdf",
    )
    result = await service.ingest(src)
    assert result.status == "failed"
    assert "not in DB" in result.detail
    assert result.filing_id is None


@pytest.mark.asyncio
async def test_ingest_fetch_failure_recorded(db_session):
    """A known company but an unreachable URL → failed result, no crash.

    Uses TCS (seeded) + a port nothing listens on so the fetch fails fast.
    """
    service = IngestionService(db_session)
    src = IngestionSource(
        ticker="TCS",
        filing_type="quarterly_result",
        source_url="http://localhost:59998/nonexistent.pdf",
    )
    result = await service.ingest(src)
    assert result.status == "failed"
    assert "fetch error" in result.detail.lower()
