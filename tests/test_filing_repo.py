"""End-to-end tests for filing + chunk repositories against real Postgres.

Embeddings here are deterministic fakes (we don't call the LLM) — the dense
search just needs *some* vectors in the column to exercise the pgvector
cosine operator + HNSW path. Query-time embedding (which needs the router)
is covered separately when the live integration test runs.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from src.config import settings
from src.models.company import Company
from src.models.filing import Filing, FilingChunk
from src.repositories.filing_repo import FilingChunkRepository, FilingRepository


async def _tcs_company_id(db_session) -> uuid.UUID:
    """Resolve the seeded TCS company id."""
    row = (
        await db_session.execute(select(Company).where(Company.ticker == "TCS"))
    ).scalar_one()
    return row.id


def _fake_vec(seed: float) -> list[float]:
    """Deterministic unit-ish vector of the configured dimension.

    We vary one leading component by ``seed`` so different chunks have
    different cosine distances to a query vector.
    """
    dim = settings.EMBEDDING_DIMENSION
    v = [0.01] * dim
    v[0] = seed
    return v


@pytest.mark.asyncio
async def test_filing_crud_and_fingerprint_idempotency(db_session):
    company_id = await _tcs_company_id(db_session)
    repo = FilingRepository(db_session)

    filing = Filing(
        company_id=company_id,
        filing_type="quarterly_result",
        fiscal_period="Q4-FY26",
        source_url="https://example.com/tcs-q4fy26.pdf",
        title="TCS Q4 FY26",
        fingerprint="abc123fingerprint",
        parsed_status="pending",
    )
    await repo.add(filing)

    # Idempotency lookup by fingerprint works.
    found = await repo.get_by_fingerprint("abc123fingerprint")
    assert found is not None
    assert found.id == filing.id

    # Status transition.
    await repo.set_status(filing.id, "parsed")
    reloaded = await repo.get_by_id(filing.id)
    assert reloaded.parsed_status == "parsed"


@pytest.mark.asyncio
async def test_list_for_company_filters_by_type(db_session):
    company_id = await _tcs_company_id(db_session)
    repo = FilingRepository(db_session)

    await repo.add(Filing(company_id=company_id, filing_type="quarterly_result",
                          source_url="https://e.com/q.pdf", fingerprint="fp-q"))
    await repo.add(Filing(company_id=company_id, filing_type="annual_report",
                          source_url="https://e.com/a.pdf", fingerprint="fp-a"))

    quarterly = await repo.list_for_company(company_id, filing_type="quarterly_result")
    assert all(f.filing_type == "quarterly_result" for f in quarterly)
    assert len(quarterly) >= 1


@pytest.mark.asyncio
async def test_chunk_insert_and_sparse_search(db_session):
    company_id = await _tcs_company_id(db_session)
    frepo = FilingRepository(db_session)
    crepo = FilingChunkRepository(db_session)

    filing = await frepo.add(Filing(
        company_id=company_id, filing_type="quarterly_result",
        source_url="https://e.com/tcs.pdf", fingerprint="fp-sparse",
    ))

    chunks = [
        FilingChunk(
            filing_id=filing.id, company_id=company_id, chunk_index=0,
            section="mda", page_number=4,
            text="Revenue grew 11.8 percent year over year driven by BFSI demand.",
            embedding=_fake_vec(0.9),
        ),
        FilingChunk(
            filing_id=filing.id, company_id=company_id, chunk_index=1,
            section="notes", page_number=12,
            text="The company declared a final dividend of rupees 30 per share.",
            embedding=_fake_vec(0.1),
        ),
    ]
    await crepo.bulk_add(chunks)
    await db_session.flush()

    assert await crepo.count_for_filing(filing.id) == 2

    # Sparse search for "revenue growth" should rank the MD&A chunk first.
    hits = await crepo.search_sparse("revenue growth", company_id=company_id, limit=10)
    assert len(hits) >= 1
    assert "Revenue grew" in hits[0].text


@pytest.mark.asyncio
async def test_chunk_dense_search_orders_by_similarity(db_session):
    company_id = await _tcs_company_id(db_session)
    frepo = FilingRepository(db_session)
    crepo = FilingChunkRepository(db_session)

    filing = await frepo.add(Filing(
        company_id=company_id, filing_type="quarterly_result",
        source_url="https://e.com/tcs2.pdf", fingerprint="fp-dense",
    ))

    near = FilingChunk(
        filing_id=filing.id, company_id=company_id, chunk_index=0,
        section="mda", text="near chunk", embedding=_fake_vec(1.0),
    )
    far = FilingChunk(
        filing_id=filing.id, company_id=company_id, chunk_index=1,
        section="mda", text="far chunk", embedding=_fake_vec(-1.0),
    )
    await crepo.bulk_add([near, far])
    await db_session.flush()

    # Query vector aligned with ``near``.
    query_vec = _fake_vec(1.0)
    hits = await crepo.search_dense(query_vec, company_id=company_id, limit=10)
    assert len(hits) == 2
    # Highest similarity score first; the ``near`` chunk should win.
    assert hits[0].text == "near chunk"
    assert hits[0].score >= hits[1].score


@pytest.mark.asyncio
async def test_section_filter_on_search(db_session):
    company_id = await _tcs_company_id(db_session)
    frepo = FilingRepository(db_session)
    crepo = FilingChunkRepository(db_session)

    filing = await frepo.add(Filing(
        company_id=company_id, filing_type="quarterly_result",
        source_url="https://e.com/tcs3.pdf", fingerprint="fp-section",
    ))
    await crepo.bulk_add([
        FilingChunk(filing_id=filing.id, company_id=company_id, chunk_index=0,
                    section="mda", text="margins improved on operating leverage",
                    embedding=_fake_vec(0.5)),
        FilingChunk(filing_id=filing.id, company_id=company_id, chunk_index=1,
                    section="notes", text="margins note disclosure",
                    embedding=_fake_vec(0.5)),
    ])
    await db_session.flush()

    hits = await crepo.search_sparse("margins", company_id=company_id, section="mda", limit=10)
    assert len(hits) >= 1
    assert all(h.section == "mda" for h in hits)
