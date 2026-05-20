"""Filings + hybrid-search endpoints.

  GET  /api/v1/filings/{ticker}       — list ingested filings for a company
  POST /api/v1/search                 — hybrid retrieval over filing chunks

The search endpoint is the same retrieval the agent's ``retrieve_filings``
tool uses — exposed over HTTP so a third-party API consumer (or our own
frontend) can run grounded search directly. Versioned + OpenAPI-documented
per the API-first architecture.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.auth import get_current_firm_id
from src.core.database import get_session
from src.repositories.company_repo import CompanyRepository
from src.repositories.filing_repo import FilingRepository
from src.schemas.filings import FilingRead, SearchHit, SearchRequest, SearchResponse
from src.services.retrieval import HybridRetrievalService

router = APIRouter(tags=["Filings"])


@router.get(
    "/filings/{ticker}",
    response_model=list[FilingRead],
    summary="List ingested filings for a company",
)
async def list_filings(
    ticker: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    filing_type: str | None = None,
) -> list[FilingRead]:
    _ = firm_id
    companies = CompanyRepository(session)
    company = await companies.get_by_ticker(ticker.upper())
    if company is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company {ticker!r} not found.",
        )
    filings = FilingRepository(session)
    rows = await filings.list_for_company(company.id, filing_type=filing_type)
    return [FilingRead.model_validate(r) for r in rows]


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Hybrid search over ingested filings",
    description=(
        "Dense (pgvector) + sparse (Postgres FTS) retrieval fused with RRF. "
        "Optionally scope to a ticker and/or filing section. This is the same "
        "retrieval the agent uses internally — exposed for direct/API use."
    ),
)
async def search_filings(
    body: SearchRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> SearchResponse:
    _ = firm_id
    company_id = None
    if body.ticker:
        companies = CompanyRepository(session)
        company = await companies.get_by_ticker(body.ticker.upper())
        if company is None:
            return SearchResponse(
                query=body.query,
                count=0,
                hits=[],
                note=f"{body.ticker} is not in PRISM's coverage universe.",
            )
        company_id = company.id

    retriever = HybridRetrievalService(session)
    results = await retriever.retrieve(
        body.query, company_id=company_id, section=body.section, limit=body.limit
    )

    if not results:
        return SearchResponse(
            query=body.query,
            count=0,
            hits=[],
            note="No matching filing chunks. Filing data may not be ingested yet.",
        )

    return SearchResponse(
        query=body.query,
        count=len(results),
        hits=[
            SearchHit(
                filing_id=r.filing_id,
                section=r.section,
                page=r.page_number,
                text=r.text,
                fused_score=r.fused_score,
                dense_rank=r.dense_rank,
                sparse_rank=r.sparse_rank,
            )
            for r in results
        ],
    )
