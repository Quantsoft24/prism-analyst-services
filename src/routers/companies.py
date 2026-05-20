"""Companies — list and lookup public-company metadata.

These endpoints are the foundation of every other feature: an analyst's
first action in any session is "pick a company." Watchlists, BMC analyses,
reports, and chat all reference ``company_id``.

Endpoints are versioned (``/api/v1/companies``), tenant-aware (every call
resolves a firm_id even though companies themselves are global metadata —
this lets us add per-firm coverage controls later), and OpenAPI-documented
so third-party API consumers can integrate.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.auth import get_current_firm_id
from src.core.database import get_session
from src.repositories.company_repo import CompanyRepository
from src.schemas.common import PageMeta, Paginated
from src.schemas.company import CompanyDetail, CompanyRead

router = APIRouter(prefix="/companies", tags=["Companies"])


@router.get(
    "",
    response_model=Paginated[CompanyRead],
    summary="List companies",
    description=(
        "Paginated list of companies in PRISM's coverage universe. "
        "Filter by ``search`` (matches ticker, name, or any alias), "
        "``sector``, or ``exchange``. Default order is alphabetical by name."
    ),
)
async def list_companies(
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    search: Annotated[str | None, Query(description="Free-text search across ticker/name/aliases.")] = None,
    sector: Annotated[str | None, Query(description="Exact-match sector filter, e.g. 'Financials'.")] = None,
    exchange: Annotated[str | None, Query(description="Exchange code: NSE | BSE.")] = None,
    limit: Annotated[int, Query(ge=1, le=200, description="Max items per page.")] = 25,
    offset: Annotated[int, Query(ge=0, description="Zero-based page offset.")] = 0,
) -> Paginated[CompanyRead]:
    # firm_id is resolved even though companies are global metadata; in
    # Phase 4 we'll use it to enforce per-firm coverage allowlists.
    _ = firm_id  # not yet used; kept to lock the auth contract early

    repo = CompanyRepository(session)
    result = await repo.list(
        search=search, sector=sector, exchange=exchange, limit=limit, offset=offset
    )
    return Paginated[CompanyRead](
        items=[CompanyRead.model_validate(c) for c in result.items],
        page=PageMeta(total=result.total, limit=limit, offset=offset),
    )


@router.get(
    "/{id_or_ticker}",
    response_model=CompanyDetail,
    summary="Get one company by UUID or NSE ticker",
    description=(
        "Resolve a company by either its UUID primary key or its NSE ticker. "
        "Returns full detail including legal name, CIN, aliases, and description."
    ),
    responses={404: {"description": "Company not found."}},
)
async def get_company(
    id_or_ticker: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> CompanyDetail:
    _ = firm_id
    repo = CompanyRepository(session)

    company = None
    # Try UUID first, fall back to ticker — saves a wasted query on the
    # common case of UUID lookups from the frontend's company picker.
    try:
        company_uuid = uuid.UUID(id_or_ticker)
        company = await repo.get_by_id(company_uuid)
    except ValueError:
        company = await repo.get_by_ticker(id_or_ticker.upper())

    if company is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company {id_or_ticker!r} not found.",
        )
    return CompanyDetail.model_validate(company)
