"""Companies — list + lookup against the read-only catalog (``company_industry``).

Same URL contract as before (``GET /api/v1/companies``,
``GET /api/v1/companies/{id_or_ticker}``) so the frontend's CompaniesView
keeps working — but the dataset is now the 4,773-row Indian-markets catalog
on the stock_chat Postgres rather than PRISM's tiny retired ``companies``
table. Fields not tracked by the catalog (legal_name, cin, aliases, website,
description) come back as ``null`` / empty.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.auth import get_current_firm_id
from src.core.catalog_database import get_catalog_session
from src.models.catalog import CompanyIndustry
from src.repositories.company_repo import CompanyRepository
from src.schemas.common import PageMeta, Paginated
from src.schemas.company import CompanyDetail, CompanyRead, synthetic_company_id

router = APIRouter(prefix="/companies", tags=["Companies"])


def _to_read(c: CompanyIndustry) -> CompanyRead:
    return CompanyRead(
        id=synthetic_company_id(isin=c.isin, code=c.code),
        ticker=c.code,
        name=c.company_name or c.code,
        exchange="NSE",
        sector=c.industry,
        industry=c.industry,
        country="IN",
        isin=c.isin,
        is_active=True,
    )


def _to_detail(c: CompanyIndustry) -> CompanyDetail:
    base = _to_read(c).model_dump()
    return CompanyDetail(
        **base,
        legal_name=None,
        cin=None,
        pan=None,
        website=None,
        description=None,
        aliases=[],
        created_at=c.fetched_at,
        updated_at=c.fetched_at,
    )


@router.get(
    "",
    response_model=Paginated[CompanyRead],
    summary="List companies (Indian NSE/BSE catalog, 4,773 entries)",
    description=(
        "Paginated list backed by ``company_industry`` on the catalog DB. "
        "Filter by ``search`` (matches ticker / scrip code / name), "
        "``sector`` (exact industry match), or ``exchange`` (accepted for "
        "back-compat — the catalog itself is not exchange-partitioned)."
    ),
)
async def list_companies(
    session: Annotated[AsyncSession, Depends(get_catalog_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    search: Annotated[str | None, Query(description="Free-text — ticker or name.")] = None,
    sector: Annotated[str | None, Query(description="Exact industry filter.")] = None,
    exchange: Annotated[str | None, Query(description="NSE | BSE (accepted, not enforced).")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Paginated[CompanyRead]:
    _ = firm_id  # auth-gated only; catalog is global
    repo = CompanyRepository(session)
    result = await repo.list(
        search=search, sector=sector, exchange=exchange, limit=limit, offset=offset
    )
    return Paginated[CompanyRead](
        items=[_to_read(c) for c in result.items],
        page=PageMeta(total=result.total, limit=limit, offset=offset),
    )


@router.get(
    "/{id_or_ticker}",
    response_model=CompanyDetail,
    summary="Get one company by id (uuid5) or NSE ticker / scrip code",
    responses={404: {"description": "Company not found in the catalog."}},
)
async def get_company(
    id_or_ticker: str,
    session: Annotated[AsyncSession, Depends(get_catalog_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> CompanyDetail:
    _ = firm_id
    repo = CompanyRepository(session)

    # Resolution: try ticker/code first (the common case from the frontend).
    # If the caller hands us a UUID, we have no inverse map (id is a uuid5 of
    # the code/isin) — fall back to ISIN if it looks like one.
    company: CompanyIndustry | None = None
    try:
        # Detect UUID format — no direct lookup, return 404 with hint.
        uuid.UUID(id_or_ticker)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Catalog companies are addressed by ticker / scrip code (e.g. "
                "'TCS') or ISIN, not by uuid. Use the ticker or call the list "
                "endpoint to discover the right value."
            ),
        )
    except ValueError:
        pass  # not a UUID — try ticker, then ISIN

    company = await repo.get_by_ticker(id_or_ticker.upper())
    if company is None and id_or_ticker.upper().startswith("INE"):
        company = await repo.get_by_isin(id_or_ticker.upper())

    if company is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company {id_or_ticker!r} not found in catalog.",
        )
    return _to_detail(company)
