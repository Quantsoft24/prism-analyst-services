"""Company data access — list, search, lookup by ticker / ISIN / alias.

Public/global metadata. Not firm-scoped (a company exists once for all firms).
Per-firm projections (watchlist, BMC analyses, etc.) reference companies by
``company_id`` and add their own ``firm_id``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.company import Company, CompanyAlias


@dataclass(slots=True)
class CompanyListResult:
    """Paginated list result — items + total count for the filtered query."""

    items: list[Company]
    total: int


class CompanyRepository:
    """Async repository for the ``companies`` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Reads ─────────────────────────────────────────────────────────────

    async def get_by_id(self, company_id: uuid.UUID) -> Company | None:
        stmt = (
            select(Company)
            .where(Company.id == company_id)
            .options(selectinload(Company.aliases))
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_by_ticker(self, ticker: str, exchange: str = "NSE") -> Company | None:
        stmt = (
            select(Company)
            .where(Company.exchange == exchange, Company.ticker == ticker.upper())
            .options(selectinload(Company.aliases))
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_by_isin(self, isin: str) -> Company | None:
        stmt = (
            select(Company)
            .where(Company.isin == isin.upper())
            .options(selectinload(Company.aliases))
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list(
        self,
        *,
        search: str | None = None,
        sector: str | None = None,
        exchange: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> CompanyListResult:
        """Paginated list with optional filters.

        ``search`` is matched against ticker, name, and aliases (ILIKE).
        The same WHERE clause is used for the count query so ``total``
        always matches what the caller would get with offset=0/limit=∞.
        """
        filters = []
        if exchange:
            filters.append(Company.exchange == exchange.upper())
        if sector:
            filters.append(Company.sector == sector)
        if search:
            pattern = f"%{search.strip()}%"
            alias_subq = (
                select(CompanyAlias.company_id)
                .where(CompanyAlias.value.ilike(pattern))
                .scalar_subquery()
            )
            filters.append(
                or_(
                    Company.ticker.ilike(pattern),
                    Company.name.ilike(pattern),
                    Company.id.in_(alias_subq),
                )
            )

        # Items
        stmt = (
            select(Company)
            .where(*filters)
            .order_by(Company.name.asc())
            .limit(limit)
            .offset(offset)
            .options(selectinload(Company.aliases))
        )
        items = (await self._session.execute(stmt)).scalars().all()

        # Count — reuse identical WHERE so pagination stays consistent
        count_stmt = select(func.count(Company.id)).where(*filters)
        total = (await self._session.execute(count_stmt)).scalar_one()

        return CompanyListResult(items=list(items), total=total)

    # ── Writes — minimal in S1; real ingestion lands in S4 ────────────────

    async def add(self, company: Company) -> Company:
        self._session.add(company)
        await self._session.flush()
        return company
