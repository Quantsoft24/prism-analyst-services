"""Company data access — backed by the external ``company_industry`` table
on the catalog DB (stock_chat Postgres).

PRISM's old ``companies`` / ``company_aliases`` tables were retired (2026-05-24).
Same public API as before — agents and routers don't change — but every read
now hits the much larger 4,773-row catalog via a read-only secondary engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.catalog import CompanyIndustry


@dataclass(slots=True)
class CompanyListResult:
    """Paginated list result — items + total count for the filtered query."""

    items: list[CompanyIndustry]
    total: int


class CompanyRepository:
    """Async read-only repository over ``company_industry``. Construct with a
    session bound to the catalog engine (``catalog_session_scope`` /
    ``get_catalog_session``)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Reads ─────────────────────────────────────────────────────────────

    async def get_by_ticker(self, ticker: str, exchange: str = "NSE") -> CompanyIndustry | None:
        """Resolve by NSE symbol / scrip code (``code`` column). ``exchange``
        is accepted for API back-compat but the catalog doesn't track it —
        a code is treated as the same company across exchanges."""
        _ = exchange  # back-compat; catalog has no exchange dimension
        stmt = select(CompanyIndustry).where(CompanyIndustry.code == ticker.upper())
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_by_isin(self, isin: str) -> CompanyIndustry | None:
        stmt = select(CompanyIndustry).where(CompanyIndustry.isin == isin.upper())
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list(
        self,
        *,
        search: str | None = None,
        sector: str | None = None,
        exchange: str | None = None,  # noqa: ARG002 — back-compat, ignored
        limit: int = 25,
        offset: int = 0,
    ) -> CompanyListResult:
        """Paginated list with optional filters.

        ``search`` matches against ``code`` and ``company_name`` (ILIKE).
        ``sector`` matches the catalog's ``industry`` column (these are
        semantically the same in PRISM's vocabulary). ``exchange`` is
        ignored — the catalog is not exchange-partitioned.
        """
        filters = []
        if sector:
            filters.append(CompanyIndustry.industry == sector)
        if search:
            pattern = f"%{search.strip()}%"
            filters.append(
                or_(
                    CompanyIndustry.code.ilike(pattern),
                    CompanyIndustry.company_name.ilike(pattern),
                )
            )

        stmt = (
            select(CompanyIndustry)
            .where(*filters)
            .order_by(CompanyIndustry.company_name.asc())
            .limit(limit)
            .offset(offset)
        )
        items = (await self._session.execute(stmt)).scalars().all()

        count_stmt = select(func.count(CompanyIndustry.code)).where(*filters)
        total = (await self._session.execute(count_stmt)).scalar_one()

        return CompanyListResult(items=list(items), total=total)

    async def distinct_sectors(self, limit: int = 200) -> list[str]:
        """Distinct ``industry`` values — used by ``list_covered_sectors``."""
        stmt = (
            select(CompanyIndustry.industry)
            .where(CompanyIndustry.industry.isnot(None))
            .distinct()
            .order_by(CompanyIndustry.industry.asc())
            .limit(limit)
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return [r for r in rows if r]
