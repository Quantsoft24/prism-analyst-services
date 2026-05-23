"""Agent-callable tools for company metadata — backed by the catalog DB
(``company_industry`` on the stock_chat Postgres, 4,773 companies).

Same API surface the LLM has always seen (``lookup_company``,
``search_companies``, ``list_covered_sectors``); the data source moved from
PRISM's tiny curated table to the full Indian-markets catalog.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.catalog_database import catalog_session_scope
from src.repositories.company_repo import CompanyRepository

if TYPE_CHECKING:
    from google.adk.tools import FunctionTool


# ── Tool implementations ───────────────────────────────────────────────────


async def lookup_company(ticker: str) -> dict:
    """Look up a company by its NSE/BSE ticker / scrip code.

    Use this when the user mentions a specific ticker (e.g. "TCS", "RELIANCE",
    "MOIL") and you need verified metadata: company name, industry, ISIN.

    Returns ``{"found": false}`` if the ticker isn't in the catalog — DO NOT
    invent details. The catalog covers 4,773 Indian NSE/BSE-listed companies.

    Args:
        ticker: Ticker symbol (uppercased internally), e.g. "TCS" or "RELIANCE".

    Returns:
        Dict with ``found`` and on success ``ticker``, ``name``, ``exchange``
        (always "NSE" in this catalog), ``sector``, ``industry``, ``country``
        ("IN"), ``isin``. Fields not tracked by the catalog (legal_name, cin,
        website, description) are returned as ``null``.
    """
    async with catalog_session_scope() as session:
        repo = CompanyRepository(session)
        c = await repo.get_by_ticker(ticker.strip().upper())
        if c is None:
            return {"found": False, "ticker": ticker}
        return {
            "found": True,
            "ticker": c.code,
            "name": c.company_name,
            "legal_name": None,
            "exchange": "NSE",
            "sector": c.industry,
            "industry": c.industry,
            "country": "IN",
            "isin": c.isin,
            "cin": None,
            "website": None,
            "description": None,
        }


async def search_companies(query: str, sector: str | None = None, limit: int = 10) -> dict:
    """Search the Indian-markets catalog by name, ticker, or sector.

    Use this when the user asks about a company by name and you're not sure
    of the ticker ("Tata Consultancy"), or wants to discover companies in a
    sector ("show me banks").

    Args:
        query: Free-text search — matches ticker/scrip code or company name.
            Pass an empty string to list all companies in a sector.
        sector: Exact-match industry filter (e.g. "Software & Services").
            Use ``list_covered_sectors`` to discover valid values.
        limit: Max results (default 10, max 25).

    Returns:
        Dict with ``total`` (int) and ``items`` (list of {ticker, name,
        sector, industry, exchange}). 4,773 companies are searchable.
    """
    limit = max(1, min(limit, 25))
    async with catalog_session_scope() as session:
        repo = CompanyRepository(session)
        result = await repo.list(
            search=query.strip() or None,
            sector=sector,
            limit=limit,
            offset=0,
        )
        return {
            "total": result.total,
            "items": [
                {
                    "ticker": c.code,
                    "name": c.company_name,
                    "sector": c.industry,
                    "industry": c.industry,
                    "exchange": "NSE",
                }
                for c in result.items
            ],
        }


async def list_covered_sectors() -> dict:
    """Distinct industries / sectors available in the catalog.

    Use this when the user asks "what sectors do you cover?" or before
    filtering ``search_companies`` by sector.

    Returns:
        Dict with ``sectors`` (list of distinct industry strings, alphabetical).
    """
    async with catalog_session_scope() as session:
        repo = CompanyRepository(session)
        sectors = await repo.distinct_sectors()
    return {"sectors": sorted(sectors)}


# ── ADK FunctionTool wrappers (lazy — same pattern as the rest) ───────────


def _build_tools() -> list["FunctionTool"]:
    from google.adk.tools import FunctionTool

    return [
        FunctionTool(func=lookup_company),
        FunctionTool(func=search_companies),
        FunctionTool(func=list_covered_sectors),
    ]


class _LazyToolList:
    def __init__(self) -> None:
        self._tools: list[FunctionTool] | None = None

    def __iter__(self):
        if self._tools is None:
            self._tools = _build_tools()
        return iter(self._tools)

    def __len__(self) -> int:
        if self._tools is None:
            self._tools = _build_tools()
        return len(self._tools)

    def to_list(self) -> list:
        if self._tools is None:
            self._tools = _build_tools()
        return list(self._tools)


COMPANY_TOOLS = _LazyToolList()
