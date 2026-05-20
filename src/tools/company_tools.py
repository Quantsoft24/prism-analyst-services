"""Agent-callable tools for company metadata.

These tools wrap the same ``CompanyRepository`` the ``/api/v1/companies``
HTTP router uses — so an agent calling ``lookup_company("TCS")`` sees exactly
the same data a third-party API consumer would. One source of truth.

Each tool opens its own short-lived async DB session via ``session_scope()``
because tools run inside ADK's runner loop, not inside a FastAPI request,
and we can't assume there's an ambient transaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.database import session_scope
from src.repositories.company_repo import CompanyRepository

if TYPE_CHECKING:
    from google.adk.tools import FunctionTool


# ── Tool implementations ───────────────────────────────────────────────────


async def lookup_company(ticker: str) -> dict:
    """Look up a company by its NSE/BSE ticker symbol.

    Use this when the user mentions a specific ticker (e.g. "TCS", "RELIANCE",
    "MOIL") and you need verified metadata about that company — sector,
    industry, ISIN, CIN, listed exchange, official description.

    Returns ``{"found": false}`` if the ticker is not in PRISM's coverage
    universe — DO NOT guess or invent details in that case; tell the user
    the company is not covered yet.

    Args:
        ticker: Uppercase ticker symbol like "TCS" or "RELIANCE".

    Returns:
        Dict with keys: ``found`` (bool), and on success ``ticker``, ``name``,
        ``legal_name``, ``exchange``, ``sector``, ``industry``, ``country``,
        ``isin``, ``cin``, ``website``, ``description``.
    """
    async with session_scope() as session:
        repo = CompanyRepository(session)
        company = await repo.get_by_ticker(ticker.strip().upper())
        if company is None:
            return {"found": False, "ticker": ticker}
        return {
            "found": True,
            "ticker": company.ticker,
            "name": company.name,
            "legal_name": company.legal_name,
            "exchange": company.exchange,
            "sector": company.sector,
            "industry": company.industry,
            "country": company.country,
            "isin": company.isin,
            "cin": company.cin,
            "website": company.website,
            "description": company.description,
        }


async def search_companies(query: str, sector: str | None = None, limit: int = 10) -> dict:
    """Search PRISM's coverage universe by name, ticker, or alias.

    Use this when the user asks about a company by name but you're not sure
    of the ticker (e.g. "Tata Consultancy" or "Reliance"), or wants to
    discover companies in a sector ("show me banks").

    Args:
        query: Free-text search — matches ticker, name, or registered aliases.
            Use an empty string ``""`` to list all companies in a sector.
        sector: Optional exact-match sector filter, e.g. "Financials",
            "Information Technology", "Energy", "Materials".
        limit: Max results to return (default 10, max 25).

    Returns:
        Dict with ``total`` (int — total matches), ``items`` (list of company
        summaries: ticker, name, sector, industry, exchange).
    """
    limit = max(1, min(limit, 25))
    async with session_scope() as session:
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
                    "ticker": c.ticker,
                    "name": c.name,
                    "sector": c.sector,
                    "industry": c.industry,
                    "exchange": c.exchange,
                }
                for c in result.items
            ],
        }


async def list_covered_sectors() -> dict:
    """Return the distinct sectors PRISM currently covers.

    Use this when the user asks "what sectors do you cover?" or as a
    discovery aid before filtering ``search_companies``.

    Returns:
        Dict with ``sectors`` (list of distinct sector strings, alphabetically
        sorted).
    """
    async with session_scope() as session:
        repo = CompanyRepository(session)
        # Fetch a representative slice — sectors are low-cardinality.
        result = await repo.list(limit=200, offset=0)
        sectors = sorted({c.sector for c in result.items if c.sector})
        return {"sectors": sectors}


# ── ADK FunctionTool wrappers ──────────────────────────────────────────────
# Built lazily so this module is importable without google-adk installed.


def _build_tools() -> list["FunctionTool"]:
    from google.adk.tools import FunctionTool

    return [
        FunctionTool(func=lookup_company),
        FunctionTool(func=search_companies),
        FunctionTool(func=list_covered_sectors),
    ]


class _LazyToolList:
    """Defers ``google.adk.tools`` import until first access — keeps module
    import cheap and import-graph friendly when ADK isn't installed yet."""

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
