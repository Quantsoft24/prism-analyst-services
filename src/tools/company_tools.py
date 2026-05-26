"""Agent-callable tools for company metadata — backed by the catalog DB
(``company_industry`` on the stock_chat Postgres, 4,773 companies).

Same API surface the LLM has always seen (``lookup_company``,
``search_companies``, ``list_covered_sectors``); the data source moved from
PRISM's tiny curated table to the full Indian-markets catalog.

Both ``lookup_company`` and ``search_companies`` are now **typo-tolerant**:
they include a ``suggestions`` list of close-but-not-exact matches so the
agent can surface "did you mean ...?" instead of confabulating.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.catalog_database import catalog_session_scope
from src.models.catalog import CompanyIndustry
from src.repositories.company_repo import CompanyRepository

if TYPE_CHECKING:
    from google.adk.tools import FunctionTool


# ── Shared formatters ──────────────────────────────────────────────────────


def _to_full_row(c: CompanyIndustry) -> dict:
    """Full record shape used by ``lookup_company``."""
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


def _to_list_row(c: CompanyIndustry) -> dict:
    """Compact record shape used by ``search_companies.items[]``."""
    return {
        "ticker": c.code,
        "name": c.company_name,
        "sector": c.industry,
        "industry": c.industry,
        "exchange": "NSE",
    }


def _to_suggestion(c: CompanyIndustry) -> dict:
    """Minimal record shape used by the ``suggestions[]`` array — keeps the
    payload small so the LLM doesn't burn tokens on near-misses."""
    return {"ticker": c.code, "name": c.company_name}


# ── Tool implementations ───────────────────────────────────────────────────


async def lookup_company(ticker: str) -> dict:
    """Look up a company by its NSE/BSE ticker / scrip code.

    Use this when the user mentions a specific ticker (e.g. "TCS", "RELIANCE",
    "MOIL") and you need verified metadata: company name, industry, ISIN.

    On a miss, returns ``{"found": false, "ticker": <input>,
    "suggestions": [...]}`` — the suggestions list contains up to 3 of the
    closest tickers/names by fuzzy match. If the user typed a typo
    ("RELIACE"), the closest real ticker/name will be there; surface it as
    "did you mean ...?" instead of inventing details.

    The catalog covers 4,773 Indian NSE/BSE-listed companies.

    Args:
        ticker: Ticker symbol (uppercased internally), e.g. "TCS" or "RELIANCE".

    Returns:
        On hit: dict with ``found=true``, ``ticker``, ``name``, ``exchange``
        ("NSE"), ``sector``, ``industry``, ``country`` ("IN"), ``isin``.
        Fields not tracked by the catalog (legal_name, cin, website,
        description) are returned as ``null``.

        On miss: ``{"found": false, "ticker": <input>, "suggestions": [
        {"ticker": "...", "name": "..."}, ...]}``.
    """
    cleaned = ticker.strip().upper()
    async with catalog_session_scope() as session:
        repo = CompanyRepository(session)
        c = await repo.get_by_ticker(cleaned)
        if c is not None:
            return _to_full_row(c)

        # Miss → fuzzy-search the catalog for close matches so the LLM can
        # offer "did you mean ...?" instead of inventing data.
        result = await repo.list(search=cleaned, limit=3)
        # Combine top items and suggestions for the surface — both are
        # "close" matches even if some scored above the hit threshold.
        near = (result.items or []) + (result.suggestions or [])
        return {
            "found": False,
            "ticker": cleaned,
            "suggestions": [_to_suggestion(c) for c in near[:3]],
        }


async def search_companies(query: str, sector: str | None = None, limit: int = 10) -> dict:
    """Search the Indian-markets catalog by name, ticker, or sector.

    Typo-tolerant: queries like "Reliac" or "Tata Consultanc" return the
    correct companies. When the query is genuinely ambiguous or
    misspelled, the response includes a ``suggestions`` list of the
    nearest sub-threshold matches — surface them to the user as
    "did you mean ...?" instead of confabulating.

    Use this when the user asks about a company by name and you're not sure
    of the ticker ("Tata Consultancy"), or wants to discover companies in a
    sector ("show me banks"). Prefer ``lookup_company`` when you already
    have the exact ticker.

    Args:
        query: Free-text search — matches ticker/scrip code or company name.
            Pass an empty string to list all companies in a sector.
        sector: Exact-match industry filter (e.g. "Software & Services").
            Use ``list_covered_sectors`` to discover valid values.
        limit: Max results (default 10, max 25).

    Returns:
        Dict with:
          * ``total`` (int) — count of strong matches across the catalog.
          * ``items`` (list) — up to ``limit`` strong matches, each
            ``{ticker, name, sector, industry, exchange}``.
          * ``suggestions`` (list) — up to 3 sub-threshold near-matches
            (typo / partial name) — only populated when ``items`` is
            empty or has ≤1 result. Use these for "did you mean ...?".
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
            "items": [_to_list_row(c) for c in result.items],
            "suggestions": [_to_suggestion(c) for c in result.suggestions],
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
