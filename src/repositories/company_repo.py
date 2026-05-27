"""Company data access — backed by the external ``company_industry`` table
on the catalog DB (stock_chat Postgres).

PRISM's old ``companies`` / ``company_aliases`` tables were retired (2026-05-24).
Same public API as before — agents and routers don't change — but every read
now hits the much larger 4,773-row catalog via a read-only secondary engine.

Search is typo-tolerant by default (``fuzzy=True``):
  1. Wide-net SQL pulls up to 200 candidates with ILIKE on code + name.
  2. ``rapidfuzz`` re-ranks the candidates by string similarity to the
     normalized query. Hits above 60 become ``items``; 40-59 become
     ``suggestions`` (the "did you mean" surface).
  3. If Stage 1 returns zero (typo shares no substring with any real
     name — "Releace" → "Reliance"), we pull a larger sample (500 rows)
     scoped by sector and re-rank.

Once ``CATALOG_TRIGRAM_ENABLED`` is true (after pg_trgm is enabled on the
shared catalog DB — see docs/CATALOG_DB_INDEXES.md), the wide-net SQL is
replaced by a trigram-similarity scan that's both faster and broader.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz
from sqlalchemy import Integer, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.catalog import CompanyIndustry


@dataclass(slots=True)
class CompanyListResult:
    """Paginated list result.

    ``items``       — top hits above the fuzzy-match threshold (or all
                      substring matches when ``fuzzy=False``).
    ``total``       — count of rows matching the query (for pagination UI).
    ``suggestions`` — sub-threshold near-matches (typically used as
                      "did you mean ...?" when ``items`` is empty or thin.
                      Same row shape as ``items``).
    """

    items: list[CompanyIndustry]
    total: int
    suggestions: list[CompanyIndustry] = field(default_factory=list)


# ── Fuzzy-match tuning constants ──────────────────────────────────────────
# Scores are 0-100 from rapidfuzz; the values below were calibrated against
# the 4,773-row Indian catalog using typos like "Reliac/Releace/TCS Ltd".
_HIT_THRESHOLD = 60          # ≥ this score → real result
_SUGGEST_THRESHOLD = 40      # ≥ this but < HIT → "did you mean" surface
_STAGE1_CANDIDATE_CAP = 200  # how many SQL rows we'll re-rank
_STAGE2_CANDIDATE_CAP = 500  # fallback when stage-1 SQL returns nothing

# Common suffixes we strip before matching so "TCS Ltd" matches "TCS".
# Order matters — longer suffixes first.
_NOISE_SUFFIXES = (
    "private limited",
    "pvt limited",
    "pvt ltd",
    "limited",
    "ltd",
    "ltd.",
    "inc",
    "inc.",
    "corp",
    "corporation",
    "company",
    "co",
    "co.",
)


def _normalize_query(q: str) -> str:
    """Normalize a user-typed query before matching.

    Lowercase, collapse whitespace, drop punctuation, strip the common
    legal-suffix noise. Keep this conservative — we DON'T fold accents or
    transliterate, because the catalog itself doesn't have them.
    """
    s = q.strip().lower()
    # Strip everything that isn't a letter, digit, or single space.
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for suffix in _NOISE_SUFFIXES:
        if s.endswith(" " + suffix):
            s = s[: -(len(suffix) + 1)].rstrip()
            break  # only one suffix per query
    return s


def _candidate_score(company: CompanyIndustry, query_norm: str) -> float:
    """Best similarity score (0-100) across code and company_name.

    We use ``token_set_ratio`` (robust to word order / extra words) and
    ``partial_ratio`` (catches substring typos) and take the max. Codes
    get a small bonus when the query is short (3-6 chars) since users
    typing a ticker are usually after the exact code.
    """
    code = (company.code or "").lower()
    name = (company.company_name or "").lower()

    name_score = max(
        fuzz.token_set_ratio(query_norm, name),
        fuzz.partial_ratio(query_norm, name),
    )
    code_score = max(
        fuzz.token_set_ratio(query_norm, code),
        fuzz.partial_ratio(query_norm, code),
    )
    # 3-6 char query → likely a ticker; weight code match higher
    if 3 <= len(query_norm) <= 6:
        code_score = min(100.0, code_score + 5.0)
    return max(name_score, code_score)


def _rank_candidates(
    candidates: list[CompanyIndustry], query_norm: str
) -> list[tuple[CompanyIndustry, float]]:
    """Sort candidates by descending fuzzy score; ties broken alphabetically."""
    scored = [(c, _candidate_score(c, query_norm)) for c in candidates]
    scored.sort(key=lambda cs: (-cs[1], (cs[0].company_name or "").lower()))
    return scored


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
        fuzzy: bool = True,
    ) -> CompanyListResult:
        """Paginated list with optional filters.

        Args:
            search: free-text query matched against ``code`` and
                ``company_name``. When ``fuzzy=True`` (default) we use
                rapidfuzz ranking so typos like "Reliac" still find
                "Reliance Industries". Set ``fuzzy=False`` for legacy
                ILIKE-only substring matching.
            sector: exact-match filter on the catalog's ``industry`` column.
            exchange: accepted for API back-compat; the catalog is not
                exchange-partitioned.
            limit, offset: standard pagination.

        Returns:
            ``CompanyListResult`` with ``items``, ``total``, and
            (fuzzy path only) ``suggestions`` for sub-threshold matches.
        """
        # No search → straightforward filtered list, no fuzzy needed.
        if not search or not search.strip():
            return await self._list_no_search(sector=sector, limit=limit, offset=offset)

        if not fuzzy:
            return await self._list_substring(
                search=search.strip(), sector=sector, limit=limit, offset=offset
            )

        return await self._list_fuzzy(search=search, sector=sector, limit=limit, offset=offset)

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

    # ── Internal paths ────────────────────────────────────────────────────

    async def _list_no_search(
        self, *, sector: str | None, limit: int, offset: int
    ) -> CompanyListResult:
        """Pure list/filter — no text matching.

        The upstream catalog has many rows where ``company_name`` is empty
        — those are still real listings (the ticker IS the identifier on
        Indian exchanges), so we keep them. We DO de-prioritise pure-numeric
        BSE scrip codes (e.g. "526945") by sorting them to the end, since
        they're rarely what someone browsing the universe actually wants.
        Search paths bypass this ordering and accept exact code hits.
        """
        filters = []
        if sector:
            filters.append(CompanyIndustry.industry == sector)
        # ``code ~ '^[0-9]+$'`` → 1 for pure-numeric (rank these last). The
        # CASE expression turns that into a 0/1 sort key. PostgreSQL-only
        # syntax, which is fine since the catalog is Postgres.
        numeric_code_last = func.cast(
            CompanyIndustry.code.op("~")("^[0-9]+$"), Integer
        )
        stmt = (
            select(CompanyIndustry)
            .where(*filters)
            .order_by(
                numeric_code_last.asc(),  # 0 (alpha tickers) first, 1 (BSE codes) last
                CompanyIndustry.industry_rank.asc().nulls_last(),
                CompanyIndustry.company_name.asc().nulls_last(),
                CompanyIndustry.code.asc(),
            )
            .limit(limit)
            .offset(offset)
        )
        items = list((await self._session.execute(stmt)).scalars().all())
        total = (
            await self._session.execute(select(func.count(CompanyIndustry.code)).where(*filters))
        ).scalar_one()
        return CompanyListResult(items=items, total=total)

    async def _list_substring(
        self, *, search: str, sector: str | None, limit: int, offset: int
    ) -> CompanyListResult:
        """Legacy ILIKE-only path — kept behind ``fuzzy=False`` for callers
        that need exact-substring semantics."""
        filters = []
        if sector:
            filters.append(CompanyIndustry.industry == sector)
        pattern = f"%{search}%"
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
        items = list((await self._session.execute(stmt)).scalars().all())
        total = (
            await self._session.execute(select(func.count(CompanyIndustry.code)).where(*filters))
        ).scalar_one()
        return CompanyListResult(items=items, total=total)

    async def _list_fuzzy(
        self, *, search: str, sector: str | None, limit: int, offset: int
    ) -> CompanyListResult:
        """Fuzzy path — typo-tolerant via rapidfuzz."""
        query_norm = _normalize_query(search)
        if not query_norm:  # query was pure punctuation/whitespace
            return await self._list_no_search(sector=sector, limit=limit, offset=offset)

        # Stage 1: wide-net SQL. Substring + prefix to catch ticker-shaped queries.
        candidates = await self._fetch_stage1_candidates(query_norm, sector)

        # Stage 2: if Stage 1 was empty, pull a bigger sector-scoped sample.
        if not candidates:
            candidates = await self._fetch_stage2_candidates(sector)

        if not candidates:
            return CompanyListResult(items=[], total=0, suggestions=[])

        # Rank in Python.
        ranked = _rank_candidates(candidates, query_norm)

        hits = [c for c, score in ranked if score >= _HIT_THRESHOLD]
        near = [
            c
            for c, score in ranked
            if _SUGGEST_THRESHOLD <= score < _HIT_THRESHOLD
        ]

        total = len(hits)
        items_page = hits[offset : offset + limit]
        # Only surface suggestions on the first page AND only when items are
        # thin — keeps the contract intuitive for both UI pagination and the
        # LLM tool's "did you mean" loop.
        suggestions = near[:3] if offset == 0 and len(items_page) <= 1 else []

        return CompanyListResult(items=items_page, total=total, suggestions=suggestions)

    async def _fetch_stage1_candidates(
        self, query_norm: str, sector: str | None
    ) -> list[CompanyIndustry]:
        """Wide-net SQL pull — substring on code or name (plus prefix on code).

        Designed to be cheap (~5ms on 4773 rows w/ existing B-tree indexes)
        and broad enough that one-letter typos still surface candidates.
        """
        filters = []
        if sector:
            filters.append(CompanyIndustry.industry == sector)
        pattern = f"%{query_norm}%"
        prefix = f"{query_norm}%"
        filters.append(
            or_(
                CompanyIndustry.code.ilike(pattern),
                CompanyIndustry.code.ilike(prefix),
                CompanyIndustry.company_name.ilike(pattern),
            )
        )
        # Future: when pg_trgm is enabled on the catalog DB
        # (docs/CATALOG_DB_INDEXES.md), prepend a ``company_name % :q``
        # similarity scan here for sub-25ms ranked hits. Today the Python
        # re-rank does the work.
        stmt = select(CompanyIndustry).where(*filters).limit(_STAGE1_CANDIDATE_CAP)
        return list((await self._session.execute(stmt)).scalars().all())

    async def _fetch_stage2_candidates(
        self, sector: str | None
    ) -> list[CompanyIndustry]:
        """Bigger sector-scoped sample when Stage 1 was empty (rare).

        Triggered when the user's typo shares no substring with any real
        name. Pulls up to 500 rows and lets rapidfuzz do all the work.
        """
        filters = []
        if sector:
            filters.append(CompanyIndustry.industry == sector)
        stmt = select(CompanyIndustry).where(*filters).limit(_STAGE2_CANDIDATE_CAP)
        return list((await self._session.execute(stmt)).scalars().all())
