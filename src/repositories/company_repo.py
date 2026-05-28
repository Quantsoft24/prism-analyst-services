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

import logging
import re
import time
from dataclasses import dataclass, field

from rapidfuzz import fuzz
from sqlalchemy import Integer, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.catalog import CompanyAlias, CompanyIndustry

logger = logging.getLogger(__name__)


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


@dataclass(slots=True)
class AliasMatch:
    """Result of a successful alias resolution.

    ``company``  — the resolved CompanyIndustry row.
    ``alias``    — the alias string that matched.
    ``code``     — the canonical ticker code.
    ``method``   — how it was resolved: 'exact' | 'trgm' | 'cache'.
    """

    company: CompanyIndustry
    alias: str
    code: str
    method: str


# ── Alias TTL cache ───────────────────────────────────────────────────────────
# The alias table is quasi-static (changes only when the generation script
# runs). Caching exact alias_norm → code lookups avoids a DB roundtrip on
# every single user query. 30-minute TTL keeps memory bounded and ensures
# new aliases propagate within half an hour.
#
# Only exact matches are cached. pg_trgm similarity queries are too
# dynamic (query-dependent) and too rare (only fire when exact misses)
# to benefit from caching.

_ALIAS_CACHE_TTL = 1800  # 30 minutes
_ALIAS_CACHE_MAX = 1000  # max entries


class _AliasCache:
    """Lightweight TTL cache for exact alias → code lookups.

    Thread-safety: asyncio is single-threaded within an event loop, so
    no locking needed. For multi-worker deployments each worker gets its
    own cache instance — that's fine given the small memory footprint
    (~100 KB for 1,000 entries).
    """

    def __init__(self, ttl: float = _ALIAS_CACHE_TTL, maxsize: int = _ALIAS_CACHE_MAX) -> None:
        self._ttl = ttl
        self._maxsize = maxsize
        self._store: dict[str, tuple[str | None, float]] = {}  # alias_norm -> (code | None, expiry)

    def get(self, alias_norm: str) -> str | None | object:
        """Return cached code, ``None`` (cached negative), or ``_MISS`` (not in cache)."""
        entry = self._store.get(alias_norm)
        if entry is None:
            return _MISS
        code, expiry = entry
        if time.monotonic() > expiry:
            del self._store[alias_norm]
            return _MISS
        return code

    def put(self, alias_norm: str, code: str | None) -> None:
        """Cache a result (code or None for negative cache)."""
        # Evict oldest entries if at capacity. Simple: clear half the cache.
        # This is rare (1,000 entries) and fast.
        if len(self._store) >= self._maxsize:
            entries = sorted(self._store.items(), key=lambda kv: kv[1][1])
            for key, _ in entries[: len(entries) // 2]:
                del self._store[key]
        self._store[alias_norm] = (code, time.monotonic() + self._ttl)

    def clear(self) -> None:
        """Flush the cache (called after alias regeneration)."""
        self._store.clear()


# Sentinel for cache miss (distinct from None, which means "negative cache").
_MISS = object()

# Module-level singleton — shared across all CompanyRepository instances
# within a single worker process.
_alias_cache = _AliasCache()


# ── Fuzzy-match tuning constants ──────────────────────────────────────────
# Scores are 0-100 from rapidfuzz; the values below were calibrated against
# the 4,773-row Indian catalog using typos like "Reliac/Releace/TCS Ltd".
_HIT_THRESHOLD = 60          # ≥ this score → real result
_SUGGEST_THRESHOLD = 50      # ≥ this but < HIT → "did you mean" surface
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
    # Normalise "&" to "and" BEFORE stripping punctuation — gives "L&T"
    # a meaningful shape ("l and t") instead of "l t" for rapidfuzz.
    s = s.replace("&", " and ")
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
    """Async read-only repository over ``company_industry`` + ``company_aliases``.
    Construct with a session bound to the catalog engine (``catalog_session_scope``
    / ``get_catalog_session``)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Alias resolution ─────────────────────────────────────────────────────────

    async def resolve_alias(self, query: str) -> AliasMatch | None:
        """Resolve a user query against the ``company_aliases`` table.

        Resolution order (first hit wins):
          1. **TTL cache** — exact alias_norm lookup in the in-memory cache.
             Avoids a DB roundtrip for the ~500 most common abbreviations.
          2. **Exact B-tree match** on ``alias_norm`` — O(1) indexed lookup.
             Covers "RIL", "Infy", "L&T" etc.
          3. **pg_trgm similarity** on ``alias_norm`` (threshold 0.6) —
             catches typos in the alias itself ("Relianse" → "reliance").

        Returns ``AliasMatch`` on success, ``None`` on no match.
        Thread-safe within a single asyncio event loop.
        """
        norm = _normalize_query(query)
        if not norm:
            return None

        # 1) Check the in-memory cache first.
        cached = _alias_cache.get(norm)
        if cached is not _MISS:
            if cached is None:
                return None  # negative cache
            # Cache hit — we have the code, fetch the company row.
            company = await self.get_by_ticker(cached)
            if company is not None:
                return AliasMatch(
                    company=company, alias=norm, code=cached, method="cache"
                )

        # 2) Exact match on alias_norm (B-tree index).
        try:
            stmt = (
                select(CompanyAlias)
                .where(CompanyAlias.alias_norm == norm)
                .order_by(CompanyAlias.confidence.desc())
                .limit(1)
            )
            alias_row = (await self._session.execute(stmt)).scalar_one_or_none()
        except Exception:
            # Table might not exist yet (pre-migration). Degrade gracefully.
            logger.debug("company_aliases table not available — skipping alias lookup")
            return None

        if alias_row is not None:
            _alias_cache.put(norm, alias_row.code)
            company = await self.get_by_ticker(alias_row.code)
            if company is not None:
                return AliasMatch(
                    company=company,
                    alias=alias_row.alias,
                    code=alias_row.code,
                    method="exact",
                )

        # 3) pg_trgm similarity (catches typos in the alias itself).
        try:
            stmt = (
                select(CompanyAlias)
                .where(
                    text("similarity(alias_norm, :q) > 0.6")
                    .bindparams(q=norm)
                )
                .order_by(
                    text("similarity(alias_norm, :q) DESC")
                    .bindparams(q=norm)
                )
                .limit(1)
            )
            alias_row = (await self._session.execute(stmt)).scalar_one_or_none()
        except Exception:
            # pg_trgm might not be available or table missing.
            logger.debug("pg_trgm alias search failed — skipping")
            alias_row = None

        if alias_row is not None:
            _alias_cache.put(norm, alias_row.code)
            company = await self.get_by_ticker(alias_row.code)
            if company is not None:
                return AliasMatch(
                    company=company,
                    alias=alias_row.alias,
                    code=alias_row.code,
                    method="trgm",
                )

        # No match — cache negative result to avoid repeated DB hits.
        _alias_cache.put(norm, None)
        return None

    # ── Reads ───────────────────────────────────────────────────────────────────────

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
