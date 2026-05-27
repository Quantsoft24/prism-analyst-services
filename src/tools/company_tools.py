"""Agent-callable tools for company metadata — backed by the catalog DB
(``company_industry`` on the stock_chat Postgres, 4,773 companies).

Three tools, all typo-tolerant (rapidfuzz Python-side; will move to
in-DB pg_trgm when the catalog admin enables it — see
``docs/CATALOG_DB_INDEXES.md``):

  * ``lookup_company(ticker)``       — exact ticker, fuzzy fallback,
                                       single-match promotion to a hit
  * ``search_companies(query, …)``   — paginated list with truncation
                                       hint when too many match
  * ``list_covered_sectors()``       — distinct sectors, deduped to
                                       collapse "Software & Services"
                                       vs "Software and Services"

Reliability hardening (2026-05-26):
  * Empty / whitespace-only input returns a structured error (the agent
    surfaces "I need a ticker or company name" instead of "not found")
  * Multi-word lookup_company input ("Tata Consultancy") is promoted to
    a ``found: true`` hit with ``disambiguation_note`` when rapidfuzz
    finds exactly one match scoring ≥ 85 — the agent says "Interpreting
    that as TCS — Tata Consultancy Services" so the user can correct
  * search_companies caps the visible items at 10 + carries
    ``total_matched`` and ``truncated`` so the LLM can say "and N more"
    instead of dumping 25 rows
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rapidfuzz import fuzz

from src.core.catalog_database import catalog_session_scope
from src.integrations.tools._errors import make_error
from src.models.catalog import CompanyIndustry
from src.repositories.company_repo import (
    CompanyRepository,
    _candidate_score,
    _normalize_query,
)

if TYPE_CHECKING:
    from google.adk.tools import FunctionTool


# ── Tuning knobs ─────────────────────────────────────────────────────────

# When lookup_company falls into the fuzzy path, we promote the top hit
# to a ``found: true`` shape ONLY when its rapidfuzz score is at least
# this high AND there's no equally-strong runner-up. The 85 cutoff was
# calibrated against the 4,773-row catalog using inputs like
# "Tata Consultancy" → TCS (95+) and "Bharti" → BHARTIARTL (90+) while
# rejecting "Tata" (multi-match: TATAMOTORS / TATAPOWER / TATASTEEL etc.).
_LOOKUP_PROMOTE_THRESHOLD = 85.0

# search_companies caps the visible items at this number even when the
# user passes a larger ``limit``. Beyond this the LLM picks badly — see
# Rule 3 in the agent prompt.
_SEARCH_VISIBLE_CAP = 10

# Input-shape recognisers for lookup_company:
#   * ISIN — 12 chars: 2 letter country prefix + 9 alphanumeric + 1 digit
#     check char (e.g. INE002A01018, US0378331005). Indian listings start
#     with ``IN``; we accept-and-resolve those, refuse foreign ones.
#   * BSE scrip code — 4–7 digit numeric (e.g. 500325, 532540). The
#     catalog stores NSE symbols only, so these get a friendly refusal
#     instead of a silent fuzzy miss.
_ISIN_PATTERN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
_BSE_NUMERIC_PATTERN = re.compile(r"^[0-9]{4,7}$")

# Hard cap on input strings to both tools. Real tickers are ≤ 16 chars,
# real ISINs are 12 chars, real company names ≤ 80 chars. 200 chars is
# generous headroom while still refusing pathological 1k+ character
# inputs that would crush rapidfuzz's Python re-rank. Returns a
# structured `input_too_long` error well before the DB is touched.
_MAX_INPUT_LENGTH = 200

# Fuzzy-match a sector hint against the catalog's actual sector names
# when an exact filter misses. ``75`` is a moderate threshold — strong
# enough to catch "banks" → "Banks", "software" → "Software & Services",
# while rejecting noise like "asdfqwer".
_SECTOR_FUZZY_THRESHOLD = 75.0


# ── Shared formatters ──────────────────────────────────────────────────────


def _to_full_row(c: CompanyIndustry, disambiguation_note: str | None = None) -> dict:
    """Full record shape used by ``lookup_company``."""
    payload: dict = {
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
    if disambiguation_note:
        payload["disambiguation_note"] = disambiguation_note
    return payload


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
    """Look up a company by its NSE ticker, Indian ISIN, or short name.

    Accepted input shapes (auto-detected in order):
      * NSE ticker — 3-6 letters (TCS, RELIANCE, MOIL) → exact match on
        ``company_industry.code``
      * Indian ISIN — 12 chars starting with ``IN`` (INE002A01018) →
        resolved via ``company_industry.isin``
      * Foreign ISIN — 12 chars NOT starting with IN (US0378331005,
        GB...) → returns a friendly "Indian listings only" message
      * BSE scrip code — 4-7 digit numeric (500325, 532540) → returns a
        friendly "use the NSE letter symbol instead" message (BSE-only
        resolution requires a catalog-side schema change)
      * Short company name — "Tata Consultancy", "Bharti Airtel" →
        fuzzy-resolved via rapidfuzz; if the top hit scores ≥ 85 AND
        the runner-up is ≥10 points behind, promoted to a ``found: true``
        response with a ``disambiguation_note`` field. Quote it in your
        reply ("Interpreting that as TCS — Tata Consultancy Services")
        so the user can correct.

    Args:
        ticker: Ticker / ISIN / numeric code / short company name. Empty
            / whitespace returns a structured error.

    Returns:
        Hit:                ``{found: true, ticker, name, exchange,
                                sector, industry, country, isin, ...,
                                disambiguation_note?}``
        Miss (with hints):  ``{found: false, ticker, suggestions: [...]}``
        Structured refusal: ``{ok: false, error, error_code, next_action}``
                            codes: missing_input, foreign_isin,
                                   bse_code_unsupported.
    """
    cleaned = (ticker or "").strip()
    if not cleaned:
        return make_error(
            message="I need a ticker or company name to look up.",
            code="missing_input",
            next_action="ask_user_to_clarify",
        )
    if len(cleaned) > _MAX_INPUT_LENGTH:
        return make_error(
            message=(
                f"That input is too long ({len(cleaned)} chars). Real "
                "tickers are at most 16 chars, ISINs are 12, company "
                "names rarely exceed 80. Trim to the ticker or first few "
                "words of the company name."
            ),
            code="input_too_long",
            next_action="ask_user_to_clarify",
        )

    # Input-shape routing (uppercased + stripped before matching).
    upper = cleaned.upper()

    # 1) ISIN shape — resolve via the catalog's ``isin`` column directly.
    if _ISIN_PATTERN.match(upper):
        if not upper.startswith("IN"):
            return make_error(
                message=(
                    f"That looks like a foreign-market ISIN ({upper[:2]}). "
                    "I only cover Indian NSE/BSE listings — give me the "
                    "Indian ticker or ISIN instead."
                ),
                code="foreign_isin",
                next_action="ask_user_to_clarify",
            )
        async with catalog_session_scope() as session:
            repo = CompanyRepository(session)
            c = await repo.get_by_isin(upper)
            if c is not None:
                return _to_full_row(
                    c,
                    disambiguation_note=f"Resolved ISIN {upper} → {c.code}",
                )
        # ISIN-shaped but not in the catalog. Tell the user honestly
        # rather than fuzzy-matching a 12-char code against names.
        return {
            "found": False,
            "ticker": upper,
            "suggestions": [],
            "isin_lookup_miss": True,
        }

    # 2) BSE scrip code shape — politely refuse (catalog is NSE-only).
    if _BSE_NUMERIC_PATTERN.match(cleaned):
        return make_error(
            message=(
                f"I only have NSE tickers right now — try the letter "
                f"symbol (e.g. RELIANCE instead of {cleaned}). BSE-only "
                "listings will be supported when the catalog gains a "
                "`bse_code` column."
            ),
            code="bse_code_unsupported",
            next_action="ask_user_to_clarify",
        )

    # 3) Treat the whole-string upper-case form as a candidate ticker.
    # For multi-word inputs ("Tata Consultancy") this won't hit and we
    # fall into the fuzzy path below — that's the desired flow.
    ticker_form = upper
    async with catalog_session_scope() as session:
        repo = CompanyRepository(session)
        c = await repo.get_by_ticker(ticker_form)
        if c is not None:
            return _to_full_row(c)

        # Miss on exact ticker → run the fuzzy path. We ask for ``limit=3``
        # because we want to *detect* a single dominant match while still
        # having visibility into close runners-up for the suggestions list.
        result = await repo.list(search=cleaned, limit=3)

        # Promotion: if the rapidfuzz top hit is unambiguously good
        # (score ≥ _LOOKUP_PROMOTE_THRESHOLD) AND the runner-up is at
        # least 10 points behind, treat it as a resolved match instead
        # of a "did you mean" miss.
        query_norm = _normalize_query(cleaned)
        if result.items and query_norm:
            top = result.items[0]
            top_score = _candidate_score(top, query_norm)
            runner_up_score = (
                _candidate_score(result.items[1], query_norm)
                if len(result.items) > 1
                else 0
            )
            if top_score >= _LOOKUP_PROMOTE_THRESHOLD and (
                top_score - runner_up_score >= 10
                or top.code.upper() == ticker_form
            ):
                note = (
                    f"Interpreting that as {top.code}"
                    + (f" — {top.company_name}" if top.company_name else "")
                    + f' (resolved from "{cleaned}")'
                )
                return _to_full_row(top, disambiguation_note=note)

        # No clean promotion — fall back to the "did you mean" surface.
        near = (result.items or []) + (result.suggestions or [])
        return {
            "found": False,
            "ticker": ticker_form,
            "suggestions": [_to_suggestion(c) for c in near[:3]],
        }


async def search_companies(query: str, sector: str | None = None, limit: int = 10) -> dict:
    """Search the Indian-markets catalog by name, ticker, or sector.

    Typo-tolerant on BOTH inputs:
      * The ``query`` matches ticker / scrip code / company name via
        rapidfuzz — "Reliac" → Reliance group.
      * The ``sector`` filter is exact-match first; on miss, the tool
        fuzzy-resolves the hint to the closest catalog sector and uses
        that. The resolved name comes back in ``resolved_sector`` so
        you can tell the user what we used.

    When the response carries ``truncated: true``, DO NOT list every
    match — tell the user "N total matched, here are the top 10 —
    narrow by sector or partial name to see more".

    Args:
        query: Free-text search — matches ticker/scrip code or company name.
            Pass an empty string to list all companies in a sector.
            Inputs longer than 200 chars get a structured ``input_too_long``
            error (real names are ≤ 80 chars).
        sector: Sector hint — exact match preferred, fuzzy-resolved if
            unique close match exists. Use ``list_covered_sectors`` for
            the canonical list.
        limit: Max visible results. Internally capped at 10 even when a
            larger value is passed (deters the LLM from dumping huge lists).

    Returns:
        Hit:                ``{total_matched, truncated, items[],
                                suggestions[], resolved_sector?}``
        Sector miss:        ``{ok: false, error, error_code:
                                "unknown_sector", next_action:
                                "ask_user_to_clarify",
                                sector_suggestions: [...]}``
        Bad input:          ``{ok: false, error, error_code:
                                "input_too_long"|"missing_input"}``
    """
    # Guard against pathological inputs BEFORE hitting the repo. A
    # 10K-char query would crush rapidfuzz's Python re-rank.
    if query is not None and len(query) > _MAX_INPUT_LENGTH:
        return make_error(
            message=(
                f"Search query is too long ({len(query)} chars). Real "
                "company names rarely exceed 80 chars. Trim and retry."
            ),
            code="input_too_long",
            next_action="ask_user_to_clarify",
        )
    if sector is not None and len(sector) > _MAX_INPUT_LENGTH:
        return make_error(
            message=(
                f"Sector filter is too long ({len(sector)} chars). Use "
                "a short sector name from `list_covered_sectors`."
            ),
            code="input_too_long",
            next_action="ask_user_to_clarify",
        )

    visible_limit = max(1, min(limit, _SEARCH_VISIBLE_CAP))
    async with catalog_session_scope() as session:
        repo = CompanyRepository(session)

        # Resolve the sector parameter against the catalog's canonical
        # list. The repo treats `sector` as an exact-match filter, so
        # off-case ("banks") or off-name ("energy" when catalog has
        # "Oil & Gas") would silently return zero. Fuzzy-resolve once,
        # then pass the resolved name (or report a structured miss).
        resolved_sector: str | None = sector
        if sector:
            resolved_sector, sector_suggestions = await _resolve_sector(repo, sector)
            if resolved_sector is None:
                return make_error(
                    message=(
                        f'No catalog sector matches "{sector}". '
                        "Pick one from the suggestions or call "
                        "`list_covered_sectors` for the full list."
                    ),
                    code="unknown_sector",
                    next_action="ask_user_to_clarify",
                    detail=", ".join(sector_suggestions[:5]),
                )

        result = await repo.list(
            search=query.strip() or None,
            sector=resolved_sector,
            limit=visible_limit,
            offset=0,
        )
        payload: dict = {
            "total_matched": result.total,
            "truncated": result.total > len(result.items),
            "items": [_to_list_row(c) for c in result.items],
            "suggestions": [_to_suggestion(c) for c in result.suggestions],
        }
        # Surface the resolved sector ONLY when fuzzy resolution kicked in
        # (i.e. the input differed from what we used). Helps the agent
        # tell the user "I filtered to Banks based on 'banks'".
        if sector and resolved_sector and resolved_sector != sector:
            payload["resolved_sector"] = resolved_sector
        return payload


async def _resolve_sector(
    repo: CompanyRepository, hint: str
) -> tuple[str | None, list[str]]:
    """Resolve a sector hint against the catalog's canonical sector list.

    Returns ``(resolved_name, suggestions)``:
      * ``resolved_name`` — the catalog's actual sector name when we can
        confidently map the hint, else ``None`` if no close match.
      * ``suggestions`` — top fuzzy-near candidates (always populated,
        even on a hit) so callers can include them on a miss.

    Resolution order:
      1. Case-insensitive exact match against any canonical sector
         (handles "banks" → "Banks", "PHARMA" → "Pharma").
      2. Rapidfuzz best-match against the deduped list at score
         ≥ _SECTOR_FUZZY_THRESHOLD (handles "softwere" → "Software & Services").
      3. Otherwise: return ``None`` + top-3 suggestions for the caller
         to surface to the user.
    """
    raw = await repo.distinct_sectors()
    canonical = _dedupe_sectors(raw)
    if not canonical:
        return None, []

    hint_lower = hint.strip().lower()
    if not hint_lower:
        return None, canonical[:3]

    # 1) Case-insensitive exact match.
    for s in canonical:
        if s.lower() == hint_lower:
            return s, []

    # 2) Fuzzy best-match — score against the canonical list.
    scored = [
        (s, fuzz.token_set_ratio(hint_lower, s.lower()))
        for s in canonical
    ]
    scored.sort(key=lambda sc: -sc[1])
    top_name, top_score = scored[0]
    if top_score >= _SECTOR_FUZZY_THRESHOLD:
        return top_name, [s for s, _ in scored[1:4]]

    # 3) No confident match — return the top-3 suggestions for the
    # "did you mean these sectors?" surface.
    return None, [s for s, _ in scored[:3]]


async def list_covered_sectors() -> dict:
    """Distinct industries / sectors available in the catalog (deduped).

    The upstream catalog has near-duplicates ("Software & Services" vs
    "Software and Services") — this tool collapses them to one canonical
    entry per normalized form. Use it when the user asks "what sectors
    do you cover?" or before filtering ``search_companies`` by sector.

    Returns:
        Dict with ``sectors`` (list of distinct industry strings,
        alphabetical, ~150–180 entries after dedup).
    """
    async with catalog_session_scope() as session:
        repo = CompanyRepository(session)
        raw_sectors = await repo.distinct_sectors()
    deduped = _dedupe_sectors(raw_sectors)
    return {"sectors": deduped}


# ── Sector deduplication helper ───────────────────────────────────────────


def _normalize_sector(name: str) -> str:
    """Collapse near-duplicates: lowercase, strip ``& / and``, collapse spaces.

    Example::
        "Software & Services" → "software services"
        "Software and Services" → "software services"
        "Oil & Gas" → "oil gas"
    """
    s = (name or "").lower()
    # Replace word "and" and the "&" / "/" punctuation with a space.
    s = re.sub(r"\band\b|[&/]", " ", s)
    # Collapse any non-alphanumeric to a space.
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _dedupe_sectors(sectors: list[str]) -> list[str]:
    """Group sectors by normalized form, keep the canonical (longest) original.

    Preserves the catalog's original casing in the returned strings (we
    pick the longest original because longer usually means more
    specific / better-rendered). Output is alphabetical by display string.
    """
    groups: dict[str, str] = {}
    for s in sectors:
        if not s:
            continue
        key = _normalize_sector(s)
        if not key:
            continue
        current = groups.get(key)
        if current is None or len(s) > len(current):
            groups[key] = s
    return sorted(groups.values())


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
