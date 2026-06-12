"""Company resolution over the investment-DB ``master_securities`` table.

This is PRISM's single source of truth for "the user typed a company — which
listed security do they mean?". It replaced the old catalog-backed company
resolution entirely (the catalog DB has since been retired); the agent's
``company_tools`` now back onto this resolver.

Design
------
* **Data source = the Stock Dashboard's cached security list.** We reuse
  ``StockRepository.list_securities()`` (``src/repositories/stock_repo.py``),
  which keeps an in-process 6-hour-TTL cache of all ~8,239 ``SecurityRead`` rows
  ``{security_id, security_name, symbol, isin, exchange, sector}``. No second
  cache, no bespoke refresh — when that cache rolls over (or the process
  restarts) our derived index rebuilds automatically (we key it on the cached
  list's object identity).
* **A company = an ISIN group.** A dual-listed name has two rows (NSE + BSE) with
  the same ISIN and distinct ``security_id``. We collapse them into one logical
  ``CompanyGroup`` carrying both ids; the ``security_id`` we hand downstream
  prefers the NSE row (BSE fallback) — stock-chat matches it against BOTH
  ``security_id_bse`` and ``security_id_nse`` so either resolves the company.
* **Edge-case ladder** (first confident win), 100% DB-driven — NO hardcoded
  alias list: exact security_id / ISIN → DB ``symbol`` column (incl. "M&M") →
  acronym *derived from* ``security_name`` (RIL/SBI/HUL) → normalized exact name
  → rapidfuzz fuzzy. Shorthand ranking uses a DB prominence signal: current Nifty
  index membership (``index_constituent``) — an acronym auto-resolves when exactly
  one of its candidates is index-listed; otherwise we return the top-N nearest
  companies (prominent first) for the agent to surface as a clarification.

Resolution always lands on a ``master_securities`` ``security_id`` — we never
hand a free-text name to downstream tools.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz
from sqlalchemy import text

from src.core.investment_database import investment_session_scope, is_investment_configured
from src.repositories.stock_repo import StockRepository
from src.schemas.stock import SecurityRead

# ── Tuning knobs (centralized — no scattered literals) ──────────────────────

# rapidfuzz scores are 0-100. A single confident match is promoted to "resolved"
# only when it clears the hit threshold AND beats the runner-up by the margin —
# otherwise we ask the user. Calibrated against the 8,239-row master.
_HIT_THRESHOLD = 88.0          # top score must reach this to auto-resolve
_RUNNER_UP_MARGIN = 12.0       # …and beat #2 by at least this much
_CANDIDATE_FLOOR = 55.0        # scores below this aren't even offered as candidates
_MAX_CANDIDATES = 6            # options we surface on an ambiguous/weak match
_SEARCH_VISIBLE_CAP = 10       # search_companies visible-row cap
_MAX_INPUT_LENGTH = 200        # reject pathological inputs before fuzzy re-rank

# Input-shape recognisers.
_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
_BSE_NUMERIC_RE = re.compile(r"^[0-9]{4,7}$")

# Legal-form suffixes stripped before name matching ("TCS Ltd" → "tcs").
_NOISE_SUFFIXES = (
    "private limited", "pvt limited", "pvt ltd", "limited", "ltd", "ltd.",
    "corporation", "corp", "company", "co.", "co",
)

# Nothing here is a hardcoded alias list. Ticker shorthand comes from the DB
# ``symbol`` column; colloquial shorthand (RIL / SBI / HUL) is *derived* from
# ``security_name`` at load time via ``_acronyms`` below. Words dropped when
# deriving an acronym: pure connectives + legal-form words (everything else
# contributes its first letter).
_ACRONYM_STOP = {"of", "the", "and"}
_LEGAL_WORDS = {"ltd", "limited", "pvt", "private", "corp", "corpn",
                "corporation", "company", "co"}


def _acronyms(name: str) -> set[str]:
    """Derive candidate acronyms from a company name — DB-driven, no hardcoded
    map. Two variants: WITH the legal suffix ("Reliance Industries Ltd" → ``RIL``,
    "Bharat Petroleum Corpn Ltd" → ``BPCL``, "State Bank Of India" → ``SBI``) and
    WITHOUT it ("Reliance Industries" → ``RI``). Collisions across companies are
    fine — they degrade to a clarification."""
    words = [w for w in re.findall(r"[a-z0-9]+", name.lower()) if w not in _ACRONYM_STOP]
    if not words:
        return set()
    full = "".join(w[0] for w in words)
    core = "".join(w[0] for w in words if w not in _LEGAL_WORDS)
    return {a.upper() for a in (full, core) if 2 <= len(a) <= 8}


def _strip_symbol(sym: str) -> str:
    """Bare alnum form of a ticker so "M&M"/"m&m"/"MM" all collide, and
    "BAJAJ-AUTO" → "BAJAJAUTO"."""
    return re.sub(r"[^A-Z0-9]", "", sym.upper())


# ── Logical company (ISIN group) ────────────────────────────────────────────


@dataclass(slots=True)
class CompanyGroup:
    """One logical company, collapsing its NSE + BSE security rows."""

    isin: str | None
    name: str
    symbol: str | None
    sector: str | None
    security_id_nse: int | None = None
    security_id_bse: int | None = None
    # Prominence from Nifty index membership (5=Nifty 50 … 1=Nifty 500, 0=none).
    # Used only to RANK/break ties — never to fabricate a match.
    prominence: int = 0

    @property
    def security_id(self) -> int:
        """The id we hand downstream — NSE preferred, BSE fallback. Either
        resolves the company in stock-chat (it matches both id columns)."""
        sid = self.security_id_nse if self.security_id_nse is not None else self.security_id_bse
        assert sid is not None  # a group always has at least one row
        return sid

    @property
    def exchanges(self) -> list[str]:
        out = []
        if self.security_id_nse is not None:
            out.append("NSE")
        if self.security_id_bse is not None:
            out.append("BSE")
        return out

    def to_dict(self) -> dict:
        return {
            "security_id": self.security_id,
            "name": self.name,
            "symbol": self.symbol,
            "isin": self.isin,
            "sector": self.sector,
            "exchanges": self.exchanges,
            "security_id_nse": self.security_id_nse,
            "security_id_bse": self.security_id_bse,
        }


@dataclass(slots=True)
class ResolveOutcome:
    """Result of a resolution attempt.

    ``resolved`` + ``company`` → a single confident match.
    Otherwise ``candidates`` holds the nearest companies (may be empty on a true
    miss) for the caller to surface as a clarification.
    """

    query: str
    resolved: bool
    company: CompanyGroup | None
    candidates: list[CompanyGroup] = field(default_factory=list)
    reason: str = ""  # short machine hint: "exact_id" | "isin" | "symbol" | …


# ── In-memory index (derived from the cached security list) ─────────────────


@dataclass(slots=True)
class _Index:
    source_id: int  # id() of the SecurityRead list this index was built from
    groups: list[CompanyGroup]
    by_security_id: dict[int, CompanyGroup]
    by_isin: dict[str, CompanyGroup]
    by_symbol: dict[str, list[CompanyGroup]]   # exact ticker + bare-alnum form
    by_acronym: dict[str, list[CompanyGroup]]  # derived from security_name
    by_norm_name: dict[str, list[CompanyGroup]]
    norm_names: list[tuple[str, CompanyGroup]]  # (normalized name, group) for fuzzy


_index: _Index | None = None


def _normalize(text: str) -> str:
    """Lowercase, ``&``→``and`` shape, drop punctuation, strip legal suffixes,
    collapse whitespace. Conservative — no transliteration."""
    s = (text or "").strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for suffix in _NOISE_SUFFIXES:
        if s.endswith(" " + suffix) or s == suffix:
            s = s[: -len(suffix)].strip()
            break
    # Merge runs of single-character tokens — the master stores some names with
    # spaced initials ("H D F C Bank Ltd." → "hdfc bank", "I T C" → "itc"), and
    # users type the joined form. Consecutive single letters are always initials
    # in company names, so this is safe.
    merged: list[str] = []
    buf: list[str] = []
    for tok in s.split():
        if len(tok) == 1:
            buf.append(tok)
        else:
            if buf:
                merged.append("".join(buf))
                buf = []
            merged.append(tok)
    if buf:
        merged.append("".join(buf))
    return " ".join(merged)


def _build_index(
    securities: list[SecurityRead], prominence: dict[int, int] | None = None,
) -> _Index:
    """Collapse the flat security rows into ISIN-grouped companies + lookups.
    ``prominence`` maps security_id → Nifty-membership score (optional)."""
    prom = prominence or {}
    groups: dict[str, CompanyGroup] = {}  # key (isin or fallback) → group
    for s in securities:
        if s.security_id is None:
            continue
        key = (s.isin or "").upper() or f"sid:{s.security_id}"
        g = groups.get(key)
        if g is None:
            g = CompanyGroup(
                isin=(s.isin or None),
                name=s.security_name or s.symbol or key,
                symbol=s.symbol,
                sector=s.sector,
            )
            groups[key] = g
        else:
            # Fill any gaps from the second-exchange row.
            g.name = g.name or s.security_name or g.name
            g.symbol = g.symbol or s.symbol
            g.sector = g.sector or s.sector
        g.prominence = max(g.prominence, prom.get(s.security_id, 0))
        if (s.exchange or "").upper() == "NSE":
            g.security_id_nse = s.security_id
            g.symbol = s.symbol or g.symbol  # prefer the NSE ticker
        elif (s.exchange or "").upper() == "BSE":
            g.security_id_bse = s.security_id
        else:  # unknown exchange — keep the id so the company is still resolvable
            g.security_id_nse = g.security_id_nse or s.security_id

    group_list = list(groups.values())
    by_sid: dict[int, CompanyGroup] = {}
    by_isin: dict[str, CompanyGroup] = {}
    by_symbol: dict[str, list[CompanyGroup]] = {}
    by_acronym: dict[str, list[CompanyGroup]] = {}
    by_norm_name: dict[str, list[CompanyGroup]] = {}
    norm_names: list[tuple[str, CompanyGroup]] = []

    def _add(d: dict[str, list[CompanyGroup]], key: str, g: CompanyGroup) -> None:
        bucket = d.setdefault(key, [])
        if g not in bucket:  # dedupe (a group can yield the same key twice)
            bucket.append(g)

    for g in group_list:
        for sid in (g.security_id_nse, g.security_id_bse):
            if sid is not None:
                by_sid[sid] = g
        if g.isin:
            by_isin[g.isin.upper()] = g
        if g.symbol:
            _add(by_symbol, g.symbol.strip().upper(), g)
            stripped = _strip_symbol(g.symbol)
            if stripped:
                _add(by_symbol, stripped, g)
        for acro in _acronyms(g.name):
            _add(by_acronym, acro, g)
        norm = _normalize(g.name)
        if norm:
            by_norm_name.setdefault(norm, []).append(g)
            norm_names.append((norm, g))
    return _Index(
        source_id=id(securities),
        groups=group_list,
        by_security_id=by_sid,
        by_isin=by_isin,
        by_symbol=by_symbol,
        by_acronym=by_acronym,
        by_norm_name=by_norm_name,
        norm_names=norm_names,
    )


_prominence_cache: dict[int, int] | None = None


async def _load_prominence() -> dict[int, int]:
    """security_id → prominence score from Nifty index membership (5=Nifty 50,
    4=Next 50, 3=Nifty 100, 2=Nifty 200, 1=Nifty 500; absent = 0). Cheap
    (~0.1s, indices are small) and cached in-process. Pure DB-derived ranking
    signal — never used to fabricate a match, only to order candidates and to
    auto-pick when exactly one candidate of an acronym is index-listed."""
    global _prominence_cache
    if _prominence_cache is not None:
        return _prominence_cache
    async with investment_session_scope() as session:
        # CURRENT membership only — use each index's latest snapshot, so a name
        # that was dropped from an index (e.g. Reliance Capital) doesn't keep a
        # stale prominence from historical membership.
        res = await session.execute(text(
            "WITH latest AS ("
            "  SELECT index_id, MAX(date) AS d FROM index_constituent GROUP BY index_id"
            ") "
            "SELECT ic.security_id AS security_id, MIN(ic.index_id) AS mi "
            "FROM index_constituent ic "
            "JOIN latest l ON l.index_id = ic.index_id AND l.d = ic.date "
            "GROUP BY ic.security_id"
        ))
        # index_id 1=Nifty50 … 5=Nifty500 → score 6-id (5…1); not present → 0.
        _prominence_cache = {int(r.security_id): max(0, 6 - int(r.mi)) for r in res}
    return _prominence_cache


async def _get_index() -> _Index | None:
    """Return the resolution index, rebuilding only when the underlying
    security cache rolled over (we key on the cached list's identity).
    Returns ``None`` when the investment DB isn't configured."""
    global _index
    if not is_investment_configured():
        return None
    async with investment_session_scope() as session:
        securities = await StockRepository(session).list_securities()
    if _index is None or _index.source_id != id(securities):
        try:
            prominence = await _load_prominence()
        except Exception:  # noqa: BLE001 — ranking is best-effort; never block resolution
            prominence = {}
        _index = _build_index(securities, prominence)
    return _index


def _best_token_fuzz(query_norm: str, name_norm: str) -> float:
    """Best single-token alignment between query and name (≥2-char tokens). High
    only when SOME word genuinely matches — including 1-2 char typos
    ("relianse"≈"reliance"=88). Near-zero for unrelated names ("blinkit" vs the
    tokens of "ITC Ltd"), which is what gates out spurious substring matches."""
    qt = [t for t in query_norm.split() if len(t) >= 2]
    nt = [t for t in name_norm.split() if len(t) >= 2]
    if not qt or not nt:
        return 0.0
    return max(fuzz.ratio(a, b) for a in qt for b in nt)


# A real token must align this well for a fuzzy match to count. Allows 1-2 char
# typos; rejects "blinkit"→"ITC" / "Zomato"→"Automation" / "Bistro by Blinkit"→"IST".
_TOKEN_GATE = 80.0


def _score(query_norm: str, name_norm: str) -> float:
    """Fuzzy similarity — best of token-set (word order) and partial (truncation/
    typos). Candidacy additionally REQUIRES a genuine token match (see
    ``_best_token_fuzz`` / the ``_TOKEN_GATE`` filter in the fuzzy ladder), which
    is what stops unrelated names ("blinkit"→"ITC", "Zomato"→"…Automation") from
    surfacing on a strong partial-ratio substring overlap alone."""
    return max(
        fuzz.token_set_ratio(query_norm, name_norm),
        fuzz.partial_ratio(query_norm, name_norm),
    )


# ── Public API ──────────────────────────────────────────────────────────────


async def resolve(query: str) -> ResolveOutcome:
    """Resolve a free-text company reference to a single ``CompanyGroup`` or a
    ranked candidate list. Never raises — a misconfigured DB or empty query
    yields ``resolved=False`` with empty candidates."""
    q = (query or "").strip()
    if not q or len(q) > _MAX_INPUT_LENGTH:
        return ResolveOutcome(query=q, resolved=False, company=None, reason="bad_input")
    idx = await _get_index()
    if idx is None:
        return ResolveOutcome(query=q, resolved=False, company=None, reason="not_configured")
    return _resolve_in_index(q, idx)


def _resolve_in_index(q: str, idx: _Index) -> ResolveOutcome:
    """Pure resolution ladder over a prebuilt index (no I/O — unit-testable)."""
    upper = q.upper()

    # 0) An explicit "security_id N" embedded in the text resolves exactly. This
    #    makes the clarification round-trip bulletproof: when the user picks an
    #    option, the reply looks like "Reliance Industries Ltd. — security_id
    #    2228", and we lock onto that id regardless of the surrounding label.
    m = re.search(r"security[_ ]?id\D{0,3}(\d{1,9})", q, re.IGNORECASE)
    if m:
        g = idx.by_security_id.get(int(m.group(1)))
        if g is not None:
            return ResolveOutcome(query=q, resolved=True, company=g, reason="exact_id")

    # 1) Exact security_id (the fast-path the clarification UI sends back).
    if q.isdigit() and len(q) <= 9:
        g = idx.by_security_id.get(int(q))
        if g is not None:
            return ResolveOutcome(query=q, resolved=True, company=g, reason="exact_id")
    # 2) ISIN.
    if _ISIN_RE.match(upper):
        g = idx.by_isin.get(upper)
        if g is not None:
            return ResolveOutcome(query=q, resolved=True, company=g, reason="isin")
        return ResolveOutcome(query=q, resolved=False, company=None, reason="isin_miss")
    # 3) Ticker / acronym fast-path — but ONLY for "code-shaped" input, so a
    #    plain word like "Reliance" (which collides with the RELIANCE symbol)
    #    falls through to a clarification among the Reliance companies. Code-shaped
    #    = ALL-CAPS code ("RELIANCE", "M&M", "BPCL") OR a short ≤5 single token
    #    ("tcs", "ril", "sbi", "wipro") that users type as shorthand. Longer
    #    mixed/lower words ("reliance", "infosys") go to the name path instead.
    code_shaped = bool(
        re.fullmatch(r"[A-Z0-9&.\-]{2,12}", q)
        or (len(q) <= 5 and re.fullmatch(r"[A-Za-z0-9&.\-]+", q))
    )
    if code_shaped:
        # (a) DB symbol column (exact ticker, incl. "M&M"; or its bare-alnum form).
        for key in (upper, _strip_symbol(q)):
            hits = idx.by_symbol.get(key)
            if hits and len(hits) == 1:
                return ResolveOutcome(query=q, resolved=True, company=hits[0], reason="symbol")
        # (b) Acronym derived from the name (RIL, SBI, HUL). Rank by prominence;
        #     if exactly ONE candidate is index-listed it's effectively
        #     unambiguous (e.g. "RIL" → only Reliance Industries is in a Nifty
        #     index) → resolve. Otherwise clarify, most-prominent first.
        acro_hits = idx.by_acronym.get(_strip_symbol(q))
        if acro_hits:
            ranked = sorted(acro_hits, key=lambda g: (-g.prominence, g.name.lower()))
            prominent = [g for g in ranked if g.prominence > 0]
            if len(ranked) == 1 or len(prominent) == 1:
                return ResolveOutcome(
                    query=q, resolved=True, company=(prominent or ranked)[0], reason="acronym",
                )
            return ResolveOutcome(
                query=q, resolved=False, company=None,
                candidates=ranked[:_MAX_CANDIDATES], reason="acronym_ambiguous",
            )

    # 4) Normalized exact name.
    search_norm = _normalize(q)
    name_hits = idx.by_norm_name.get(search_norm)
    if name_hits and len(name_hits) == 1:
        return ResolveOutcome(query=q, resolved=True, company=name_hits[0], reason="exact_name")

    # 5) Fuzzy ranking over all groups.
    if not search_norm:
        return ResolveOutcome(query=q, resolved=False, company=None, reason="empty_after_norm")
    # Score each name AND its token-gate (does any real word align?). The gate is
    # a HARD requirement below: it's what separates "Reliance" (token "reliance"
    # = 100) from "blinkit"/"Zomato" (no word aligns ≥ gate → no candidates).
    scored: list[tuple[float, float, CompanyGroup]] = [
        (_score(search_norm, nn), _best_token_fuzz(search_norm, nn), g)
        for nn, g in idx.norm_names
    ]
    # Highest score first; break ties by prominence (Nifty membership) so e.g.
    # "Reliance" lists Reliance Industries first — then clarifies among the rest.
    scored.sort(key=lambda t: (-t[0], -t[2].prominence, t[2].name.lower()))
    top_score, top_gate, top = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0

    if (
        top_score >= _HIT_THRESHOLD
        and top_gate >= _TOKEN_GATE
        and (top_score - runner_up) >= _RUNNER_UP_MARGIN
    ):
        return ResolveOutcome(query=q, resolved=True, company=top, reason="fuzzy")

    candidates = [
        g for sc, gate, g in scored
        if sc >= _CANDIDATE_FLOOR and gate >= _TOKEN_GATE
    ][:_MAX_CANDIDATES]
    if not candidates:
        # Nothing matched well — the name is likely NOT a listed company (a
        # private subsidiary, brand, product, or an unlisted/typo'd name). Return
        # NO garbage candidates; downstream tells the user it's outside coverage
        # and offers the securities search instead of guessing ITC for "blinkit".
        return ResolveOutcome(
            query=q, resolved=False, company=None, candidates=[], reason="not_found"
        )
    return ResolveOutcome(
        query=q, resolved=False, company=None, candidates=candidates, reason="ambiguous"
    )


async def get_by_security_id(security_id: int) -> CompanyGroup | None:
    idx = await _get_index()
    return idx.by_security_id.get(security_id) if idx else None


async def search(
    query: str | None, sector: str | None = None, limit: int = _SEARCH_VISIBLE_CAP,
) -> tuple[list[CompanyGroup], int]:
    """List view for the ``search_companies`` tool — distinct companies matching
    a name/symbol substring and/or an exact sector. Returns ``(rows, total)``."""
    idx = await _get_index()
    if idx is None:
        return [], 0
    rows = idx.groups
    if sector:
        sec = sector.strip().lower()
        rows = [g for g in rows if (g.sector or "").lower() == sec]
    q = (query or "").strip()
    if q:
        qn = _normalize(q)
        scored = [(_score(qn, _normalize(g.name)), g) for g in rows]
        scored = [(sc, g) for sc, g in scored if sc >= _CANDIDATE_FLOOR]
        scored.sort(key=lambda sc: (-sc[0], sc[1].name.lower()))
        rows = [g for _, g in scored]
    else:
        rows = sorted(rows, key=lambda g: g.name.lower())
    total = len(rows)
    return rows[: max(1, min(limit, _SEARCH_VISIBLE_CAP))], total


async def list_sectors() -> list[str]:
    """Distinct ``master_securities.sector`` values (the 22-value SEBI taxonomy)."""
    idx = await _get_index()
    if idx is None:
        return []
    return sorted({g.sector for g in idx.groups if g.sector})


def to_clarification_options(groups: list[CompanyGroup]) -> list[dict]:
    """Build deterministic single-select options from candidate companies.
    ``value`` is the security_id (int) the UI sends back to resolve exactly."""
    return [
        {
            "id": str(g.security_id),
            "label": g.name,
            "hint": " · ".join(
                p for p in (g.symbol, "/".join(g.exchanges), g.sector) if p
            ),
            "value": g.security_id,
        }
        for g in groups
    ]
