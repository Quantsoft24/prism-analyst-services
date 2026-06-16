"""Deep-dive "explore further" suggestions.

After the agent answers, we surface up to a few compact chips that deep-link the
user into a dedicated tool interface (Business Model Canvas, Stock Dashboard,
News & Sentiment, Regulatory Lens, Portfolio Builder) so they can dig deeper —
including the cases the conversational agent *can't* serve itself (e.g. building
a portfolio or screening stocks), which become a one-click handoff instead of a
dead-end.

Design:
  * **Deterministic + rule-based** — synthesized from the turn's ``tool_trace``
    + the user's message. NO extra LLM call (zero added latency/cost, no
    hallucinated tools/links).
  * **One registry entry per capability** (``_RULES``) — adding a future tool is
    one entry here (+ one ``ACTION_ROUTES`` entry on the frontend). This is the
    single extension point.
  * **Silent by default** — a rule emits ONLY when its trigger clearly matches.
    No match → no chip. We never spam.
  * **Bounded + curated** — capped at ``settings.DEEP_DIVE_MAX_SUGGESTIONS``
    (configurable), deduped by action, priority-ranked. A rule that fired
    because a matching *tool ran* outranks one that fired only on an
    intent-keyword, so the most relevant handoffs win the limited slots.
  * **Lightweight seeding** — chips carry only the deep-link params the target
    route already supports (``ticker`` for BMC, ``security_id`` for the stock
    dashboard, ``company`` for news); never an entity we didn't resolve.

The output is attached to ``FinalAnswer.suggested_actions`` (persisted in
``result_payload`` → replays in history). The frontend maps ``action`` → a route
via its own ``ACTION_ROUTES`` map and drops unknown actions silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.config import settings
from src.schemas.chat import DeepDiveSuggestion

# ── Capability → tool signals ───────────────────────────────────────────────
# Tool names (or prefixes) whose presence in the turn means the user engaged a
# given capability and would plausibly want the full tool UI.
_BMC_PREFIX = "bmc_"
_NEWS_PREFIX = "news_"
_SEBI_PREFIX = "sebi_"
_STOCK_TOOLS = frozenset(
    {"financials_query", "stock_technicals", "stock_filings_read", "stock_filings_lookup"}
)

# ── Intent keywords (lowercased substrings) ─────────────────────────────────
# Kept precise to avoid false fires. Used as the SECONDARY trigger (lower
# priority than "a matching tool actually ran") and as the ONLY trigger for the
# portfolio capability-gap (the agent has no portfolio/screener tool).
_PORTFOLIO_KW = (
    "build a portfolio",
    "build portfolio",
    "construct a portfolio",
    "create a portfolio",
    "portfolio of",
    "screen stocks",
    "stock screener",
    "screener",
    "filter stocks",
    "stocks with",
    "stocks where",
    "stocks that have",
    "find stocks",
    "rank stocks",
)
_BIZMODEL_KW = (
    "business model",
    "revenue model",
    "how does it make money",
    "how do they make money",
    "moat",
    "value proposition",
    "unit economics",
)
_PRICE_KW = (
    "share price",
    "stock price",
    "valuation",
    "p/e",
    "pe ratio",
    "price chart",
    "technical",
    " rsi",
    "macd",
    "market cap",
)
_NEWS_KW = ("news", "sentiment", "headlines", "latest on")


@dataclass(frozen=True)
class _Company:
    """A company resolved during the turn (the entity a chip can carry)."""

    security_id: int | None
    name: str | None
    symbol: str | None
    sector: str | None


@dataclass(frozen=True)
class _Ctx:
    """Everything a rule needs to decide + build a suggestion. Computed once."""

    user_message: str  # lowercased
    tools_run: frozenset[str]
    companies: tuple[_Company, ...]  # resolved, ordered, deduped by security_id
    bmc_ticker: str | None  # ticker from a bmc_* tool's args, if any

    @property
    def primary(self) -> _Company | None:
        return self.companies[0] if self.companies else None

    def ran_prefix(self, prefix: str) -> bool:
        return any(t.startswith(prefix) for t in self.tools_run)

    def ran_any(self, names: frozenset[str]) -> bool:
        return bool(self.tools_run & names)

    def has_kw(self, kws: tuple[str, ...]) -> bool:
        return any(k in self.user_message for k in kws)


# ── Context extraction from the tool trace ──────────────────────────────────


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _coerce_security_id(value: Any) -> int | None:
    if isinstance(value, bool):  # bool is an int subclass — reject
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _company_from_payload(payload: dict) -> _Company | None:
    """Build a _Company from a resolve_company(-ies) payload entry, if it carries
    at least a usable identifier (security_id, name, or symbol)."""
    sid = _coerce_security_id(payload.get("security_id"))
    name = payload.get("name") if isinstance(payload.get("name"), str) else None
    symbol = payload.get("symbol") if isinstance(payload.get("symbol"), str) else None
    sector = payload.get("sector") if isinstance(payload.get("sector"), str) else None
    if sid is None and not name and not symbol:
        return None
    return _Company(security_id=sid, name=name, symbol=symbol, sector=sector)


def _resolved_companies(tool_trace: list[dict[str, Any]]) -> tuple[_Company, ...]:
    """Extract companies resolved this turn, in call order, deduped by
    security_id (falling back to name when no id). Reads the trimmed ``response``
    attached to resolve_company / resolve_companies trace entries."""
    out: list[_Company] = []
    seen: set[Any] = set()

    def _add(company: _Company | None) -> None:
        if company is None:
            return
        key = company.security_id if company.security_id is not None else company.name
        if key is None or key in seen:
            return
        seen.add(key)
        out.append(company)

    for entry in tool_trace:
        if not isinstance(entry, dict):
            continue
        tool = entry.get("tool")
        resp = _as_dict(entry.get("response"))
        if not resp:
            continue
        if tool == "resolve_company" and resp.get("found"):
            _add(_company_from_payload(resp))
        elif tool == "resolve_companies":
            for item in resp.get("resolved") or []:
                _add(_company_from_payload(_as_dict(item)))
    return tuple(out)


def _bmc_ticker_from_trace(tool_trace: list[dict[str, Any]]) -> str | None:
    """The ticker a bmc_* tool was called with this turn (mirrors the legacy BMC
    handoff card). Uppercased; None if no bmc tool carried one."""
    for entry in tool_trace:
        if not isinstance(entry, dict):
            continue
        tool = entry.get("tool") or ""
        if tool.startswith(_BMC_PREFIX):
            ticker = _as_dict(entry.get("args")).get("ticker")
            if isinstance(ticker, str) and ticker.strip():
                return ticker.strip().upper()
    return None


def _build_ctx(user_message: str, tool_trace: list[dict[str, Any]]) -> _Ctx:
    trace = [e for e in (tool_trace or []) if isinstance(e, dict)]
    tools_run = frozenset(
        str(e.get("tool")) for e in trace if isinstance(e.get("tool"), str)
    )
    return _Ctx(
        user_message=(user_message or "").lower(),
        tools_run=tools_run,
        companies=_resolved_companies(trace),
        bmc_ticker=_bmc_ticker_from_trace(trace),
    )


def _label(base: str, company: _Company | None) -> str:
    """Short chip label, scoped to the company when we know it."""
    if company and company.name:
        return f"{base} · {company.name}"
    return base


# ── Rules (the registry — one entry per capability) ─────────────────────────
# Each rule returns ``(priority, DeepDiveSuggestion)`` when it should fire, else
# ``None`` (stay silent). Higher priority wins the limited slots. Rules that fire
# because a matching TOOL ran outrank intent-keyword-only fires.
Rule = Callable[[_Ctx], "tuple[int, DeepDiveSuggestion] | None"]


def _rule_bmc(ctx: _Ctx) -> "tuple[int, DeepDiveSuggestion] | None":
    ran = ctx.ran_prefix(_BMC_PREFIX)
    intent = ctx.has_kw(_BIZMODEL_KW) and ctx.primary is not None
    if not (ran or intent):
        return None
    ticker = ctx.bmc_ticker or (ctx.primary.symbol if ctx.primary else None)
    context = {"ticker": ticker} if ticker else {}
    return (
        90 if ran else 55,
        DeepDiveSuggestion(
            action="bmc",
            label=_label("Business Model Canvas", ctx.primary),
            context=context,
        ),
    )


def _rule_stock_dashboard(ctx: _Ctx) -> "tuple[int, DeepDiveSuggestion] | None":
    ran = ctx.ran_any(_STOCK_TOOLS)
    intent = ctx.has_kw(_PRICE_KW) and ctx.primary is not None
    if not (ran or intent):
        return None
    sid = ctx.primary.security_id if ctx.primary else None
    # Don't surface a price/financials handoff with no company to land on.
    if sid is None:
        return None
    return (
        80 if ran else 45,
        DeepDiveSuggestion(
            action="stock_dashboard",
            label=_label("Price & financials", ctx.primary),
            context={"security_id": sid},
        ),
    )


def _rule_news(ctx: _Ctx) -> "tuple[int, DeepDiveSuggestion] | None":
    ran = ctx.ran_prefix(_NEWS_PREFIX)
    intent = ctx.has_kw(_NEWS_KW) and ctx.primary is not None
    if not (ran or intent):
        return None
    name = ctx.primary.name if ctx.primary else None
    context = {"company": name} if name else {}
    return (
        70 if ran else 40,
        DeepDiveSuggestion(
            action="news",
            label=_label("News & sentiment", ctx.primary),
            context=context,
        ),
    )


def _rule_regulatory(ctx: _Ctx) -> "tuple[int, DeepDiveSuggestion] | None":
    if not ctx.ran_prefix(_SEBI_PREFIX):
        return None
    return (
        60,
        DeepDiveSuggestion(action="regulatory", label="Regulatory Lens", context={}),
    )


def _rule_portfolio(ctx: _Ctx) -> "tuple[int, DeepDiveSuggestion] | None":
    # Capability gap: the agent has no portfolio/screener tool, so a portfolio /
    # screening ask is always a dead-end we convert into a tool handoff.
    if not ctx.has_kw(_PORTFOLIO_KW):
        return None
    return (
        100,
        DeepDiveSuggestion(
            action="portfolio", label="Open Portfolio Builder", context={}
        ),
    )


_RULES: tuple[Rule, ...] = (
    _rule_portfolio,
    _rule_bmc,
    _rule_stock_dashboard,
    _rule_news,
    _rule_regulatory,
)


# ── Public entrypoint ───────────────────────────────────────────────────────


def synthesize(
    user_message: str, tool_trace: list[dict[str, Any]] | None
) -> list[DeepDiveSuggestion]:
    """Curate deep-dive suggestions for one answered turn.

    Returns a priority-ranked, deduped-by-action list capped at
    ``settings.DEEP_DIVE_MAX_SUGGESTIONS``. Empty (silent) when nothing clearly
    relevant matches. Never raises — a synthesis failure must never break the
    answer path (suggestions are a non-critical enhancement).
    """
    try:
        ctx = _build_ctx(user_message, tool_trace or [])
        scored: list[tuple[int, DeepDiveSuggestion]] = []
        for rule in _RULES:
            hit = rule(ctx)
            if hit is not None:
                scored.append(hit)

        # Dedup by action, keeping the highest-priority hit for each.
        best: dict[str, tuple[int, DeepDiveSuggestion]] = {}
        for priority, sug in scored:
            current = best.get(sug.action)
            if current is None or priority > current[0]:
                best[sug.action] = (priority, sug)

        ranked = sorted(best.values(), key=lambda ps: ps[0], reverse=True)
        cap = max(0, int(getattr(settings, "DEEP_DIVE_MAX_SUGGESTIONS", 3)))
        return [sug for _, sug in ranked[:cap]]
    except Exception:  # noqa: BLE001 — suggestions must never break the answer
        import logging

        logging.getLogger(__name__).exception("deep_dive.synthesize failed; no suggestions")
        return []
