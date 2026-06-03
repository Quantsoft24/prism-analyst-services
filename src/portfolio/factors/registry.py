"""Schema-derived factor registry.

The catalog is **metadata-driven**: each factor declares where its inputs come
from (real ``annual_data`` variables + daily ``market_cap``), its formula, unit,
default direction/operator, and any sectors it does not apply to. The UI catalog
and the Filtering/Factor-Builder surfaces are generated from this — adding a
factor is a data edit here, not a rewrite.

This module covers the **fundamental + market** factors (valuation / quality /
size), all computable from the six whitelisted tables. Price-window factors
(momentum / liquidity / volatility) and multi-year growth factors are added in
the same registry as their compute paths land.

Decisions baked in (see DECISIONS.md / the plan):
  * PAT (net profit) = ``PBT − Provision for direct tax`` (no direct line).
  * Net worth / equity = ``Capital & Reserves``.
  * Leverage / interest-coverage / ROCE are excluded for ``Financial Services``
    (bank liabilities are deposits, not comparable debt) — excluded names yield
    ``None`` and are counted in coverage, never zero-filled.
  * Values are returned in display units (ratios in ×, returns/margins in %).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from src.portfolio import constants as C

Category = Literal["valuation", "quality", "growth", "size", "momentum", "liquidity", "volatility"]
Direction = Literal["higher_better", "lower_better"]
Operator = Literal[">", ">=", "<", "<=", "=", "between", "top_k", "bottom_k"]
DataKind = Literal["fundamental", "market", "price_window", "growth"]


@dataclass(frozen=True)
class FactorInputs:
    """What a factor's compute needs, so the engine fetches the minimum set."""

    annual_variables: tuple[str, ...] = ()
    needs_market_cap: bool = False


@dataclass(frozen=True)
class FactorDef:
    id: str
    name: str
    category: Category
    unit: str
    direction: Direction
    default_operator: Operator
    data_kind: DataKind
    inputs: FactorInputs
    # Compute from a per-security context (resolved fundamentals + market cap).
    compute: Callable[["FactorContext"], float | None]
    source_tables: tuple[str, ...]
    description: str = ""
    exclude_sectors: tuple[str, ...] = ()
    decimals: int = 2


@dataclass
class FactorContext:
    """Per-security inputs handed to a factor's ``compute``."""

    funds: dict[str, float]          # resolved annual variables (lagged, one basis)
    market_cap: float | None         # ₹ crore, point-in-time
    close: float | None = None
    sector: str | None = None
    # Price-window / growth stats are attached by their compute paths later.
    extra: dict[str, float] = field(default_factory=dict)

    def v(self, variable: str) -> float | None:
        return self.funds.get(variable)

    @property
    def pat(self) -> float | None:
        """Net profit = PBT − direct tax (None if either input is missing)."""
        pbt = self.funds.get(C.V_PBT)
        tax = self.funds.get(C.V_DIRECT_TAX)
        if pbt is None or tax is None:
            return None
        return pbt - tax


def _div(num: float | None, den: float | None) -> float | None:
    """Safe ratio: None if either side is missing or the denominator is ~0."""
    if num is None or den is None or den == 0:
        return None
    return num / den


# ── Compute callables (display units) ────────────────────────────────────────

def _pe(c: FactorContext) -> float | None:
    return _div(c.market_cap, c.pat)                       # ×

def _earnings_yield(c: FactorContext) -> float | None:
    r = _div(c.pat, c.market_cap)
    return r * 100 if r is not None else None              # %

def _pb(c: FactorContext) -> float | None:
    return _div(c.market_cap, c.v(C.V_EQUITY))             # ×

def _ps(c: FactorContext) -> float | None:
    return _div(c.market_cap, c.v(C.V_REVENUE))            # ×

def _roe(c: FactorContext) -> float | None:
    r = _div(c.pat, c.v(C.V_EQUITY))
    return r * 100 if r is not None else None              # %

def _roce(c: FactorContext) -> float | None:
    equity = c.v(C.V_EQUITY)
    ltd = c.v(C.V_LT_BORROWINGS) or 0.0
    cap_employed = (equity + ltd) if equity is not None else None
    r = _div(c.v(C.V_PBIT), cap_employed)
    return r * 100 if r is not None else None              # %

def _net_margin(c: FactorContext) -> float | None:
    r = _div(c.pat, c.v(C.V_REVENUE))
    return r * 100 if r is not None else None              # %

def _op_margin(c: FactorContext) -> float | None:
    r = _div(c.v(C.V_PBIT), c.v(C.V_REVENUE))
    return r * 100 if r is not None else None              # %

def _debt_equity(c: FactorContext) -> float | None:
    ltd = c.v(C.V_LT_BORROWINGS)
    std = c.v(C.V_ST_BORROWINGS)
    if ltd is None and std is None:
        return None
    debt = (ltd or 0.0) + (std or 0.0)
    return _div(debt, c.v(C.V_EQUITY))                     # ×

def _interest_coverage(c: FactorContext) -> float | None:
    return _div(c.v(C.V_PBIT), c.v(C.V_INTEREST))          # ×

def _market_cap(c: FactorContext) -> float | None:
    return c.market_cap                                    # ₹ crore


def _extra(key: str) -> Callable[[FactorContext], float | None]:
    """Factor value precomputed by the engine (price-window / growth) and stashed
    on the context under ``key``."""

    def fn(c: FactorContext) -> float | None:
        return c.extra.get(key)

    return fn


_MS = ("master_securities", "prices_and_securities")
_AD = ("annual_data",)
_AD_MKT = ("annual_data", "prices_and_securities")
_FIN = (C.SECTOR_FINANCIALS,)

# ── The registry ─────────────────────────────────────────────────────────────
_FACTORS: list[FactorDef] = [
    # Valuation
    FactorDef("pe", "P/E", "valuation", "×", "lower_better", "<", "fundamental",
              FactorInputs((C.V_PBT, C.V_DIRECT_TAX), True), _pe, _AD_MKT,
              "Market cap ÷ net profit (PBT − direct tax)."),
    FactorDef("earnings_yield", "Earnings Yield", "valuation", "%", "higher_better", ">=",
              "fundamental", FactorInputs((C.V_PBT, C.V_DIRECT_TAX), True), _earnings_yield,
              _AD_MKT, "Net profit ÷ market cap."),
    FactorDef("pb", "P/B", "valuation", "×", "lower_better", "<", "fundamental",
              FactorInputs((C.V_EQUITY,), True), _pb, _AD_MKT,
              "Market cap ÷ net worth (Capital & Reserves)."),
    FactorDef("ps", "P/S", "valuation", "×", "lower_better", "<", "fundamental",
              FactorInputs((C.V_REVENUE,), True), _ps, _AD_MKT, "Market cap ÷ revenue."),
    # Quality
    FactorDef("roe", "ROE", "quality", "%", "higher_better", ">=", "fundamental",
              FactorInputs((C.V_PBT, C.V_DIRECT_TAX, C.V_EQUITY), False), _roe, _AD,
              "Net profit ÷ net worth."),
    FactorDef("roce", "ROCE", "quality", "%", "higher_better", ">=", "fundamental",
              FactorInputs((C.V_PBIT, C.V_EQUITY, C.V_LT_BORROWINGS), False), _roce, _AD,
              "PBIT ÷ capital employed (equity + long-term borrowings).",
              exclude_sectors=_FIN),
    FactorDef("net_margin", "Net Margin", "quality", "%", "higher_better", ">=", "fundamental",
              FactorInputs((C.V_PBT, C.V_DIRECT_TAX, C.V_REVENUE), False), _net_margin, _AD,
              "Net profit ÷ revenue."),
    FactorDef("op_margin", "Operating Margin", "quality", "%", "higher_better", ">=",
              "fundamental", FactorInputs((C.V_PBIT, C.V_REVENUE), False), _op_margin, _AD,
              "PBIT ÷ revenue."),
    FactorDef("debt_equity", "Debt / Equity", "quality", "×", "lower_better", "<=",
              "fundamental", FactorInputs((C.V_LT_BORROWINGS, C.V_ST_BORROWINGS, C.V_EQUITY), False),
              _debt_equity, _AD, "Total borrowings ÷ net worth.", exclude_sectors=_FIN),
    FactorDef("interest_coverage", "Interest Coverage", "quality", "×", "higher_better", ">=",
              "fundamental", FactorInputs((C.V_PBIT, C.V_INTEREST), False), _interest_coverage,
              _AD, "PBIT ÷ interest expense.", exclude_sectors=_FIN),
    # Size
    FactorDef("market_cap", "Market Cap", "size", C.UNIT_CRORE, "higher_better", "top_k",
              "market", FactorInputs((), True), _market_cap, _MS,
              "Point-in-time market capitalisation.", decimals=0),
    # Momentum (price-window — adjusted close-to-close returns)
    FactorDef("ret_3m", "3M Return", "momentum", "%", "higher_better", "top_k",
              "price_window", FactorInputs(), _extra("ret_3m"), ("prices_and_securities",),
              "Total return over the last ~3 months (63 sessions)."),
    FactorDef("ret_6m", "6M Return", "momentum", "%", "higher_better", "top_k",
              "price_window", FactorInputs(), _extra("ret_6m"), ("prices_and_securities",),
              "Total return over the last ~6 months (126 sessions)."),
    FactorDef("ret_12m", "12M Return", "momentum", "%", "higher_better", "top_k",
              "price_window", FactorInputs(), _extra("ret_12m"), ("prices_and_securities",),
              "Total return over the last ~12 months (252 sessions)."),
    # Volatility (price-window)
    FactorDef("volatility", "Volatility (1Y)", "volatility", "%", "lower_better", "bottom_k",
              "price_window", FactorInputs(), _extra("volatility"), ("prices_and_securities",),
              "Annualised volatility of daily returns over ~1 year."),
    # Liquidity (price-window)
    FactorDef("adv", "Avg Daily Value (3M)", "liquidity", C.UNIT_CRORE, "higher_better",
              ">=", "price_window", FactorInputs(), _extra("adv"), ("prices_and_securities",),
              "Average daily traded value over ~3 months.", decimals=1),
    # Growth (multi-year, point-in-time annual series)
    FactorDef("rev_cagr_3y", "Revenue CAGR (3Y)", "growth", "%", "higher_better", ">=",
              "growth", FactorInputs((C.V_REVENUE,)), _extra("rev_cagr_3y"), _AD,
              "3-year CAGR of revenue (point-in-time)."),
    FactorDef("pat_cagr_3y", "PAT CAGR (3Y)", "growth", "%", "higher_better", ">=",
              "growth", FactorInputs((C.V_PBT, C.V_DIRECT_TAX)), _extra("pat_cagr_3y"), _AD,
              "3-year CAGR of net profit (PBT − tax), point-in-time."),
]

REGISTRY: dict[str, FactorDef] = {f.id: f for f in _FACTORS}


def all_factors() -> list[FactorDef]:
    return list(_FACTORS)


def get_factor(factor_id: str) -> FactorDef | None:
    return REGISTRY.get(factor_id)


def required_variables(factor_ids: list[str]) -> list[str]:
    """Union of annual variables needed to compute the given factors."""
    out: set[str] = set()
    for fid in factor_ids:
        f = REGISTRY.get(fid)
        if f:
            out.update(f.inputs.annual_variables)
    return sorted(out)


def needs_market_cap(factor_ids: list[str]) -> bool:
    return any(
        REGISTRY[fid].inputs.needs_market_cap for fid in factor_ids if fid in REGISTRY
    )
