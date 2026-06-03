"""Batched preload + in-memory factor computation for the backtest engine.

The naive engine made ~5-6 DB round-trips **per rebalance** (membership,
fundamentals snapshot + series, market snapshot, sectors, price history) — slow
over a high-latency RDS. This module pulls everything the run needs in **a
handful of bulk queries up front**, then rebuilds the point-in-time factor matrix
for each rebalance entirely in memory. Same math as ``factors.compute`` (it
reuses the registry compute callables, price-stat helpers, the 6-month lag, and
custom-factor evaluation/normalization) — just no per-date I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.portfolio.constants import DEFAULT_BASIS, Basis
from src.portfolio.factors import price_stats as ps
from src.portfolio.factors.compute import FactorCoverage, FactorMatrix, _growth_from_series
from src.portfolio.factors.custom import CustomFactorDef, normalize
from src.portfolio.factors.expression import evaluate
from src.portfolio.factors.registry import (
    REGISTRY,
    FactorContext,
    needs_market_cap,
    required_variables,
)
from src.portfolio.lag import is_usable
from src.portfolio.repository import PortfolioRepository

# Trading-day buffer before the first rebalance so 12m momentum / 1y vol windows
# are available at the first screen (~280 sessions ≈ 400 calendar days).
_PRICE_BUFFER_DAYS = 420


@dataclass
class PreloadedData:
    snapshots: list[tuple[date, list[int]]]                                  # index membership, ascending
    sectors: dict[int, str | None]
    prices: dict[int, list[tuple[date, float | None, float | None, float | None]]]  # (date, close, mcap, tradeval)
    annual: dict[int, dict[str, dict[str, list[tuple[str, float]]]]]         # sid->basis->var->[(period,val)]

    def members_asof(self, d: date) -> list[int]:
        """Point-in-time constituents: the latest snapshot with ``date <= d``."""
        chosen: list[int] | None = None
        for sd, sids in self.snapshots:
            if sd <= d:
                chosen = sids
            else:
                break
        return chosen or []


def needed_annual_variables(factor_ids: list[str], custom: list[CustomFactorDef]) -> list[str]:
    reg = [f for f in factor_ids if f in REGISTRY]
    reg += [r for c in custom for r in c.refs if r in REGISTRY]
    return required_variables(reg)


async def preload(
    repo: PortfolioRepository,
    index_id: int,
    rebalance_dates: list[date],
    end: date,
    factor_ids: list[str],
    custom: list[CustomFactorDef],
) -> PreloadedData:
    """Bulk-load everything the backtest needs in a few queries."""
    from datetime import timedelta

    snapshots = await repo.index_snapshots(index_id)
    data = PreloadedData(snapshots=snapshots, sectors={}, prices={}, annual={})

    universe = sorted({s for d in rebalance_dates for s in data.members_asof(d)})
    if not universe:
        return data

    price_start = min(rebalance_dates) - timedelta(days=_PRICE_BUFFER_DAYS)
    data.prices = await repo.bulk_prices(universe, price_start, end)
    data.sectors = await repo.sectors(universe)
    variables = needed_annual_variables(factor_ids, custom)
    if variables:
        data.annual = await repo.bulk_annual(universe, variables, f"{end.year:04d}-{end.month:02d}")
    return data


# ── In-memory point-in-time helpers ──────────────────────────────────────────

def _resolve_snapshot(
    annual_sid: dict[str, dict[str, list[tuple[str, float]]]],
    as_of: date,
    variables: list[str],
    prefer_basis: Basis,
) -> tuple[Basis, dict[str, float]]:
    fallback: Basis = "standalone" if prefer_basis == "consolidated" else "consolidated"
    for basis in (prefer_basis, fallback):
        bdata = annual_sid.get(basis, {})
        vals: dict[str, float] = {}
        for v in variables:
            latest: float | None = None
            for period, value in bdata.get(v, []):     # ascending → keep last usable
                if is_usable(period, as_of):
                    latest = value
            if latest is not None:
                vals[v] = latest
        if vals:
            return basis, vals
    return prefer_basis, {}


def _resolve_series(
    annual_sid: dict[str, dict[str, list[tuple[str, float]]]],
    as_of: date,
    variables: list[str],
    prefer_basis: Basis,
) -> dict[str, list[tuple[str, float]]]:
    fallback: Basis = "standalone" if prefer_basis == "consolidated" else "consolidated"
    for basis in (prefer_basis, fallback):
        bdata = annual_sid.get(basis, {})
        out = {v: [(p, val) for p, val in bdata.get(v, []) if is_usable(p, as_of)] for v in variables}
        if any(out.values()):
            return out
    return {v: [] for v in variables}


def _market_at(
    prices_sid: list[tuple[date, float | None, float | None, float | None]], as_of: date
) -> tuple[float | None, float | None]:
    close, mcap = None, None
    for d, c, mc, _tv in prices_sid:
        if d <= as_of:
            close, mcap = c, mc
        else:
            break
    return close, mcap


def _price_window(
    prices_sid: list[tuple[date, float | None, float | None, float | None]], as_of: date, n: int = 280
) -> list[tuple[date, float | None, float | None]]:
    rows = [(d, c, tv) for d, c, _mc, tv in prices_sid if d <= as_of]
    return rows[-n:]


def factor_matrix_inmem(
    data: PreloadedData,
    security_ids: list[int],
    as_of: date,
    factor_ids: list[str],
    *,
    basis: Basis = DEFAULT_BASIS,
    custom: list[CustomFactorDef] | None = None,
) -> FactorMatrix:
    """Rebuild the point-in-time factor matrix from preloaded data (no I/O).
    Mirrors ``factors.compute.compute_factors``."""
    custom = custom or []
    requested = [f for f in factor_ids if f in REGISTRY]
    custom_refs = {r for c in custom for r in c.refs if r in REGISTRY}
    base_ids = sorted(set(requested) | custom_refs)
    out_ids = requested + [c.id for c in custom]
    if not security_ids or not base_ids:
        return FactorMatrix(as_of, security_ids, out_ids, {}, {}, data.sectors)

    kinds = {f: REGISTRY[f].data_kind for f in base_ids}
    fundamental_ids = [f for f in base_ids if kinds[f] in ("fundamental", "market")]
    price_ids = [f for f in base_ids if kinds[f] == "price_window"]
    growth_ids = [f for f in base_ids if kinds[f] == "growth"]
    snap_vars = required_variables(fundamental_ids)
    want_market = needs_market_cap(fundamental_ids) or bool(price_ids)
    gvars = required_variables(growth_ids)

    base_values: dict[int, dict[str, float | None]] = {}
    resolved_basis: dict[int, Basis] = {}
    for sid in security_ids:
        annual_sid = data.annual.get(sid, {})
        sec_basis, funds = (
            _resolve_snapshot(annual_sid, as_of, snap_vars, basis) if snap_vars else (basis, {})
        )
        resolved_basis[sid] = sec_basis
        prices_sid = data.prices.get(sid, [])
        close, market_cap = _market_at(prices_sid, as_of) if want_market else (None, None)
        extra: dict[str, float] = {}
        if price_ids:
            series = _price_window(prices_sid, as_of)
            for k, v in {
                "ret_3m": ps.trailing_return_pct(series, ps.TD_3M),
                "ret_6m": ps.trailing_return_pct(series, ps.TD_6M),
                "ret_12m": ps.trailing_return_pct(series, ps.TD_12M),
                "volatility": ps.annualized_vol_pct(series),
                "adv": ps.adv_crore(series),
            }.items():
                if v is not None:
                    extra[k] = v
        if growth_ids:
            for k, v in _growth_from_series(_resolve_series(annual_sid, as_of, gvars, basis)).items():
                if v is not None:
                    extra[k] = v
        ctx = FactorContext(
            funds=funds, market_cap=market_cap, close=close,
            sector=data.sectors.get(sid), extra=extra,
        )
        row: dict[str, float | None] = {}
        for fid in base_ids:
            fdef = REGISTRY[fid]
            if fdef.exclude_sectors and ctx.sector in fdef.exclude_sectors:
                row[fid] = None
                continue
            try:
                row[fid] = fdef.compute(ctx)
            except (TypeError, ValueError, ZeroDivisionError):
                row[fid] = None
        base_values[sid] = row

    custom_values: dict[str, dict[int, float | None]] = {}
    for c in custom:
        raw: dict[int, float | None] = {}
        for sid in security_ids:
            inputs = {ref: base_values[sid].get(ref) for ref in c.refs}
            try:
                raw[sid] = evaluate(c.expression, inputs)
            except (TypeError, ValueError, ZeroDivisionError):
                raw[sid] = None
        custom_values[c.id] = normalize(raw, c.normalization)

    values: dict[int, dict[str, float | None]] = {}
    for sid in security_ids:
        row = {rid: base_values[sid].get(rid) for rid in requested}
        for c in custom:
            row[c.id] = custom_values[c.id].get(sid)
        values[sid] = row

    total = len(security_ids)
    coverage = [
        FactorCoverage(fid, sum(values[s][fid] is not None for s in security_ids), total)
        for fid in out_ids
    ]
    return FactorMatrix(as_of, security_ids, out_ids, values, resolved_basis, data.sectors, coverage)
