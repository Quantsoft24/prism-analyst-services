"""Point-in-time factor computation for a universe as of a date.

Pulls the minimum inputs per factor family — lagged fundamentals + market
snapshot (valuation/quality/size), recent price history (momentum/volatility/
liquidity), and the multi-year lagged annual series (growth) — evaluates each
requested factor per security, and reports honest coverage ("computable for N of
M"). Missing inputs / sector-excluded names yield ``None`` — never a silent zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from src.portfolio import constants as C
from src.portfolio.constants import DEFAULT_BASIS, Basis
from src.portfolio.factors import price_stats as ps
from src.portfolio.factors.custom import CustomFactorDef, normalize
from src.portfolio.factors.expression import evaluate
from src.portfolio.factors.registry import (
    REGISTRY,
    FactorContext,
    needs_market_cap,
    required_variables,
)
from src.portfolio.repository import PortfolioRepository


@dataclass
class FactorCoverage:
    factor_id: str
    computable: int
    total: int


@dataclass
class FactorMatrix:
    """Factor values for a universe as of a date."""

    as_of: date
    security_ids: list[int]
    factor_ids: list[str]
    values: dict[int, dict[str, float | None]]   # security_id -> factor_id -> value
    basis: dict[int, Basis]                       # resolved basis per security
    sectors: dict[int, str | None]
    coverage: list[FactorCoverage] = field(default_factory=list)

    def value(self, security_id: int, factor_id: str) -> float | None:
        return self.values.get(security_id, {}).get(factor_id)


def _growth_from_series(
    series: dict[str, list[tuple[str, float]]],
) -> dict[str, float | None]:
    """Compute the growth factors from one security's lagged annual series."""
    out: dict[str, float | None] = {}
    out["rev_cagr_3y"] = ps.cagr_pct(series.get(C.V_REVENUE, []), years=3)
    # PAT per period = PBT − direct tax, aligned by fiscal period.
    pbt = dict(series.get(C.V_PBT, []))
    tax = dict(series.get(C.V_DIRECT_TAX, []))
    pat_periods = sorted(set(pbt) & set(tax))
    pat_series = [(p, pbt[p] - tax[p]) for p in pat_periods]
    out["pat_cagr_3y"] = ps.cagr_pct(pat_series, years=3)
    return out


async def compute_factors(
    repo: PortfolioRepository,
    security_ids: list[int],
    as_of: date,
    factor_ids: list[str],
    *,
    basis: Basis = DEFAULT_BASIS,
    custom: list[CustomFactorDef] | None = None,
) -> FactorMatrix:
    """Compute ``factor_ids`` (registry) + any ``custom`` factors for
    ``security_ids`` as of ``as_of`` (point-in-time).

    Registry ids requested in ``factor_ids`` AND the base ids referenced by the
    custom expressions are computed; custom values are then evaluated per
    security and normalized cross-sectionally. The output matrix exposes the
    requested registry ids + the custom ids.
    """
    custom = custom or []
    requested = [fid for fid in factor_ids if fid in REGISTRY]
    custom_refs = {r for c in custom for r in c.refs if r in REGISTRY}
    # Base registry ids actually computed (requested + custom dependencies).
    base_ids = sorted(set(requested) | custom_refs)
    out_ids = requested + [c.id for c in custom]

    if not security_ids or not base_ids:
        empty = FactorMatrix(as_of, security_ids, out_ids, {}, {}, {})
        return empty

    kinds = {fid: REGISTRY[fid].data_kind for fid in base_ids}
    fundamental_ids = [f for f in base_ids if kinds[f] in ("fundamental", "market")]
    price_ids = [f for f in base_ids if kinds[f] == "price_window"]
    growth_ids = [f for f in base_ids if kinds[f] == "growth"]

    # Fundamentals snapshot (point-value) + market cap, for valuation/quality/size.
    snap_vars = required_variables(fundamental_ids)
    funds = (
        await repo.fundamentals_snapshot(security_ids, as_of, snap_vars, prefer_basis=basis)
        if snap_vars
        else {}
    )
    market = (
        await repo.market_snapshot(security_ids, as_of)
        if needs_market_cap(fundamental_ids)
        else {}
    )
    sectors = await repo.sectors(security_ids)

    # Price-window stats (momentum / volatility / liquidity).
    price_extra: dict[int, dict[str, float | None]] = {}
    if price_ids:
        hist = await repo.price_history(security_ids, as_of)
        for sid, series in hist.items():
            price_extra[sid] = {
                "ret_3m": ps.trailing_return_pct(series, ps.TD_3M),
                "ret_6m": ps.trailing_return_pct(series, ps.TD_6M),
                "ret_12m": ps.trailing_return_pct(series, ps.TD_12M),
                "volatility": ps.annualized_vol_pct(series),
                "adv": ps.adv_crore(series),
            }

    # Growth (multi-year lagged series).
    growth_extra: dict[int, dict[str, float | None]] = {}
    if growth_ids:
        gvars = required_variables(growth_ids)
        series_by_sec = await repo.fundamentals_series(
            security_ids, as_of, gvars, prefer_basis=basis
        )
        for sid, (_b, series) in series_by_sec.items():
            growth_extra[sid] = _growth_from_series(series)

    base_values: dict[int, dict[str, float | None]] = {}
    resolved_basis: dict[int, Basis] = {}

    for sid in security_ids:
        sec_basis, sec_funds = funds.get(sid, (basis, {}))
        resolved_basis[sid] = sec_basis
        snap = market.get(sid)
        extra: dict[str, float] = {}
        extra.update({k: v for k, v in price_extra.get(sid, {}).items() if v is not None})
        extra.update({k: v for k, v in growth_extra.get(sid, {}).items() if v is not None})
        ctx = FactorContext(
            funds=sec_funds,
            market_cap=snap.market_cap if snap else None,
            close=snap.close if snap else None,
            sector=sectors.get(sid),
            extra=extra,
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

    # Custom factors: evaluate the expression per security, then normalize
    # cross-sectionally over the universe.
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

    # Assemble the output matrix (requested registry ids + custom ids).
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
    return FactorMatrix(
        as_of=as_of,
        security_ids=security_ids,
        factor_ids=out_ids,
        values=values,
        basis=resolved_basis,
        sectors=sectors,
        coverage=coverage,
    )
