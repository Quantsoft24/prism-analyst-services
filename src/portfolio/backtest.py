"""Point-in-time backtest engine (vectorized NAV via numpy).

For each rebalance date on the real trading calendar it rebuilds the eligible
universe (dated index membership), recomputes lagged factors, screens, and
weights — exactly the live Suggested-Portfolio path, replayed through history
with no look-ahead. Holdings are then propagated forward on corporate-action-
adjusted closes to build the NAV; the benchmark NAV is the cumulative index
daily return. No survivorship bias: each rebalance uses the constituents and
fundamentals knowable as of that date.

Designed to scale to the full Nifty 500 over full history; the per-rebalance
screen reuses the cached factor pipeline and the return math is a single numpy
price-matrix pass. Pure compute given a repo — the durable job/worker wraps it.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from src.portfolio.backtest_data import factor_matrix_inmem, preload
from src.portfolio.calendar import Frequency, generate_rebalance_dates
from src.portfolio.constants import DEFAULT_BASIS, Basis
from src.portfolio.factors import registry
from src.portfolio.factors.compute import FactorMatrix
from src.portfolio.factors.custom import CustomFactorDef
from src.portfolio.metrics import PerfMetrics, compute_metrics, drawdown_series
from src.portfolio.repository import PortfolioRepository
from src.portfolio.screening import Filter, apply_filters
from src.portfolio.weighting import Scheme, compute_weights

ProgressCb = Callable[[float, str], Awaitable[None]]


@dataclass
class WeightingConfig:
    scheme: Scheme = "equal"
    score_factor_id: str | None = None
    max_weight: float | None = None
    max_sector_weight: float | None = None


@dataclass
class BacktestSpec:
    index_id: int
    start: date
    end: date
    frequency: Frequency = "quarterly"
    filters: list[Filter] = field(default_factory=list)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    basis: Basis = DEFAULT_BASIS
    benchmark_index_id: int | None = None     # default = index_id
    custom: list[CustomFactorDef] = field(default_factory=list)


@dataclass
class HoldingSnap:
    security_id: int
    symbol: str | None
    sector: str | None
    weight: float
    is_new: bool


@dataclass
class RebalanceSnap:
    date: date
    n_holdings: int
    turnover: float
    holdings: list[HoldingSnap]


@dataclass
class SectorActive:
    sector: str
    portfolio: float       # portfolio weight in this sector
    benchmark: float       # benchmark (cap-weighted universe) weight
    active: float          # portfolio − benchmark


@dataclass
class FactorTilt:
    factor_id: str
    exposure: float        # portfolio weighted-avg z-score vs the universe


@dataclass
class Contributor:
    security_id: int
    symbol: str | None
    contribution: float    # summed weight × period-return across the backtest


# Style "lenses" always reported in attribution (independent of the strategy's
# own filters): one representative factor per classic equity style, with a sign
# so a positive exposure always means "more of this style".
STYLE_FACTORS: list[tuple[str, str, float]] = [
    ("Value", "earnings_yield", 1.0),     # higher earnings yield = cheaper = value
    ("Quality", "roe", 1.0),              # higher ROE = higher quality
    ("Growth", "rev_cagr_3y", 1.0),       # higher revenue growth = growthier
    ("Momentum", "ret_12m", 1.0),         # higher 12m return = more momentum
    ("Size", "market_cap", 1.0),          # higher market cap = larger-cap tilt
    ("Low Volatility", "volatility", -1.0),  # lower vol = more low-vol tilt
]
STYLE_IDS: list[str] = [fid for _, fid, _ in STYLE_FACTORS]


@dataclass
class Attribution:
    as_of: date
    sector_active: list[SectorActive]
    factor_tilts: list[FactorTilt]
    top_contributors: list[Contributor]
    bottom_contributors: list[Contributor]
    style_tilts: list[FactorTilt] = field(default_factory=list)


@dataclass
class BacktestResult:
    dates: list[date]
    nav: list[float]
    benchmark_nav: list[float]
    drawdown: list[float]
    metrics: PerfMetrics
    benchmark_metrics: PerfMetrics
    rebalances: list[RebalanceSnap]
    attribution: Attribution | None = None
    notes: list[str] = field(default_factory=list)


def _needed_factors(spec: BacktestSpec) -> list[str]:
    valid = set(registry.REGISTRY)
    needed = {f.factor_id for f in spec.filters if f.factor_id in valid} | {"market_cap"}
    w = spec.weighting
    if w.scheme == "factor_score" and w.score_factor_id in valid:
        needed.add(w.score_factor_id)
    if w.scheme == "inverse_vol":
        needed.add("volatility")
    return sorted(needed)


def _build_panel(
    closes: dict[int, list[tuple[date, float | None]]],
    held: list[int],
    master_dates: list[date],
) -> np.ndarray:
    """[T, N] forward-filled adjusted-close matrix aligned to ``master_dates``."""
    t, n = len(master_dates), len(held)
    panel = np.full((t, n), np.nan)
    date_idx = {d: i for i, d in enumerate(master_dates)}
    for j, sid in enumerate(held):
        for d, c in closes.get(sid, []):
            i = date_idx.get(d)
            if i is not None and c is not None:
                panel[i, j] = c
        col = panel[:, j]
        # Forward-fill (leading NaNs — pre-listing — stay NaN and are excluded).
        filled = np.where(~np.isnan(col), np.arange(t), 0)
        np.maximum.accumulate(filled, out=filled)
        had = ~np.isnan(col[filled])
        col2 = np.where(had, col[filled], np.nan)
        panel[:, j] = col2
    return panel


def _turnover(prev: dict[int, float], cur: dict[int, float]) -> float:
    keys = set(prev) | set(cur)
    return 0.5 * sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)


def _ffill(a: np.ndarray) -> np.ndarray:
    """Forward-fill non-finite values in a 1-D equity curve (carry the last
    finite NAV over a gap); leading non-finite become the first finite value."""
    a = np.asarray(a, dtype=float)
    finite = np.isfinite(a)
    if not finite.any():
        return np.ones_like(a)
    idx = np.where(finite, np.arange(a.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    first = int(np.argmax(finite))
    out[:first] = a[first]
    return out


async def run_backtest(
    repo: PortfolioRepository,
    spec: BacktestSpec,
    progress: ProgressCb | None = None,
) -> BacktestResult:
    notes: list[str] = []
    bench_id = spec.benchmark_index_id or spec.index_id

    bench = await repo.benchmark_series(bench_id, spec.start, spec.end)
    master_dates = [d for d, _ in bench]
    if len(master_dates) < 2:
        raise ValueError("No benchmark trading-day history in the selected window.")

    rebal_dates = generate_rebalance_dates(master_dates, spec.frequency, spec.start, spec.end)
    if not rebal_dates:
        raise ValueError("No rebalance dates fall within the selected window.")

    factor_ids = _needed_factors(spec)
    w = spec.weighting
    custom_dir = {c.id: c.direction for c in spec.custom}
    score_dir = (
        registry.REGISTRY[w.score_factor_id].direction
        if w.scheme == "factor_score" and w.score_factor_id in registry.REGISTRY
        else custom_dir.get(w.score_factor_id, "higher_better")
    )

    # 0) Bulk preload — a few queries serve the whole run (no per-rebalance I/O).
    #    Style-lens factors are preloaded too (variables only) so attribution can
    #    report style exposure at the last rebalance without extra I/O; the
    #    per-rebalance screen still computes only the strategy's own `factor_ids`.
    if progress:
        await progress(0.05, "Loading data")
    preload_ids = sorted(set(factor_ids) | set(STYLE_IDS))
    data = await preload(repo, spec.index_id, rebal_dates, spec.end, preload_ids, spec.custom)

    # 1) Screen + weight at each rebalance, in memory (point-in-time).
    holdings_by_date: dict[date, dict[int, float]] = {}
    last_fm: FactorMatrix | None = None
    last_members: list[int] = []
    for k, d in enumerate(rebal_dates):
        members = data.members_asof(d)
        if not members:
            holdings_by_date[d] = {}
            continue
        fm = factor_matrix_inmem(data, members, d, factor_ids, basis=spec.basis, custom=spec.custom)
        last_fm, last_members = fm, members
        survivors = apply_filters(fm, spec.filters).survivors
        market_cap = {s: fm.value(s, "market_cap") for s in survivors}
        score = (
            {s: fm.value(s, w.score_factor_id) for s in survivors}
            if w.scheme == "factor_score" and w.score_factor_id
            else None
        )
        vol = {s: fm.value(s, "volatility") for s in survivors} if w.scheme == "inverse_vol" else None
        holdings_by_date[d] = compute_weights(
            survivors, scheme=w.scheme, market_cap=market_cap, score=score,
            score_direction=score_dir, volatility=vol, sector=fm.sectors,
            max_weight=w.max_weight, max_sector_weight=w.max_sector_weight,
        )
        if progress and (k % 4 == 0 or k == len(rebal_dates) - 1):
            await progress(0.1 + 0.7 * (k + 1) / len(rebal_dates), f"Rebalance {k + 1}/{len(rebal_dates)}")

    all_held = sorted({s for h in holdings_by_date.values() for s in h})
    if not all_held:
        raise ValueError("No securities passed the filters at any rebalance date.")

    # 2) Price panel (from the preloaded prices) + forward NAV.
    start_idx = master_dates.index(rebal_dates[0])
    closes = {sid: [(pd, c) for pd, c, _mc, _tv in data.prices.get(sid, [])] for sid in all_held}
    if progress:
        await progress(0.85, "Computing NAV")
    panel = _build_panel(closes, all_held, master_dates)
    col_of = {sid: j for j, sid in enumerate(all_held)}

    nav = np.full(len(master_dates), np.nan)
    nav[start_idx] = 1.0
    bounds = [master_dates.index(d) for d in rebal_dates] + [len(master_dates) - 1]
    for p in range(len(rebal_dates)):
        s, e = bounds[p], bounds[p + 1]
        if e <= s:
            continue
        held = holdings_by_date[rebal_dates[p]]
        w_vec = np.zeros(len(all_held))
        for sid, wt in held.items():
            w_vec[col_of[sid]] = wt
        base = panel[s, :]
        valid = (~np.isnan(base)) & (base > 0) & (w_vec > 0)
        if not valid.any():
            nav[s : e + 1] = nav[s]
            continue
        wv = np.where(valid, w_vec, 0.0)
        wv = wv / wv.sum()
        growth = panel[s : e + 1, :] / np.where(valid, base, 1.0)
        growth = np.where(np.isnan(growth), 1.0, growth)
        port_factor = growth @ wv               # [period_len]
        nav[s : e + 1] = nav[s] * port_factor

    dates = master_dates[start_idx:]
    nav_arr = _ffill(nav[start_idx:])    # carry NAV over any residual price gap

    # 3) Benchmark NAV over the same axis (cumulative daily return).
    bench_ret = np.array([r if r is not None else 0.0 for _, r in bench], dtype=float)[start_idx:]
    bench_nav = np.empty(len(bench_ret))
    bench_nav[0] = 1.0
    if len(bench_ret) > 1:
        bench_nav[1:] = np.cumprod(1.0 + bench_ret[1:])

    # 4) Metrics + drawdown.
    metrics = compute_metrics(nav_arr, dates)
    bench_metrics = compute_metrics(bench_nav, dates)
    dd = drawdown_series(nav_arr)

    # 5) Holdings-by-date (with new-vs-prior + turnover).
    meta = await repo.securities_meta(all_held)
    rebalances: list[RebalanceSnap] = []
    prev: dict[int, float] = {}
    for d in rebal_dates:
        held = holdings_by_date[d]
        snaps = sorted(
            (
                HoldingSnap(
                    security_id=s,
                    symbol=meta.get(s, (None, None, None))[0],
                    sector=meta.get(s, (None, None, None))[2],
                    weight=wt, is_new=s not in prev,
                )
                for s, wt in held.items()
            ),
            key=lambda h: h.weight, reverse=True,
        )
        rebalances.append(
            RebalanceSnap(date=d, n_holdings=len(held), turnover=_turnover(prev, held), holdings=snaps)
        )
        prev = held

    # 6) Attribution (benchmark-relative): sector active weights, factor tilts,
    #    and the top/bottom return contributors.
    attribution = _attribution(
        last_fm, last_members, holdings_by_date, factor_ids, panel, bounds, col_of, rebal_dates, meta,
        data=data, basis=spec.basis, custom=spec.custom,
    )

    if progress:
        await progress(1.0, "Done")
    return BacktestResult(
        dates=dates,
        nav=[float(x) for x in nav_arr],
        benchmark_nav=[float(x) for x in bench_nav],
        drawdown=[float(x) for x in dd],
        metrics=metrics,
        benchmark_metrics=bench_metrics,
        rebalances=rebalances,
        attribution=attribution,
        notes=notes,
    )


def _weighted_tilt(
    fm: FactorMatrix, members: list[int], port: dict[int, float], fid: str
) -> float | None:
    """Portfolio weight-weighted z-score of factor ``fid`` vs the universe.
    Positive = portfolio tilts toward higher values of the factor."""
    uni = [fm.value(s, fid) for s in members if fm.value(s, fid) is not None]
    if len(uni) < 3:
        return None
    arr = np.array(uni, dtype=float)
    mean, std = float(arr.mean()), float(arr.std())
    if std == 0:
        return None
    zsum, wsum = 0.0, 0.0
    for s, wt in port.items():
        v = fm.value(s, fid)
        if v is not None:
            zsum += wt * (v - mean) / std
            wsum += wt
    return zsum / wsum if wsum > 0 else None


def _attribution(
    last_fm: FactorMatrix | None,
    last_members: list[int],
    holdings_by_date: dict[date, dict[int, float]],
    factor_ids: list[str],
    panel: np.ndarray,
    bounds: list[int],
    col_of: dict[int, int],
    rebal_dates: list[date],
    meta: dict[int, tuple[str | None, str | None, str | None]],
    *,
    data=None,
    basis: Basis = DEFAULT_BASIS,
    custom: list[CustomFactorDef] | None = None,
) -> Attribution | None:
    if last_fm is None:
        return None
    port = holdings_by_date.get(last_fm.as_of, {})
    if not port:
        return None

    # Sector active weights: portfolio vs the cap-weighted universe (benchmark).
    mcap = {s: last_fm.value(s, "market_cap") for s in last_members}
    total_mc = sum(v for v in mcap.values() if v) or 1.0
    bench_sec: dict[str, float] = {}
    for s in last_members:
        mc = mcap.get(s)
        if mc:
            sec = last_fm.sectors.get(s) or "Unclassified"
            bench_sec[sec] = bench_sec.get(sec, 0.0) + mc / total_mc
    port_sec: dict[str, float] = {}
    for s, wt in port.items():
        sec = last_fm.sectors.get(s) or "Unclassified"
        port_sec[sec] = port_sec.get(sec, 0.0) + wt
    sector_active = sorted(
        (
            SectorActive(sec, port_sec.get(sec, 0.0), bench_sec.get(sec, 0.0),
                         port_sec.get(sec, 0.0) - bench_sec.get(sec, 0.0))
            for sec in set(port_sec) | set(bench_sec)
        ),
        key=lambda s: abs(s.active), reverse=True,
    )

    # Factor tilts: portfolio weight-weighted z-score vs the universe, for the
    # strategy's own factors.
    tilts: list[FactorTilt] = []
    for fid in factor_ids:
        z = _weighted_tilt(last_fm, last_members, port, fid)
        if z is not None:
            tilts.append(FactorTilt(fid, z))
    tilts.sort(key=lambda t: abs(t.exposure), reverse=True)

    # Style tilts: always-on exposure to the classic equity styles (Value /
    # Quality / Growth / Momentum / Size / Low-vol), so the user sees style
    # posture even when those factors aren't part of the filters. Computed from
    # the preloaded panel at the last rebalance (no extra I/O).
    style_tilts: list[FactorTilt] = []
    if data is not None:
        style_fm = factor_matrix_inmem(
            data, last_members, last_fm.as_of, STYLE_IDS, basis=basis, custom=[],
        )
        for label, fid, sign in STYLE_FACTORS:
            z = _weighted_tilt(style_fm, last_members, port, fid)
            if z is not None:
                style_tilts.append(FactorTilt(label, sign * z))

    # Contributors: summed weight × period return across the run.
    contrib: dict[int, float] = {}
    for p in range(len(rebal_dates)):
        s_idx, e_idx = bounds[p], bounds[p + 1]
        for sid, wt in holdings_by_date[rebal_dates[p]].items():
            j = col_of.get(sid)
            if j is None:
                continue
            base, endv = panel[s_idx, j], panel[e_idx, j]
            if base and base > 0 and np.isfinite(base) and np.isfinite(endv):
                contrib[sid] = contrib.get(sid, 0.0) + float(wt) * float(endv / base - 1.0)
    ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)

    def _contribs(pairs: list[tuple[int, float]]) -> list[Contributor]:
        return [Contributor(sid, meta.get(sid, (None, None, None))[0], val) for sid, val in pairs]

    return Attribution(
        as_of=last_fm.as_of,
        sector_active=sector_active,
        factor_tilts=tilts,
        top_contributors=_contribs(ranked[:8]),
        bottom_contributors=_contribs(list(reversed(ranked[-8:]))),
        style_tilts=style_tilts,
    )
