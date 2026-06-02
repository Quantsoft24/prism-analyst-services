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

from src.portfolio.calendar import Frequency, generate_rebalance_dates
from src.portfolio.constants import DEFAULT_BASIS, Basis
from src.portfolio.factors import registry
from src.portfolio.factors.compute import compute_factors
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
class BacktestResult:
    dates: list[date]
    nav: list[float]
    benchmark_nav: list[float]
    drawdown: list[float]
    metrics: PerfMetrics
    benchmark_metrics: PerfMetrics
    rebalances: list[RebalanceSnap]
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

    # 1) Screen + weight at each rebalance (the heavy, point-in-time loop).
    holdings_by_date: dict[date, dict[int, float]] = {}
    for k, d in enumerate(rebal_dates):
        members = await repo.members_as_of(spec.index_id, d)
        if not members:
            holdings_by_date[d] = {}
            continue
        fm = await compute_factors(
            repo, members, d, factor_ids, basis=spec.basis, custom=spec.custom
        )
        survivors = apply_filters(fm, spec.filters).survivors
        market_cap = {s: fm.value(s, "market_cap") for s in survivors}
        score = (
            {s: fm.value(s, w.score_factor_id) for s in survivors}
            if w.scheme == "factor_score" and w.score_factor_id
            else None
        )
        custom_dir = {c.id: c.direction for c in spec.custom}
        score_dir = (
            registry.REGISTRY[w.score_factor_id].direction
            if w.scheme == "factor_score" and w.score_factor_id in registry.REGISTRY
            else custom_dir.get(w.score_factor_id, "higher_better")
        )
        vol = {s: fm.value(s, "volatility") for s in survivors} if w.scheme == "inverse_vol" else None
        holdings_by_date[d] = compute_weights(
            survivors, scheme=w.scheme, market_cap=market_cap, score=score,
            score_direction=score_dir, volatility=vol, sector=fm.sectors,
            max_weight=w.max_weight, max_sector_weight=w.max_sector_weight,
        )
        if progress:
            await progress(0.1 + 0.7 * (k + 1) / len(rebal_dates), f"Rebalance {k + 1}/{len(rebal_dates)}")

    all_held = sorted({s for h in holdings_by_date.values() for s in h})
    if not all_held:
        raise ValueError("No securities passed the filters at any rebalance date.")

    # 2) Price panel + forward NAV.
    start_idx = master_dates.index(rebal_dates[0])
    closes = await repo.closes_panel(all_held, master_dates[start_idx], spec.end)
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
        notes=notes,
    )
