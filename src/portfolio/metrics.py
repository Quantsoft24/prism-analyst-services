"""Performance metrics for an equity curve (numpy, no DB).

All operate on a NAV array (portfolio value, starting at 1.0) plus the matching
trading dates. Returns are simple daily; risk-free is 0 (configurable later).
Kept separate + pure so they're unit-tested against hand-computed cases and
reused by the backtest engine and the saved-result summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

TRADING_DAYS_YEAR = 252.0


@dataclass
class PerfMetrics:
    total_return: float          # fraction (0.25 = +25%)
    cagr: float
    ann_vol: float
    sharpe: float
    max_drawdown: float          # negative fraction (-0.30 = -30%)
    best_day: float
    worst_day: float
    n_days: int


def daily_returns(nav: np.ndarray) -> np.ndarray:
    """Simple daily returns, with non-finite values (from any residual price gap)
    dropped so vol/Sharpe stay well-defined."""
    nav = np.asarray(nav, dtype=float)
    if nav.size < 2:
        return np.array([])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = nav[1:] / nav[:-1] - 1.0
    return r[np.isfinite(r)]


def drawdown_series(nav: np.ndarray) -> np.ndarray:
    """Underwater curve: NAV / running-peak − 1 (≤ 0)."""
    nav = np.asarray(nav, dtype=float)
    if nav.size == 0:
        return nav
    peak = np.maximum.accumulate(nav)
    return nav / peak - 1.0


def _years_between(dates: list[date]) -> float:
    if len(dates) < 2:
        return 0.0
    return max((dates[-1] - dates[0]).days / 365.25, 1e-9)


def compute_metrics(nav: np.ndarray, dates: list[date]) -> PerfMetrics:
    nav = np.asarray(nav, dtype=float)
    if nav.size < 2:
        return PerfMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, int(nav.size))
    rets = daily_returns(nav)
    total = float(nav[-1] / nav[0] - 1.0)
    years = _years_between(dates)
    cagr = float((nav[-1] / nav[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    vol = float(np.std(rets, ddof=1) * np.sqrt(TRADING_DAYS_YEAR)) if rets.size > 1 else 0.0
    mean_ann = float(np.mean(rets) * TRADING_DAYS_YEAR) if rets.size else 0.0
    sharpe = float(mean_ann / vol) if vol > 0 else 0.0
    dd = float(drawdown_series(nav).min())
    return PerfMetrics(
        total_return=total, cagr=cagr, ann_vol=vol, sharpe=sharpe,
        max_drawdown=dd, best_day=float(rets.max()), worst_day=float(rets.min()),
        n_days=int(nav.size),
    )
