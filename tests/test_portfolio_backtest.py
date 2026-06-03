"""Unit tests for rebalance-date generation + performance metrics (pure, no DB)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from src.portfolio.calendar import generate_rebalance_dates
from src.portfolio.metrics import compute_metrics, drawdown_series


def _trading_days(start: date, end: date) -> list[date]:
    """Weekday calendar — a stand-in trading-day axis."""
    d, out = start, []
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_quarterly_rebalances_land_on_trading_days():
    td = _trading_days(date(2024, 1, 1), date(2024, 12, 31))
    reb = generate_rebalance_dates(td, "quarterly", date(2024, 1, 1), date(2024, 12, 31))
    # Cadence is anchored to the chosen start month (Jan) → Jan/Apr/Jul/Oct.
    assert 3 <= len(reb) <= 4
    assert all(r in td for r in reb)
    assert reb == sorted(set(reb))                      # ascending, unique
    assert date(2024, 1, 31) in reb                     # first = start-month end
    # Sep 30 2024 is a Monday; Jun 30 2024 is a Sunday → maps back to Fri Jun 28
    # (anchored to Jan, the quarterly steps are Jan/Apr/Jul/Oct, not calendar Qs).
    months = sorted({r.month for r in reb})
    assert months == [1, 4, 7, 10]


def test_monthly_count_and_15d():
    td = _trading_days(date(2024, 1, 1), date(2024, 6, 30))
    assert 5 <= len(generate_rebalance_dates(td, "monthly", date(2024, 1, 1), date(2024, 6, 30))) <= 6
    fifteen = generate_rebalance_dates(td, "15d", date(2024, 1, 1), date(2024, 3, 31))
    assert len(fifteen) >= 5


def test_metrics_on_known_curve():
    # Doubling over exactly 1 year (252 sessions) → ~100% total, CAGR ~100%.
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(253)]
    nav = np.linspace(1.0, 2.0, 253)
    m = compute_metrics(nav, dates)
    assert abs(m.total_return - 1.0) < 1e-9
    assert m.cagr > 0.9
    assert m.max_drawdown == 0.0                        # monotonic up → no drawdown


def test_drawdown_series():
    nav = np.array([1.0, 1.2, 0.9, 1.1])
    dd = drawdown_series(nav)
    assert abs(dd[0]) < 1e-9                             # at peak
    assert abs(dd[2] - (0.9 / 1.2 - 1.0)) < 1e-9        # -25% trough
    assert dd.min() < 0
