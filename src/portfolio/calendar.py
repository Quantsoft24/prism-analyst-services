"""Rebalance-date generation on the real trading-day calendar (pure, no DB).

The construction rules screen on a period boundary and execute on an actual
trading day. Given the benchmark's trading-day axis, this maps each calendar
period-end (or 15-day step) to the **last trading day on or before** it — so
month-end strategies rebalance on the real last trading session of the month.
"""

from __future__ import annotations

import bisect
import calendar as _cal
from datetime import date, timedelta
from typing import Literal

Frequency = Literal["15d", "monthly", "quarterly", "semiannual", "annual"]

FREQ_MONTHS: dict[str, int] = {
    "monthly": 1,
    "quarterly": 3,
    "semiannual": 6,
    "annual": 12,
}
FREQ_LABEL: dict[str, str] = {
    "15d": "Every 15 days",
    "monthly": "Monthly",
    "quarterly": "Quarterly",
    "semiannual": "Semi-annual",
    "annual": "Annual",
}


def _month_end(year: int, month: int) -> date:
    return date(year, month, _cal.monthrange(year, month)[1])


def _on_or_before(trading_days: list[date], target: date) -> date | None:
    """The latest trading day ``<= target`` (None if before the calendar starts)."""
    i = bisect.bisect_right(trading_days, target)
    return trading_days[i - 1] if i > 0 else None


def _candidates(frequency: Frequency, start: date, end: date) -> list[date]:
    """Calendar period boundaries between ``start`` and ``end`` for the freq."""
    out: list[date] = []
    if frequency == "15d":
        d = start
        while d <= end:
            out.append(d)
            d += timedelta(days=15)
        return out
    step = FREQ_MONTHS[frequency]
    # March-anchored period ends are natural for Indian FY-aligned strategies,
    # but to stay general we step month-ends from the start month.
    y, m = start.year, start.month
    while True:
        cand = _month_end(y, m)
        if cand > end:
            break
        if cand >= start:
            out.append(cand)
        total = (y * 12 + (m - 1)) + step
        y, m = divmod(total, 12)
        m += 1
    return out


def generate_rebalance_dates(
    trading_days: list[date], frequency: Frequency, start: date, end: date
) -> list[date]:
    """Rebalance trading days for ``frequency`` within ``[start, end]`` — each is
    the last actual trading day on/before a calendar boundary. De-duplicated and
    ascending; empty if the window has no trading days."""
    if not trading_days:
        return []
    seen: set[date] = set()
    out: list[date] = []
    for cand in _candidates(frequency, start, end):
        td = _on_or_before(trading_days, cand)
        if td is not None and td >= start and td not in seen:
            seen.add(td)
            out.append(td)
    return out
