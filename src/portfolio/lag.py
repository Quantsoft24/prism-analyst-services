"""Point-in-time reporting lag — the institutional-correctness wedge.

``annual_data.date`` is a fiscal period end as ``'YYYY-MM'``. A value is only
*knowable* ``ANNUAL_DATA_LAG_MONTHS`` (=6) months after its period-end, so when
screening / backtesting as of date ``D`` we must use only rows whose
``known_as_of <= D``. This module is the single source of that rule — both as
Python helpers (for app logic / tests) and as a reusable SQL fragment (for the
set-based factor/backtest queries). Never read ``annual_data`` for screening
without going through here.
"""

from __future__ import annotations

import calendar
from datetime import date

from src.portfolio.constants import ANNUAL_DATA_LAG_MONTHS


def _add_months(d: date, months: int) -> date:
    """Add ``months`` calendar months to ``d``, clamping the day-of-month."""
    total = d.year * 12 + (d.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def period_end(period: str) -> date:
    """``'2025-03'`` → ``date(2025, 3, 31)`` (last day of the fiscal month)."""
    year_s, month_s = period.split("-")
    year, month = int(year_s), int(month_s)
    return date(year, month, calendar.monthrange(year, month)[1])


def known_as_of(period: str, lag_months: int = ANNUAL_DATA_LAG_MONTHS) -> date:
    """The earliest date a fiscal period's annual data is treated as known —
    ``period_end + lag_months``."""
    return _add_months(period_end(period), lag_months)


def is_usable(period: str, as_of: date, lag_months: int = ANNUAL_DATA_LAG_MONTHS) -> bool:
    """True if ``period``'s annual data is knowable as of ``as_of`` (no look-ahead)."""
    return known_as_of(period, lag_months) <= as_of


# Reusable SQL predicate for the lag, over ``annual_data.date`` ('YYYY-MM').
# Bind params: ``:as_of`` (date), ``:lag_months`` (int). Mirrors ``is_usable``.
#   period_end   = (date || '-01')::date + 1 month - 1 day
#   known_as_of  = period_end + :lag_months months
# Usable iff known_as_of <= :as_of.
LAG_USABLE_SQL: str = (
    "((to_date(date || '-01', 'YYYY-MM-DD') "
    " + interval '1 month' - interval '1 day' "
    " + make_interval(months => :lag_months)) <= :as_of)"
)
