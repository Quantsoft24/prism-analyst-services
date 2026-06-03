"""Pure functions for price-window + growth factor math (no DB).

Kept side-effect-free and dependency-light (stdlib only) so they're unit-tested
in isolation and reused by both the screening compute path and the backtest
engine. Prices are corporate-action adjusted (owner-confirmed), so close-to-close
returns are clean. Trading-day lookbacks (≈21/mo) are used instead of calendar
months — robust to holidays, standard for momentum.
"""

from __future__ import annotations

import math

# Trading-day lookbacks (~21 sessions/month).
TD_3M = 63
TD_6M = 126
TD_12M = 252
VOL_WINDOW = 252   # ~1y of daily returns
ADV_WINDOW = 63    # ~3m average daily traded value

# (trade_date, close, trade_value) tuples, ascending by date.
PriceSeries = list[tuple[object, float | None, float | None]]


def _closes(series: PriceSeries) -> list[float]:
    return [c for _, c, _ in series if c is not None]


def trailing_return_pct(series: PriceSeries, lookback: int) -> float | None:
    """Total return (%) over the last ``lookback`` trading sessions."""
    closes = _closes(series)
    if len(closes) < lookback + 1:
        return None
    base, last = closes[-1 - lookback], closes[-1]
    if base <= 0:
        return None
    return (last / base - 1.0) * 100.0


def annualized_vol_pct(series: PriceSeries, window: int = VOL_WINDOW) -> float | None:
    """Annualised volatility (%) of daily simple returns over the last ``window``."""
    closes = _closes(series)[-(window + 1) :]
    rets = [
        closes[i] / closes[i - 1] - 1.0
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]
    if len(rets) < 10:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252.0) * 100.0


def adv_crore(series: PriceSeries, window: int = ADV_WINDOW) -> float | None:
    """Average daily traded value (₹ crore) over the last ``window`` sessions."""
    vals = [v for _, _, v in series if v is not None][-window:]
    if not vals:
        return None
    return sum(vals) / len(vals)


def cagr_pct(periods_values: list[tuple[str, float | None]], years: int = 3) -> float | None:
    """CAGR (%) between the latest fiscal period and the one ``years`` earlier.

    ``periods_values`` is ascending ``[(period 'YYYY-MM', value), …]`` (already
    point-in-time/lagged by the caller). Returns None unless a base period exactly
    ``years`` fiscal-years before the latest exists with a positive base value
    (negative/zero bases make CAGR meaningless)."""
    pv = [(p, v) for p, v in periods_values if v is not None]
    if len(pv) < 2:
        return None
    latest_p, latest_v = pv[-1]
    target_year = int(latest_p[:4]) - years
    base = next((iv for iv in pv if int(iv[0][:4]) == target_year), None)
    if base is None:
        return None
    base_p, base_v = base
    n = int(latest_p[:4]) - int(base_p[:4])
    if n <= 0 or base_v <= 0 or latest_v <= 0:
        return None
    return ((latest_v / base_v) ** (1.0 / n) - 1.0) * 100.0
