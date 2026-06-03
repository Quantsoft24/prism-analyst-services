"""Unit tests for the pure price-window + growth factor math (no DB)."""

from __future__ import annotations

import math
from datetime import date

from src.portfolio.factors import price_stats as ps


def _series(closes, values=None):
    """Build (date, close, value) tuples; dates are arbitrary but ascending."""
    values = values or [1.0] * len(closes)
    return [(date(2025, 1, 1), c, v) for c, v in zip(closes, values, strict=False)]


def test_trailing_return_pct():
    closes = [100.0] + [0.0] * 62 + [110.0]   # 64 points → lookback 63
    s = _series(closes)
    assert round(ps.trailing_return_pct(s, 63), 2) == 10.0   # 110/100 - 1
    # Not enough history → None.
    assert ps.trailing_return_pct(_series([100.0, 110.0]), 63) is None
    # Non-positive base → None.
    assert ps.trailing_return_pct(_series([0.0, 110.0]), 1) is None


def test_annualized_vol_pct():
    # Constant price → zero volatility.
    assert ps.annualized_vol_pct(_series([100.0] * 60)) == 0.0
    # Alternating ±1% → positive, annualised.
    closes = [100.0]
    for i in range(1, 60):
        closes.append(closes[-1] * (1.01 if i % 2 else 0.99))
    v = ps.annualized_vol_pct(_series(closes))
    assert v is not None and v > 0
    # Too few points → None.
    assert ps.annualized_vol_pct(_series([100.0] * 5)) is None


def test_adv_crore():
    s = _series([10.0] * 5, values=[2.0, 4.0, 6.0, 8.0, 10.0])
    assert ps.adv_crore(s, window=5) == 6.0
    assert ps.adv_crore(_series([], values=[])) is None


def test_cagr_pct():
    # 100 → 133.1 over 3 years = 10%/yr.
    series = [("2022-03", 100.0), ("2023-03", 110.0), ("2024-03", 121.0), ("2025-03", 133.1)]
    assert math.isclose(ps.cagr_pct(series, years=3), 10.0, abs_tol=1e-6)
    # No base period exactly 3y back → None.
    assert ps.cagr_pct([("2024-03", 121.0), ("2025-03", 133.1)], years=3) is None
    # Negative/zero base → None (CAGR undefined).
    assert ps.cagr_pct([("2022-03", -5.0), ("2025-03", 100.0)], years=3) is None
