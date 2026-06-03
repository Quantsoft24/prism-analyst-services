"""Tests for the point-in-time annual-data lag + the factor registry/compute.

Pure unit tests — no DB. They lock in the institutional-correctness guarantee
(no look-ahead from the 6-month reporting lag) and the factor math (PAT = PBT −
tax, sector exclusions, missing-input → None never zero).
"""

from __future__ import annotations

from datetime import date

from src.portfolio import constants as C
from src.portfolio.factors.registry import REGISTRY, FactorContext, required_variables
from src.portfolio.lag import is_usable, known_as_of, period_end


def test_period_end_is_month_end():
    assert period_end("2025-03") == date(2025, 3, 31)
    assert period_end("2024-12") == date(2024, 12, 31)
    assert period_end("2024-02") == date(2024, 2, 29)  # leap year clamp


def test_known_as_of_adds_six_months():
    assert known_as_of("2025-03") == date(2025, 9, 30)
    assert known_as_of("2025-12") == date(2026, 6, 30)


def test_usable_respects_the_lag_boundary():
    # FY ending 2025-03 is only knowable from 2025-09-30 — not before.
    assert not is_usable("2025-03", date(2025, 8, 1))
    assert not is_usable("2025-03", date(2025, 9, 29))
    assert is_usable("2025-03", date(2025, 9, 30))   # boundary inclusive
    assert is_usable("2025-03", date(2026, 1, 1))


def test_lag_is_configurable():
    assert known_as_of("2025-03", lag_months=0) == date(2025, 3, 31)
    assert known_as_of("2025-03", lag_months=12) == date(2026, 3, 31)


# ── Factor math ──────────────────────────────────────────────────────────────

def _ctx(funds=None, market_cap=None, sector=None):
    return FactorContext(funds=funds or {}, market_cap=market_cap, sector=sector)


def test_pat_is_pbt_minus_tax():
    c = _ctx({C.V_PBT: 1000.0, C.V_DIRECT_TAX: 250.0})
    assert c.pat == 750.0
    # Missing either input → None (never treated as zero).
    assert _ctx({C.V_PBT: 1000.0}).pat is None
    assert _ctx({C.V_DIRECT_TAX: 250.0}).pat is None


def test_pe_and_roe_and_missing_inputs():
    pe = REGISTRY["pe"].compute
    roe = REGISTRY["roe"].compute
    c = _ctx({C.V_PBT: 1000.0, C.V_DIRECT_TAX: 200.0, C.V_EQUITY: 4000.0}, market_cap=16000.0)
    assert round(pe(c), 2) == 20.0          # 16000 / (1000-200)
    assert round(roe(c), 2) == 20.0         # 800 / 4000 * 100
    # No equity → ROE None, not 0.
    assert roe(_ctx({C.V_PBT: 1000.0, C.V_DIRECT_TAX: 200.0})) is None
    # Zero denominator → None (guarded).
    assert pe(_ctx({C.V_PBT: 0.0, C.V_DIRECT_TAX: 0.0}, market_cap=16000.0)) is None


def test_leverage_factors_exclude_financials():
    for fid in ("debt_equity", "interest_coverage", "roce"):
        assert C.SECTOR_FINANCIALS in REGISTRY[fid].exclude_sectors


def test_required_variables_union():
    vars_ = required_variables(["pe", "pb"])
    assert C.V_PBT in vars_ and C.V_DIRECT_TAX in vars_ and C.V_EQUITY in vars_
