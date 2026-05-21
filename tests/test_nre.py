"""Tests for the Numerical Reasoning Engine — pure, exhaustive, no LLM/DB.

These are the most important tests in the codebase to get right: every number
PRISM shows an analyst flows through here. We cover happy paths, edge cases,
and the error conditions that must NOT silently produce wrong numbers.
"""

from __future__ import annotations

import math

import pytest

from src.services.nre import engine
from src.tools.nre_tools import (
    compute_cagr,
    compute_growth,
    compute_margin,
    compute_percent_of,
    compute_ratio,
)

# ── growth_pct ──────────────────────────────────────────────────────────────


def test_growth_positive():
    r = engine.growth_pct(657990, 706980)
    assert r.operation == "growth_pct"
    assert r.unit == "%"
    assert r.result == pytest.approx(7.4453, abs=1e-3)
    assert r.inputs == {"start": 657990, "end": 706980}


def test_growth_negative():
    assert engine.growth_pct(100, 90).result == pytest.approx(-10.0)


def test_growth_zero_start_raises():
    with pytest.raises(engine.NREError, match="start value is 0"):
        engine.growth_pct(0, 100)


def test_growth_uses_abs_start_for_sign():
    # From a negative base, an increase should read as positive growth.
    assert engine.growth_pct(-100, -50).result == pytest.approx(50.0)


# ── cagr_pct ────────────────────────────────────────────────────────────────


def test_cagr_basic():
    # Doubling over 3 years ≈ 25.99% CAGR.
    r = engine.cagr_pct(100, 200, 3)
    assert r.result == pytest.approx(25.9921, abs=1e-3)
    assert r.unit == "%"


def test_cagr_flat_is_zero():
    assert engine.cagr_pct(100, 100, 5).result == pytest.approx(0.0)


def test_cagr_nonpositive_start_raises():
    with pytest.raises(engine.NREError, match="start must be > 0"):
        engine.cagr_pct(0, 100, 3)


def test_cagr_zero_periods_raises():
    with pytest.raises(engine.NREError, match="periods must be > 0"):
        engine.cagr_pct(100, 200, 0)


# ── margin_pct ──────────────────────────────────────────────────────────────


def test_margin_basic():
    # Operating margin = 668380 / 2670210 * 100 ≈ 25.03% (TCS FY26 fact sheet).
    r = engine.margin_pct(668380, 2670210)
    assert r.result == pytest.approx(25.0291, abs=1e-3)


def test_margin_zero_denominator_raises():
    with pytest.raises(engine.NREError, match="denominator is 0"):
        engine.margin_pct(100, 0)


# ── ratio / delta / pct_of ──────────────────────────────────────────────────


def test_ratio():
    r = engine.ratio(150, 100)
    assert r.result == pytest.approx(1.5)
    assert r.unit == "x"


def test_ratio_zero_divisor_raises():
    with pytest.raises(engine.NREError):
        engine.ratio(1, 0)


def test_delta():
    assert engine.delta(100, 130).result == pytest.approx(30.0)


def test_pct_of():
    assert engine.pct_of(25, 200).result == pytest.approx(12.5)


def test_pct_of_zero_whole_raises():
    with pytest.raises(engine.NREError):
        engine.pct_of(1, 0)


# ── sum / average ───────────────────────────────────────────────────────────


def test_sum():
    assert engine.sum_values([1, 2, 3.5]).result == pytest.approx(6.5)


def test_sum_empty_is_zero():
    assert engine.sum_values([]).result == 0.0


def test_average():
    assert engine.average([2, 4, 6]).result == pytest.approx(4.0)


def test_average_empty_raises():
    with pytest.raises(engine.NREError, match="empty series"):
        engine.average([])


# ── Tool wrappers: errors come back as data, not exceptions ─────────────────


def test_tool_growth_ok():
    out = compute_growth(start=100, end=110)
    assert out["result"] == pytest.approx(10.0)
    assert out["unit"] == "%"
    assert "error" not in out


def test_tool_growth_error_is_data():
    out = compute_growth(start=0, end=100)
    assert "error" in out
    assert "start value is 0" in out["error"]


def test_tool_cagr_error_is_data():
    out = compute_cagr(start=-5, end=100, periods=3)
    assert "error" in out


def test_tool_margin_ok():
    out = compute_margin(numerator=25, denominator=100)
    assert out["result"] == pytest.approx(25.0)


def test_tool_ratio_and_percent_of():
    assert compute_ratio(a=10, b=4)["result"] == pytest.approx(2.5)
    assert compute_percent_of(part=1, whole=4)["result"] == pytest.approx(25.0)


def test_results_are_finite():
    """Defensive: no operation should ever leak inf/nan to a caller."""
    for r in (
        engine.growth_pct(1, 1_000_000),
        engine.cagr_pct(1, 1_000_000, 10),
        engine.margin_pct(1, 1_000_000),
    ):
        assert math.isfinite(r.result)
