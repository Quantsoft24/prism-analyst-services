"""Unit tests for the screening + weighting engines (pure, no DB)."""

from __future__ import annotations

from datetime import date

from src.portfolio.factors.compute import FactorMatrix
from src.portfolio.screening import Filter, apply_filters
from src.portfolio.weighting import compute_weights


def _fm(values: dict[int, dict[str, float | None]]) -> FactorMatrix:
    sids = list(values)
    fids = sorted({f for row in values.values() for f in row})
    return FactorMatrix(date(2026, 1, 1), sids, fids, values, {}, {})


def test_sequential_and_filters_and_funnel():
    fm = _fm({
        1: {"pe": 10.0, "roe": 25.0},
        2: {"pe": 30.0, "roe": 22.0},
        3: {"pe": 12.0, "roe": 8.0},
        4: {"pe": 15.0, "roe": None},   # missing roe → drops on a roe filter
    })
    res = apply_filters(fm, [Filter("pe", "<", 20.0), Filter("roe", ">=", 15.0)])
    assert res.survivors == [1]                      # only #1 passes both
    assert [s.remaining for s in res.funnel] == [4, 3, 1]  # universe, after f1, after f2


def test_top_k_selects_highest():
    fm = _fm({1: {"mc": 100.0}, 2: {"mc": 300.0}, 3: {"mc": 200.0}, 4: {"mc": None}})
    res = apply_filters(fm, [Filter("mc", "top_k", k=2)])
    assert set(res.survivors) == {2, 3}              # missing value excluded


def test_between_is_inclusive_and_order_agnostic():
    fm = _fm({1: {"x": 5.0}, 2: {"x": 10.0}, 3: {"x": 15.0}})
    res = apply_filters(fm, [Filter("x", "between", value=12.0, value2=4.0)])
    assert set(res.survivors) == {1, 2}


def test_equal_and_market_cap_weights():
    eq = compute_weights([1, 2, 3, 4], scheme="equal")
    assert all(abs(w - 0.25) < 1e-9 for w in eq.values())

    mc = compute_weights([1, 2], scheme="market_cap", market_cap={1: 300.0, 2: 100.0})
    assert abs(mc[1] - 0.75) < 1e-9 and abs(mc[2] - 0.25) < 1e-9
    assert abs(sum(mc.values()) - 1.0) < 1e-9


def test_per_name_cap_water_fills():
    w = compute_weights(
        [1, 2, 3], scheme="market_cap",
        market_cap={1: 900.0, 2: 50.0, 3: 50.0}, max_weight=0.5,
    )
    assert abs(w[1] - 0.5) < 1e-9                     # capped
    assert abs(w[2] - 0.25) < 1e-9 and abs(w[3] - 0.25) < 1e-9   # freed weight split
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_factor_score_tilts_to_better_rank():
    w = compute_weights(
        [1, 2, 3], scheme="factor_score",
        score={1: 30.0, 2: 20.0, 3: 10.0}, score_direction="higher_better",
    )
    assert w[1] > w[2] > w[3]                         # best rank → most weight
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_missing_input_drops_from_book():
    # #3 has no market cap → excluded from a market-cap book (not zero-weighted).
    w = compute_weights([1, 2, 3], scheme="market_cap", market_cap={1: 100.0, 2: 100.0, 3: None})
    assert set(w) == {1, 2}
    assert abs(sum(w.values()) - 1.0) < 1e-9
