"""Unit tests for the custom-factor expression engine + normalization (no DB)."""

from __future__ import annotations

import math

import pytest

from src.portfolio.factors.custom import CustomFactorDef, normalize
from src.portfolio.factors.expression import ExpressionError, evaluate, parse_refs, validate


def test_parse_refs_and_validate():
    assert parse_refs("(roe + earnings_yield) / pb") == {"roe", "earnings_yield", "pb"}
    assert validate("roe / pb", {"roe", "pb", "x"}) == {"roe", "pb"}
    with pytest.raises(ExpressionError):
        validate("roe / unknown_factor", {"roe", "pb"})        # unknown ref
    with pytest.raises(ExpressionError):
        parse_refs("__import__('os')")                          # disallowed node
    with pytest.raises(ExpressionError):
        parse_refs("roe ** 2")                                  # disallowed op
    with pytest.raises(ExpressionError):
        parse_refs("42")                                        # no factor ref


def test_evaluate_arithmetic_and_missing():
    vals = {"roe": 20.0, "earnings_yield": 5.0, "pb": 2.0}
    assert evaluate("(roe + earnings_yield) / pb", vals) == 12.5
    assert evaluate("-roe", vals) == -20.0
    # Missing input → None (never zero).
    assert evaluate("roe / pb", {"roe": 10.0}) is None
    # Divide by zero → None.
    assert evaluate("roe / pb", {"roe": 10.0, "pb": 0.0}) is None


def test_custom_factor_def_captures_refs():
    cf = CustomFactorDef(id="cf1", name="Quality-Value", expression="roe / pb")
    assert cf.refs == frozenset({"roe", "pb"})


def test_normalize_zscore_and_rank():
    raw = {1: 10.0, 2: 20.0, 3: 30.0, 4: None}
    z = normalize(raw, "zscore")
    assert z[4] is None
    assert abs(sum(v for v in z.values() if v is not None)) < 1e-9     # mean ~0
    assert z[1] < z[2] < z[3]
    r = normalize(raw, "rank")
    assert r[1] == 0.0 and r[3] == 1.0 and r[2] == 0.5                  # percentile rank
    assert r[4] is None
    assert normalize(raw, "none") == raw                               # passthrough


def test_normalize_handles_constant_and_empty():
    assert normalize({1: 5.0, 2: 5.0}, "zscore") == {1: 0.0, 2: 0.0}   # zero variance
    assert normalize({}, "rank") == {}
    assert all(v is None for v in normalize({1: None}, "zscore").values())


def test_rank_is_monotonic_with_value():
    raw = {i: float(i) for i in range(5)}
    r = normalize(raw, "rank")
    assert math.isclose(r[0], 0.0) and math.isclose(r[4], 1.0)
    assert r[0] < r[1] < r[2] < r[3] < r[4]
