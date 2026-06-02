"""Unit tests for backtest spec serialization + the result-cache hash (no DB)."""

from __future__ import annotations

from datetime import date

from src.portfolio.serialize import spec_from_dict, strategy_hash

SPEC = {
    "index_id": 3,
    "start": "2018-04-01",
    "end": "2026-05-01",
    "frequency": "quarterly",
    "basis": "consolidated",
    "benchmark_index_id": None,
    "filters": [
        {"factor_id": "roe", "op": ">=", "value": 15.0, "value2": None, "k": None},
        {"factor_id": "ret_12m", "op": "top_k", "value": None, "value2": None, "k": 20},
    ],
    "weighting": {"scheme": "market_cap", "score_factor_id": None, "max_weight": 0.1, "max_sector_weight": None},
}


def test_spec_from_dict_roundtrip():
    spec = spec_from_dict(SPEC)
    assert spec.index_id == 3
    assert spec.start == date(2018, 4, 1) and spec.end == date(2026, 5, 1)
    assert spec.frequency == "quarterly"
    assert spec.basis == "consolidated"
    assert len(spec.filters) == 2
    assert spec.filters[1].op == "top_k" and spec.filters[1].k == 20
    assert spec.weighting.scheme == "market_cap" and spec.weighting.max_weight == 0.1


def test_basis_falls_back_to_default():
    bad = dict(SPEC, basis="garbage")
    assert spec_from_dict(bad).basis == "consolidated"


def test_strategy_hash_is_stable_and_sensitive():
    h1 = strategy_hash(SPEC)
    # Key order doesn't matter (sorted), value changes do.
    reordered = {k: SPEC[k] for k in reversed(list(SPEC))}
    assert strategy_hash(reordered) == h1
    changed = dict(SPEC, end="2026-06-01")
    assert strategy_hash(changed) != h1
    assert len(h1) == 64                              # sha256 hex
