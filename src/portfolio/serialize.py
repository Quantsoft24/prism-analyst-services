"""(De)serialization between a backtest spec/result and JSON-able dicts.

The spec dict is what we persist (``pb_backtests.spec``), hash for the result
cache, and replay; the result dict is the stored ``result`` JSONB the API
returns. Kept free of Pydantic/schema imports to avoid a cycle — the router maps
its request model to a spec dict, then everything downstream uses plain dicts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import date

from src.portfolio.backtest import (
    BacktestResult,
    BacktestSpec,
    WeightingConfig,
)
from src.portfolio.calendar import Frequency
from src.portfolio.constants import DEFAULT_BASIS, Basis
from src.portfolio.factors.custom import CustomFactorDef
from src.portfolio.screening import Filter


def spec_from_dict(d: dict) -> BacktestSpec:
    """Rebuild a ``BacktestSpec`` from its persisted dict."""
    w = d.get("weighting") or {}
    basis: Basis = d.get("basis") if d.get("basis") in ("consolidated", "standalone") else DEFAULT_BASIS
    freq: Frequency = d.get("frequency", "quarterly")
    return BacktestSpec(
        index_id=int(d["index_id"]),
        start=date.fromisoformat(d["start"]),
        end=date.fromisoformat(d["end"]),
        frequency=freq,
        filters=[
            Filter(
                factor_id=f["factor_id"], op=f["op"],
                value=f.get("value"), value2=f.get("value2"), k=f.get("k"),
            )
            for f in d.get("filters", [])
        ],
        weighting=WeightingConfig(
            scheme=w.get("scheme", "equal"),
            score_factor_id=w.get("score_factor_id"),
            max_weight=w.get("max_weight"),
            max_sector_weight=w.get("max_sector_weight"),
        ),
        basis=basis,
        benchmark_index_id=d.get("benchmark_index_id"),
        custom=[
            CustomFactorDef(
                id=c["id"], name=c.get("name", c["id"]), expression=c["expression"],
                direction=c.get("direction", "higher_better"),
                normalization=c.get("normalization", "none"),
            )
            for c in d.get("custom_factors", [])
        ],
    )


def strategy_hash(spec_dict: dict) -> str:
    """Stable hash of the spec for the result cache (order-insensitive JSON)."""
    blob = json.dumps(spec_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def result_to_dict(res: BacktestResult) -> dict:
    """Serialize a ``BacktestResult`` for the ``result`` JSONB column / API."""
    return {
        "dates": [d.isoformat() for d in res.dates],
        "nav": res.nav,
        "benchmark_nav": res.benchmark_nav,
        "drawdown": res.drawdown,
        "metrics": asdict(res.metrics),
        "benchmark_metrics": asdict(res.benchmark_metrics),
        "rebalances": [
            {
                "date": r.date.isoformat(),
                "n_holdings": r.n_holdings,
                "turnover": r.turnover,
                "holdings": [
                    {
                        "security_id": h.security_id,
                        "symbol": h.symbol,
                        "sector": h.sector,
                        "weight": h.weight,
                        "is_new": h.is_new,
                    }
                    for h in r.holdings
                ],
            }
            for r in res.rebalances
        ],
        "notes": res.notes,
    }
