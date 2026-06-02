"""Custom (user-composed) factor definitions + cross-sectional normalization.

A custom factor is an arithmetic expression over base factor ids (e.g.
``(roe + earnings_yield) / pb``) plus a direction and a normalization mode.
Normalization is applied **cross-sectionally** over the screened universe — the
right place to reconcile unit/scale mismatches between composed factors.
"""

from __future__ import annotations

import statistics
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Literal

from src.portfolio.factors.expression import parse_refs

Normalization = Literal["none", "zscore", "rank"]


@dataclass
class CustomFactorDef:
    id: str                      # namespaced id used in filters/weighting (not a registry id)
    name: str
    expression: str
    direction: str = "higher_better"
    normalization: Normalization = "none"
    refs: frozenset[str] = field(default_factory=frozenset, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "refs", frozenset(parse_refs(self.expression)))


def normalize(raw: dict[int, float | None], method: Normalization) -> dict[int, float | None]:
    """Cross-sectional normalization over the securities with a value.

    ``none`` → raw values; ``zscore`` → (x − mean)/σ; ``rank`` → percentile rank
    in [0, 1] (higher value → higher rank, ties averaged). Missing stay None.
    """
    vals = [v for v in raw.values() if v is not None]
    if method == "none" or not vals:
        return dict(raw)

    if method == "zscore":
        mean = sum(vals) / len(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return {
            k: (0.0 if sd == 0 else (v - mean) / sd) if v is not None else None
            for k, v in raw.items()
        }

    if method == "rank":
        order = sorted(vals)
        n = len(order)
        out: dict[int, float | None] = {}
        for k, v in raw.items():
            if v is None:
                out[k] = None
                continue
            lo, hi = bisect_left(order, v), bisect_right(order, v)
            avg_rank = (lo + hi - 1) / 2.0          # 0-indexed average rank
            out[k] = avg_rank / (n - 1) if n > 1 else 0.5
        return out

    return dict(raw)
