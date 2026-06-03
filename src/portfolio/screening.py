"""Sequential-AND factor screening over a computed factor matrix.

Filters apply in order; each narrows the surviving set (AND). A security with a
``None`` value for a filter's factor cannot satisfy a numeric comparison, so it
drops out and is counted — never treated as zero. The per-step funnel
(``universe → after filter 1 → …``) drives the UI's "N → M names" display.

Pure logic (no DB) — the caller computes the ``FactorMatrix`` first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.portfolio.factors.compute import FactorMatrix

Operator = Literal[">", ">=", "<", "<=", "=", "between", "top_k", "bottom_k"]
_EQ_TOL = 1e-9


@dataclass(frozen=True)
class Filter:
    """One screening step. ``value2`` is the upper bound for ``between``; ``k`` is
    the count for ``top_k`` / ``bottom_k``."""

    factor_id: str
    op: Operator
    value: float | None = None
    value2: float | None = None
    k: int | None = None


@dataclass
class FunnelStep:
    label: str
    remaining: int


@dataclass
class ScreenResult:
    survivors: list[int]            # security_ids passing all filters (input order)
    funnel: list[FunnelStep]        # universe → after each filter


def _passes(val: float | None, f: Filter) -> bool:
    if val is None:
        return False
    if f.op == ">":
        return f.value is not None and val > f.value
    if f.op == ">=":
        return f.value is not None and val >= f.value
    if f.op == "<":
        return f.value is not None and val < f.value
    if f.op == "<=":
        return f.value is not None and val <= f.value
    if f.op == "=":
        return f.value is not None and abs(val - f.value) <= _EQ_TOL
    if f.op == "between":
        lo, hi = f.value, f.value2
        if lo is None or hi is None:
            return False
        if lo > hi:
            lo, hi = hi, lo
        return lo <= val <= hi
    return True  # top_k / bottom_k handled separately


def apply_filters(fm: FactorMatrix, filters: list[Filter]) -> ScreenResult:
    """Run ``filters`` as sequential AND over the matrix's universe."""
    survivors = list(fm.security_ids)
    funnel = [FunnelStep("Universe", len(survivors))]

    for i, f in enumerate(filters, start=1):
        if f.op in ("top_k", "bottom_k"):
            k = f.k or 0
            ranked = [
                (sid, fm.value(sid, f.factor_id))
                for sid in survivors
                if fm.value(sid, f.factor_id) is not None
            ]
            ranked.sort(key=lambda t: t[1], reverse=(f.op == "top_k"))
            keep = {sid for sid, _ in ranked[: max(k, 0)]}
            survivors = [sid for sid in survivors if sid in keep]
        else:
            survivors = [sid for sid in survivors if _passes(fm.value(sid, f.factor_id), f)]
        funnel.append(FunnelStep(f"Filter {i}: {f.factor_id} {f.op}", len(survivors)))

    return ScreenResult(survivors=survivors, funnel=funnel)
