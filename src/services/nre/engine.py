"""NRE engine — pure, deterministic financial math.

Every function:
  * is total + side-effect-free (same inputs → same output, no I/O)
  * validates inputs and raises ``NREError`` with a clear message on bad data
    (division by zero, negative CAGR base, empty series, …)
  * returns an ``NREResult`` carrying the operation name, the echoed inputs,
    the result, and a unit — so callers can show "how this number was derived"

Rounding: results are rounded to 4 significant decimal places for display
stability; callers needing full precision can recompute. We never round the
INPUTS — only the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class NREError(ValueError):
    """Raised when a computation can't be performed on the given inputs."""


@dataclass(slots=True)
class NREResult:
    operation: str
    result: float
    unit: str
    inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "result": self.result,
            "unit": self.unit,
            "inputs": self.inputs,
        }


def _round(x: float) -> float:
    return round(x, 4)


def growth_pct(start: float, end: float) -> NREResult:
    """Percent change from ``start`` to ``end``: (end - start) / |start| * 100.

    Used for YoY / QoQ growth. ``start`` must be non-zero.
    """
    if start == 0:
        raise NREError("growth_pct: start value is 0 — percentage change is undefined.")
    value = (end - start) / abs(start) * 100
    return NREResult("growth_pct", _round(value), "%", {"start": start, "end": end})


def cagr_pct(start: float, end: float, periods: float) -> NREResult:
    """Compound annual growth rate over ``periods``:
    ((end / start) ** (1 / periods) - 1) * 100.

    ``start`` must be > 0, ``end`` >= 0, ``periods`` > 0 (CAGR is undefined for
    a non-positive base or zero horizon).
    """
    if start <= 0:
        raise NREError("cagr_pct: start must be > 0.")
    if end < 0:
        raise NREError("cagr_pct: end must be >= 0.")
    if periods <= 0:
        raise NREError("cagr_pct: periods must be > 0.")
    value = ((end / start) ** (1 / periods) - 1) * 100
    return NREResult(
        "cagr_pct", _round(value), "%", {"start": start, "end": end, "periods": periods}
    )


def margin_pct(numerator: float, denominator: float) -> NREResult:
    """Margin / share: numerator / denominator * 100. e.g. operating margin =
    operating income / revenue * 100. ``denominator`` must be non-zero."""
    if denominator == 0:
        raise NREError("margin_pct: denominator is 0.")
    value = numerator / denominator * 100
    return NREResult(
        "margin_pct", _round(value), "%", {"numerator": numerator, "denominator": denominator}
    )


def ratio(a: float, b: float) -> NREResult:
    """Ratio a / b (e.g. debt-to-equity). ``b`` must be non-zero."""
    if b == 0:
        raise NREError("ratio: divisor is 0.")
    return NREResult("ratio", _round(a / b), "x", {"a": a, "b": b})


def delta(start: float, end: float) -> NREResult:
    """Absolute change end - start (same unit as inputs)."""
    return NREResult("delta", _round(end - start), "abs", {"start": start, "end": end})


def pct_of(part: float, whole: float) -> NREResult:
    """What percent ``part`` is of ``whole``: part / whole * 100."""
    if whole == 0:
        raise NREError("pct_of: whole is 0.")
    return NREResult("pct_of", _round(part / whole * 100), "%", {"part": part, "whole": whole})


def sum_values(values: list[float]) -> NREResult:
    """Sum of a series. Empty series → 0."""
    total = float(sum(values))
    return NREResult("sum", _round(total), "abs", {"values": values, "count": len(values)})


def average(values: list[float]) -> NREResult:
    """Arithmetic mean. Empty series raises (mean is undefined)."""
    if not values:
        raise NREError("average: empty series.")
    return NREResult(
        "average", _round(sum(values) / len(values)), "abs", {"values": values, "count": len(values)}
    )
