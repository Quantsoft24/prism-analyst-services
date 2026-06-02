"""Portfolio weighting schemes + optional caps (pure logic, no DB).

Schemes (industry-standard, all backed by data we have):
  * ``equal``        — 1/N.
  * ``market_cap``   — proportional to point-in-time market cap.
  * ``factor_score`` — tilt toward better-ranked names. Uses cross-sectional
    RANK (not raw values) so it is robust to negative/odd-scale factors.
  * ``inverse_vol``  — proportional to 1/volatility (risk-parity-lite).

A security missing the scheme's required input is dropped from the book and
reported by the caller (never zero-weighted into the denominator). Optional
``max_weight`` (per name) and ``max_sector_weight`` caps are enforced by
water-filling: cap the offenders, redistribute the freed weight to the rest in
proportion, repeat to convergence. Final weights sum to 1.
"""

from __future__ import annotations

from typing import Literal

Scheme = Literal["equal", "market_cap", "factor_score", "inverse_vol"]
Direction = Literal["higher_better", "lower_better"]
_EPS = 1e-12


def _normalize(raw: dict[int, float]) -> dict[int, float]:
    total = sum(raw.values())
    if total <= 0:
        n = len(raw)
        return {k: 1.0 / n for k in raw} if n else {}
    return {k: v / total for k, v in raw.items()}


def _base_weights(
    survivors: list[int],
    scheme: Scheme,
    market_cap: dict[int, float | None] | None,
    score: dict[int, float | None] | None,
    score_direction: Direction,
    volatility: dict[int, float | None] | None,
) -> dict[int, float]:
    if scheme == "equal":
        return _normalize({s: 1.0 for s in survivors})

    if scheme == "market_cap":
        raw = {s: mc for s in survivors if (mc := (market_cap or {}).get(s)) and mc > 0}
        return _normalize(raw)

    if scheme == "inverse_vol":
        raw = {
            s: 1.0 / v
            for s in survivors
            if (v := (volatility or {}).get(s)) is not None and v > 0
        }
        return _normalize(raw)

    if scheme == "factor_score":
        scored = [(s, sc) for s in survivors if (sc := (score or {}).get(s)) is not None]
        if not scored:
            return {}
        # Rank best→worst; best gets the highest weight. Ranks are 1..n.
        scored.sort(key=lambda t: t[1], reverse=(score_direction == "higher_better"))
        n = len(scored)
        raw = {sid: float(n - i) for i, (sid, _) in enumerate(scored)}
        return _normalize(raw)

    return _normalize({s: 1.0 for s in survivors})


def _apply_cap(weights: dict[int, float], cap: float, max_iter: int = 50) -> dict[int, float]:
    """Water-fill a per-name cap: clamp offenders to ``cap``, redistribute the
    freed weight proportionally among the rest, repeat. If the cap is infeasible
    (cap * n < 1) it degrades to equal weight at the cap."""
    if cap <= 0 or not weights:
        return weights
    if cap * len(weights) <= 1.0 + _EPS:
        eq = 1.0 / len(weights)
        return dict.fromkeys(weights, eq)
    w = dict(weights)
    for _ in range(max_iter):
        over = {k: v for k, v in w.items() if v > cap + _EPS}
        if not over:
            break
        for k in over:
            w[k] = cap
        capped_total = cap * len(over)
        free = 1.0 - capped_total
        under = {k: v for k, v in w.items() if v < cap - _EPS}
        under_total = sum(under.values())
        if under_total <= 0:
            break
        for k in under:
            w[k] = under[k] / under_total * free
    return _normalize(w)


def _apply_sector_cap(
    weights: dict[int, float],
    sector: dict[int, str | None],
    cap: float,
    max_iter: int = 50,
) -> dict[int, float]:
    """Cap each sector's total weight, scaling its names down and redistributing
    to under-cap sectors in proportion. Best-effort iterative; sums to 1."""
    if cap <= 0 or not weights:
        return weights
    w = dict(weights)
    for _ in range(max_iter):
        totals: dict[str | None, float] = {}
        for k, v in w.items():
            totals[sector.get(k)] = totals.get(sector.get(k), 0.0) + v
        over = {s: t for s, t in totals.items() if t > cap + _EPS}
        if not over:
            break
        for s, t in over.items():
            scale = cap / t
            for k in w:
                if sector.get(k) == s:
                    w[k] *= scale
        # Redistribute the freed weight to under-cap names in proportion.
        under = {k: v for k, v in w.items() if totals.get(sector.get(k), 0.0) <= cap + _EPS}
        under_total = sum(under.values())
        free = 1.0 - sum(w.values())
        if under_total <= 0 or free <= _EPS:
            break
        for k in under:
            w[k] += under[k] / under_total * free
    return _normalize(w)


def compute_weights(
    survivors: list[int],
    *,
    scheme: Scheme = "equal",
    market_cap: dict[int, float | None] | None = None,
    score: dict[int, float | None] | None = None,
    score_direction: Direction = "higher_better",
    volatility: dict[int, float | None] | None = None,
    sector: dict[int, str | None] | None = None,
    max_weight: float | None = None,
    max_sector_weight: float | None = None,
) -> dict[int, float]:
    """Weights (summing to 1) for ``survivors`` under ``scheme`` + optional caps."""
    weights = _base_weights(survivors, scheme, market_cap, score, score_direction, volatility)
    if not weights:
        return {}
    if max_weight is not None:
        weights = _apply_cap(weights, max_weight)
    if max_sector_weight is not None and sector is not None:
        weights = _apply_sector_cap(weights, sector, max_sector_weight)
        if max_weight is not None:
            weights = _apply_cap(weights, max_weight)
    return weights
