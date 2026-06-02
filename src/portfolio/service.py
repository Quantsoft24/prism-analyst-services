"""Orchestration for the portfolio builder API.

Ties the read-only repo + factor engine + screening + weighting into the two
read surfaces the UI needs now: the factor catalog and the **Suggested
Portfolio** (``run_screen``). Point-in-time correctness comes from the layers
below; this module just composes them and shapes the API response (with honest
coverage + a filter funnel).
"""

from __future__ import annotations

from src.portfolio.constants import DEFAULT_BASIS, Basis
from src.portfolio.factors import registry
from src.portfolio.factors.compute import compute_factors
from src.portfolio.factors.custom import CustomFactorDef
from src.portfolio.repository import PortfolioRepository
from src.portfolio.screening import Filter, apply_filters
from src.portfolio.weighting import compute_weights
from src.schemas.portfolio import (
    BacktestRequest,
    CoverageRead,
    FactorMetaRead,
    FactorPreviewRequest,
    FactorPreviewResponse,
    FactorPreviewRow,
    FunnelStepRead,
    HoldingRead,
    ScreenRequest,
    ScreenResponse,
    UniverseRead,
)

# Factors always shown on holdings even if not filtered on.
_DEFAULT_DISPLAY = ["market_cap", "pe", "roe"]


def factor_catalog() -> list[FactorMetaRead]:
    return [
        FactorMetaRead(
            id=f.id, name=f.name, category=f.category, unit=f.unit,
            direction=f.direction, default_operator=f.default_operator,
            data_kind=f.data_kind, source_tables=list(f.source_tables),
            description=f.description, exclude_sectors=list(f.exclude_sectors),
            decimals=f.decimals,
        )
        for f in registry.all_factors()
    ]


def _resolve_basis(raw: str | None) -> Basis:
    return raw if raw in ("consolidated", "standalone") else DEFAULT_BASIS


def backtest_request_to_spec_dict(req: BacktestRequest) -> dict:
    """Canonical, JSON-able spec dict — persisted, hashed for the cache, replayed
    by the worker. Order of keys/filters is preserved so the hash is stable."""
    return {
        "index_id": req.index_id,
        "start": req.start.isoformat(),
        "end": req.end.isoformat(),
        "frequency": req.frequency,
        "basis": _resolve_basis(req.basis),
        "benchmark_index_id": req.benchmark_index_id,
        "filters": [f.model_dump() for f in req.filters],
        "weighting": req.weighting.model_dump(),
        "custom_factors": [cf.model_dump() for cf in req.custom_factors],
    }


async def run_screen(repo: PortfolioRepository, req: ScreenRequest) -> ScreenResponse:
    notes: list[str] = []
    basis = _resolve_basis(req.basis)

    # As-of: caller's date, else the latest trading day ("today" for this data).
    as_of = req.as_of or await repo.latest_trade_date()
    if as_of is None:
        raise ValueError("No price data available to anchor the as-of date.")
    if req.as_of and req.as_of > (latest := await repo.latest_trade_date() or req.as_of):
        as_of = latest
        notes.append(f"as_of clamped to the latest available trading day ({latest}).")

    universes = {u.index_id: u for u in await repo.list_universes()}
    uni = universes.get(req.index_id)
    if uni is None:
        raise ValueError(f"Unknown universe index_id={req.index_id}.")

    members = await repo.members_as_of(req.index_id, as_of)
    if not members:
        notes.append("No index membership snapshot on or before the as-of date.")
        return ScreenResponse(
            as_of=as_of, universe=UniverseRead(**uni.__dict__), membership_count=0,
            basis=basis, weighting_scheme=req.weighting.scheme, holdings=[],
            funnel=[], coverage=[], dropped_no_weight=0, notes=notes,
        )

    # Custom factors (inline) — validated on construction; bad expr → ValueError.
    custom_defs = [
        CustomFactorDef(
            id=cf.id, name=cf.name, expression=cf.expression,
            direction=cf.direction, normalization=cf.normalization,
        )
        for cf in req.custom_factors
    ]
    custom_dir = {c.id: c.direction for c in custom_defs}

    # Factors to compute = filters + weighting inputs + display. Custom ids are
    # valid references too (resolved by the engine alongside the registry ids).
    valid = set(registry.REGISTRY) | {c.id for c in custom_defs}
    filter_fids = [f.factor_id for f in req.filters if f.factor_id in valid]
    display_fids = [f for f in (req.display_factors or _DEFAULT_DISPLAY) if f in valid]
    # Only registry ids are computed directly; custom ids come from the engine.
    needed = {f for f in filter_fids if f in registry.REGISTRY}
    needed |= {f for f in display_fids if f in registry.REGISTRY}
    needed.add("market_cap")
    w = req.weighting
    if w.scheme == "factor_score" and w.score_factor_id in registry.REGISTRY:
        needed.add(w.score_factor_id)
    if w.scheme == "inverse_vol":
        needed.add("volatility")

    fm = await compute_factors(
        repo, members, as_of, sorted(needed), basis=basis, custom=custom_defs
    )

    # Screen (sequential AND) — filters may reference registry OR custom ids.
    filters = [
        Filter(f.factor_id, f.op, f.value, f.value2, f.k)
        for f in req.filters
        if f.factor_id in valid
    ]
    screen = apply_filters(fm, filters)
    survivors = screen.survivors

    # Weighting inputs from the computed matrix.
    market_cap = {s: fm.value(s, "market_cap") for s in survivors}
    score = (
        {s: fm.value(s, w.score_factor_id) for s in survivors}
        if w.scheme == "factor_score" and w.score_factor_id
        else None
    )
    score_dir = (
        registry.REGISTRY[w.score_factor_id].direction
        if w.scheme == "factor_score" and w.score_factor_id in registry.REGISTRY
        else custom_dir.get(w.score_factor_id, "higher_better")
    )
    volatility = {s: fm.value(s, "volatility") for s in survivors} if w.scheme == "inverse_vol" else None

    weights = compute_weights(
        survivors, scheme=w.scheme, market_cap=market_cap, score=score,
        score_direction=score_dir, volatility=volatility, sector=fm.sectors,
        max_weight=w.max_weight, max_sector_weight=w.max_sector_weight,
    )
    dropped = [s for s in survivors if s not in weights]
    if dropped:
        notes.append(
            f"{len(dropped)} screened name(s) lack the weighting input "
            f"({w.scheme}) and are excluded from the book."
        )

    meta = await repo.securities_meta(survivors)
    holdings = sorted(
        (
            HoldingRead(
                security_id=s,
                symbol=meta.get(s, (None, None, None))[0],
                name=meta.get(s, (None, None, None))[1],
                sector=fm.sectors.get(s),
                weight=weights[s],
                factors={fid: fm.value(s, fid) for fid in display_fids},
            )
            for s in survivors
            if s in weights
        ),
        key=lambda h: h.weight,
        reverse=True,
    )

    coverage = [
        CoverageRead(factor_id=c.factor_id, computable=c.computable, total=c.total)
        for c in fm.coverage
        if c.factor_id in (set(filter_fids) | set(display_fids))
    ]

    return ScreenResponse(
        as_of=as_of,
        universe=UniverseRead(**uni.__dict__),
        membership_count=len(members),
        basis=basis,
        weighting_scheme=w.scheme,
        holdings=holdings,
        funnel=[FunnelStepRead(label=s.label, remaining=s.remaining) for s in screen.funnel],
        coverage=coverage,
        dropped_no_weight=len(dropped),
        notes=notes,
    )


async def preview_factor(
    repo: PortfolioRepository, req: FactorPreviewRequest
) -> FactorPreviewResponse:
    """Live ranking of a factor (registry id OR inline custom expression) over a
    universe — powers the Factor Builder's "watch it rank real names" preview."""
    as_of = req.as_of or await repo.latest_trade_date()
    if as_of is None:
        raise ValueError("No price data available to anchor the as-of date.")

    custom_defs: list[CustomFactorDef] = []
    needed: list[str] = []
    if req.custom is not None:
        cf = req.custom
        custom_defs = [
            CustomFactorDef(
                id=cf.id, name=cf.name, expression=cf.expression,
                direction=cf.direction, normalization=cf.normalization,
            )
        ]
        factor_id = cf.id
    elif req.factor_id and req.factor_id in registry.REGISTRY:
        factor_id = req.factor_id
        needed = [factor_id]
    else:
        raise ValueError("Provide a valid factor_id or a custom expression.")

    members = await repo.members_as_of(req.index_id, as_of)
    fm = await compute_factors(
        repo, members, as_of, needed, basis=_resolve_basis(req.basis), custom=custom_defs
    )
    present = [(s, fm.value(s, factor_id)) for s in members if fm.value(s, factor_id) is not None]
    present.sort(key=lambda t: t[1], reverse=True)
    top = present[: req.limit]
    bottom = list(reversed(present[-req.limit :])) if present else []

    meta = await repo.securities_meta([s for s, _ in (top + bottom)])

    def _rows(pairs: list[tuple[int, float]]) -> list[FactorPreviewRow]:
        return [
            FactorPreviewRow(
                security_id=s,
                symbol=meta.get(s, (None, None, None))[0],
                name=meta.get(s, (None, None, None))[1],
                sector=fm.sectors.get(s),
                value=v,
            )
            for s, v in pairs
        ]

    cov = next((c for c in fm.coverage if c.factor_id == factor_id), None)
    return FactorPreviewResponse(
        as_of=as_of,
        factor_id=factor_id,
        computable=cov.computable if cov else len(present),
        total=len(members),
        top=_rows(top),
        bottom=_rows(bottom),
    )
