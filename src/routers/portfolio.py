"""Systematic Portfolio Builder — read APIs (`/api/v1/portfolio/*`).

Direct read-only reads against the investment RDS (like ``stocks.py``), with
PRISM's firm-auth/audit gate. Point-in-time correctness lives in
``src.portfolio`` (the 6-month annual-data lag + dated index membership).

  * ``GET  /api/v1/portfolio/universes``  — the 5 Nifty universes (the dropdown)
  * ``GET  /api/v1/portfolio/factors``    — the schema-derived factor catalog
  * ``POST /api/v1/portfolio/screen``     — the Suggested Portfolio (filters →
    weighted holdings + funnel + coverage)

If the investment DB isn't configured the session dependency raises and these
routes 503 — the rest of the app is unaffected.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.principal import Principal, get_current_principal
from src.core.auth import get_current_firm_id
from src.core.database import get_session
from src.core.investment_database import get_investment_session
from src.models.portfolio import PortfolioBacktest, PortfolioCustomFactor, PortfolioStrategy
from src.portfolio.factors.expression import ExpressionError
from src.portfolio.factors.expression import validate as validate_expression
from src.portfolio.factors.registry import REGISTRY
from src.portfolio.job_repo import JobRepository
from src.portfolio.repository import PortfolioRepository
from src.portfolio.serialize import strategy_hash
from src.portfolio.service import (
    backtest_request_to_spec_dict,
    factor_catalog,
    preview_factor,
    run_screen,
)
from src.portfolio.store import CustomFactorStore, StrategyStore
from src.schemas.portfolio import (
    BacktestJobRead,
    BacktestRequest,
    CustomFactorCreate,
    CustomFactorRead,
    ExpressionValidateRequest,
    ExpressionValidateResponse,
    FactorMetaRead,
    FactorPreviewRequest,
    FactorPreviewResponse,
    IndexSeriesResponse,
    ScreenRequest,
    ScreenResponse,
    StrategyCreate,
    StrategyRead,
    UniverseRead,
)

router = APIRouter(prefix="/portfolio", tags=["Portfolio Builder"])


def _job_read(job: PortfolioBacktest, *, include_result: bool = True) -> BacktestJobRead:
    return BacktestJobRead(
        id=job.id, name=job.name, status=job.status, progress=job.progress,
        stage=job.stage, error=job.error, spec=job.spec,
        created_at=job.created_at, started_at=job.started_at, finished_at=job.finished_at,
        result=job.result if include_result else None,
    )


@router.get("/universes", response_model=list[UniverseRead], summary="Selectable universes")
async def list_universes(
    session: Annotated[AsyncSession, Depends(get_investment_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> list[UniverseRead]:
    repo = PortfolioRepository(session)
    return [UniverseRead(**u.__dict__) for u in await repo.list_universes()]


@router.get("/factors", response_model=list[FactorMetaRead], summary="Factor catalog")
async def list_factors(
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> list[FactorMetaRead]:
    # Pure metadata — no DB needed (and no investment-DB dependency, so the
    # catalog is available even if the RDS is down).
    return factor_catalog()


@router.post("/screen", response_model=ScreenResponse, summary="Build the Suggested Portfolio")
async def screen(
    req: ScreenRequest,
    session: Annotated[AsyncSession, Depends(get_investment_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> ScreenResponse:
    repo = PortfolioRepository(session)
    try:
        return await run_screen(repo, req)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


# ── Backtest (async job; runs on the worker, reads the live RDS) ─────────────


@router.post("/backtest", response_model=BacktestJobRead, summary="Submit a backtest job")
async def submit_backtest(
    req: BacktestRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> BacktestJobRead:
    firm_id = principal.firm_id
    if req.start >= req.end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="start must be before end.",
        )
    spec = backtest_request_to_spec_dict(req)
    sh = strategy_hash(spec)
    repo = JobRepository(session)
    # Result cache: an identical spec already computed → return it instantly.
    cached = await repo.find_cached(firm_id, sh)
    if cached is not None:
        return _job_read(cached)
    job = await repo.create(
        firm_id=firm_id, name=req.name, spec=spec, strategy_hash=sh,
        created_by=principal.user_id,
    )
    return _job_read(job)            # the worker fills in progress/result


@router.get("/backtest/{job_id}", response_model=BacktestJobRead, summary="Backtest status / result")
async def get_backtest(
    job_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> BacktestJobRead:
    job = await JobRepository(session).get(firm_id, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Backtest not found.")
    return _job_read(job)


@router.get("/backtests", response_model=list[BacktestJobRead], summary="Recent backtests")
async def list_backtests(
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> list[BacktestJobRead]:
    jobs = await JobRepository(session).list_for_firm(firm_id)
    return [_job_read(j, include_result=False) for j in jobs]


@router.delete("/backtest/{job_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a backtest")
async def delete_backtest(
    job_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> None:
    if not await JobRepository(session).delete(firm_id, job_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Backtest not found.")


@router.get("/index-series", response_model=IndexSeriesResponse, summary="Benchmark index NAV series")
async def index_series(
    session: Annotated[AsyncSession, Depends(get_investment_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    index_id: Annotated[int, Query(description="indices_list.index_id")],
    start: Annotated[date, Query()],
    end: Annotated[date, Query()],
) -> IndexSeriesResponse:
    """Cumulative NAV (growth of ₹1) for an index — lets the NAV chart switch the
    benchmark to any universe without re-running the backtest."""
    repo = PortfolioRepository(session)
    series = await repo.benchmark_series(index_id, start, end)
    dates: list[str] = []
    nav: list[float] = []
    cum = 1.0
    for i, (d, r) in enumerate(series):
        if i > 0:
            cum *= 1.0 + (r or 0.0)
        dates.append(d.isoformat())
        nav.append(cum)
    name = next((u.index_name for u in await repo.list_universes() if u.index_id == index_id), None)
    return IndexSeriesResponse(index_id=index_id, index_name=name, dates=dates, nav=nav)


# ── Factor Builder: live preview ─────────────────────────────────────────────


@router.post("/factors/preview", response_model=FactorPreviewResponse, summary="Rank a factor over a universe")
async def factor_preview(
    req: FactorPreviewRequest,
    session: Annotated[AsyncSession, Depends(get_investment_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> FactorPreviewResponse:
    try:
        return await preview_factor(PortfolioRepository(session), req)
    except (ValueError, ExpressionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


# ── Custom factors (persisted; primary DB) ───────────────────────────────────


def _cf_read(cf: PortfolioCustomFactor) -> CustomFactorRead:
    return CustomFactorRead(
        id=cf.id, name=cf.name, expression=cf.expression, direction=cf.direction,
        normalization=cf.normalization, created_at=cf.created_at,
    )


@router.post("/custom-factors/validate", response_model=ExpressionValidateResponse, summary="Validate an expression")
async def validate_custom_factor(
    req: ExpressionValidateRequest,
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> ExpressionValidateResponse:
    try:
        refs = validate_expression(req.expression, set(REGISTRY))
        return ExpressionValidateResponse(ok=True, refs=sorted(refs))
    except ExpressionError as exc:
        return ExpressionValidateResponse(ok=False, error=str(exc))


@router.post("/custom-factors", response_model=CustomFactorRead, summary="Save a custom factor")
async def create_custom_factor(
    req: CustomFactorCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> CustomFactorRead:
    try:
        validate_expression(req.expression, set(REGISTRY))
    except ExpressionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    cf = await CustomFactorStore(session).create(
        firm_id=principal.firm_id, name=req.name, expression=req.expression,
        direction=req.direction, normalization=req.normalization,
        created_by=principal.user_id,
    )
    return _cf_read(cf)


@router.get("/custom-factors", response_model=list[CustomFactorRead], summary="List saved custom factors")
async def list_custom_factors(
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> list[CustomFactorRead]:
    return [_cf_read(c) for c in await CustomFactorStore(session).list(firm_id)]


@router.delete("/custom-factors/{cf_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a custom factor")
async def delete_custom_factor(
    cf_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> None:
    if not await CustomFactorStore(session).delete(firm_id, cf_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Custom factor not found.")


# ── Saved strategies (primary DB) ────────────────────────────────────────────


def _strategy_read(s: PortfolioStrategy) -> StrategyRead:
    return StrategyRead(
        id=s.id, name=s.name, config=s.config,
        created_at=s.created_at, updated_at=s.updated_at,
    )


@router.post("/strategies", response_model=StrategyRead, summary="Save a strategy")
async def create_strategy(
    req: StrategyCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> StrategyRead:
    s = await StrategyStore(session).create(
        firm_id=principal.firm_id, name=req.name, config=req.config,
        created_by=principal.user_id,
    )
    return _strategy_read(s)


@router.get("/strategies", response_model=list[StrategyRead], summary="List saved strategies")
async def list_strategies(
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> list[StrategyRead]:
    return [_strategy_read(s) for s in await StrategyStore(session).list(firm_id)]


@router.get("/strategies/{strategy_id}", response_model=StrategyRead, summary="Get a saved strategy")
async def get_strategy(
    strategy_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> StrategyRead:
    s = await StrategyStore(session).get(firm_id, strategy_id)
    if s is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found.")
    return _strategy_read(s)


@router.delete("/strategies/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a strategy")
async def delete_strategy(
    strategy_id: UUID,
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> None:
    if not await StrategyStore(session).delete(firm_id, strategy_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found.")
