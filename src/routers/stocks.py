"""Stocks — NSE/BSE security master + daily price series (Stock Dashboard).

Direct read-only reads against the investment RDS (like ``companies.py``, NOT
an httpx proxy like ``news.py``). Powers the frontend Stock Dashboard:

  * ``GET /api/v1/stocks/securities``            — the full search index
  * ``GET /api/v1/stocks/{security_id}``         — one security's master detail
  * ``GET /api/v1/stocks/{security_id}/prices``  — daily OHLC/volume/value/mcap

If the investment DB isn't configured the dependency raises and these routes
503 — the rest of the app is unaffected.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.auth import get_current_firm_id
from src.core.investment_database import get_investment_session
from src.models.investment import PriceRow
from src.repositories.stock_repo import StockRepository
from src.schemas.stock import (
    BalanceSheetResponse,
    FinancialBasis,
    PricePoint,
    PriceSeriesResponse,
    SecurityDetail,
    SecurityRead,
    StockRange,
)

router = APIRouter(prefix="/stocks", tags=["Stocks"])


def _to_point(r: PriceRow) -> PricePoint:
    """Map a DB bar to the wire schema (Decimal → float for the chart)."""
    return PricePoint(
        time=r.trade_date,
        open=float(r.open) if r.open is not None else None,
        high=float(r.high) if r.high is not None else None,
        low=float(r.low) if r.low is not None else None,
        close=float(r.close) if r.close is not None else None,
        trade_volume=r.trade_volume,
        trade_value=float(r.trade_value) if r.trade_value is not None else None,
        market_cap=float(r.market_cap) if r.market_cap is not None else None,
    )


@router.get(
    "/securities",
    response_model=list[SecurityRead],
    summary="Full NSE/BSE security search index (8,230 entries)",
    description=(
        "Lightweight list of every security (security_id, name, symbol, ISIN, "
        "exchange, sector). The frontend fetches this ONCE and filters it "
        "in-memory for instant search suggestions. Dual-listed companies appear "
        "twice — once per exchange, with distinct security_ids."
    ),
)
async def list_securities(
    session: Annotated[AsyncSession, Depends(get_investment_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    response: Response,
) -> list[SecurityRead]:
    _ = firm_id  # auth-gated only; the master is global
    repo = StockRepository(session)
    items = await repo.list_securities()
    # Rarely changes — let the browser/CDN cache it for an hour.
    response.headers["Cache-Control"] = "public, max-age=3600"
    return items


@router.get(
    "/{security_id}",
    response_model=SecurityDetail,
    summary="One security's master detail (dashboard header)",
    responses={404: {"description": "Security not found."}},
)
async def get_security(
    security_id: int,
    session: Annotated[AsyncSession, Depends(get_investment_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> SecurityDetail:
    _ = firm_id
    repo = StockRepository(session)
    sec = await repo.get_security(security_id)
    if sec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Security {security_id} not found.",
        )
    return SecurityDetail.model_validate(sec)


@router.get(
    "/{security_id}/prices",
    response_model=PriceSeriesResponse,
    summary="Daily price series for a security over a time range",
    description=(
        "Returns the security's master detail, its latest bar, and the daily "
        "series (ascending by trade_date) for the requested range. Ranges: 5D "
        "(last 5 rows), 1M/6M/1Y/3Y/5Y (calendar window anchored at the "
        "security's latest trade date), MAX (full history)."
    ),
    responses={404: {"description": "Security not found."}},
)
async def get_security_prices(
    security_id: int,
    session: Annotated[AsyncSession, Depends(get_investment_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    range: Annotated[StockRange, Query(description="Time window.")] = "1M",
) -> PriceSeriesResponse:
    _ = firm_id
    repo = StockRepository(session)
    sec = await repo.get_security(security_id)
    if sec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Security {security_id} not found.",
        )
    rows = await repo.get_price_series(security_id, range)
    points = [_to_point(r) for r in rows]
    return PriceSeriesResponse(
        security=SecurityDetail.model_validate(sec),
        range=range,
        latest=points[-1] if points else None,
        points=points,
    )


@router.get(
    "/{security_id}/balance-sheet",
    response_model=BalanceSheetResponse,
    summary="Annual balance sheet (tree) over the last ~10 fiscal years",
    description=(
        "Tree-structured balance sheet (Total assets + Capital & Liabilities) "
        "with one column per fiscal year, values in ₹ crore. ``basis`` selects "
        "standalone vs consolidated and falls back to whichever is available. "
        "Empty branches are pruned for the security."
    ),
    responses={404: {"description": "Security not found."}},
)
async def get_balance_sheet(
    security_id: int,
    session: Annotated[AsyncSession, Depends(get_investment_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    basis: Annotated[FinancialBasis, Query(description="standalone | consolidated")] = "consolidated",
) -> BalanceSheetResponse:
    _ = firm_id
    repo = StockRepository(session)
    sec = await repo.get_security(security_id)
    if sec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Security {security_id} not found.",
        )
    return await repo.get_balance_sheet(security_id, basis)
