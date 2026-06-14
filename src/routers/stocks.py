"""Stocks — NSE/BSE security master + daily price series (Stock Dashboard).

Direct read-only reads against the investment RDS (like ``companies.py``, NOT
an httpx proxy like ``news.py``). Powers the frontend Stock Dashboard:

  * ``GET /api/v1/stocks/securities``            — the full search index
  * ``GET /api/v1/stocks/indices/latest``        — index levels + day move (landing)
  * ``GET /api/v1/stocks/movers``                — top gainers/losers/most-active
  * ``GET /api/v1/stocks/{security_id}``         — one security's master detail
  * ``GET /api/v1/stocks/{security_id}/prices``  — daily OHLC/volume/value/mcap
  * ``GET /api/v1/stocks/{security_id}/balance-sheet`` / ``/income-statement``

If the investment DB isn't configured the dependency raises and these routes
503 — the rest of the app is unaffected.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.core.auth import get_current_firm_id
from src.core.investment_database import get_investment_session
from src.models.investment import PriceRow
from src.repositories.stock_repo import StockRepository
from src.schemas.stock import (
    BalanceSheetResponse,
    FinancialBasis,
    IncomeStatementResponse,
    IndexLatest,
    MoverKind,
    MoverRow,
    MoversResponse,
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


# ── Reports Viewer — thin proxy to the stock-chat filings service ───────────
# Keeps the browser off the internal HTTP-only stock-chat service (mixed
# content over HTTPS / CORS / no caller auth) and on PRISM's own HTTPS API.
# Same rationale as the news + BMC proxies. Catalog-only listing → fast.
# Declared BEFORE /{security_id} so "reports" isn't matched as an int path.
_REPORTS_TIMEOUT = 20.0


@router.get(
    "/reports",
    summary="List a company's filings by category (Reports Viewer)",
    description=(
        "Thin proxy to the stock-chat service's chronological catalog listing. "
        "``category`` is one of: Annual Report, Result, Board Meeting, AGM/EGM, "
        "Corp. Action, Company Update, Insider Trading / SAST, Others. Returns "
        "the upstream JSON (resolved_company, total, filings[]). An unmatched "
        "company comes back with ``resolved_company: null`` and no filings."
    ),
)
async def list_reports(
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    company: Annotated[str, Query(min_length=1, description="Company name (resolved upstream).")],
    category: Annotated[str, Query(min_length=1, description="Filing category (exact).")],
    limit: Annotated[int, Query(ge=1, le=500)] = 25,
    offset: Annotated[int, Query(ge=0)] = 0,
    order: Annotated[Literal["desc", "asc"], Query()] = "desc",
) -> Any:
    _ = firm_id  # auth-gated only; the upstream is firm-agnostic
    body = {
        "company": company,
        "category": category,
        "limit": limit,
        "offset": offset,
        "order": order,
    }
    url = f"{settings.STOCK_CHAT_URL.rstrip('/')}/tools/list-by-category"
    # Upstream is POST today; flips to GET later (then: client.get(url, params=body)).
    try:
        async with httpx.AsyncClient(timeout=_REPORTS_TIMEOUT) as client:
            resp = await client.post(url, json=body)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Filings service unreachable: {exc}",
        ) from exc
    if resp.status_code >= 400:
        try:
            detail: Any = resp.json().get("detail", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)
    try:
        return resp.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Filings service returned a non-JSON response.",
        ) from exc


# ── Filing-PDF proxy (citation → exact-page deep link) ──────────────────────
# BSE/NSE serve filing PDFs with CORS / X-Frame-Options that block embedding
# them directly in our Workspace. We stream them through PRISM so the Report-tab
# viewer can open `…/reports/pdf?url=<bse pdf>#page=N` same-origin. SSRF-guarded:
# only the exchange hosts are allowed (the URL comes from stock-chat's catalog).
_PDF_PROXY_TIMEOUT = 30.0
_PDF_ALLOWED_HOSTS = ("bseindia.com", "nseindia.com")


@router.get(
    "/reports/pdf",
    summary="Stream a filing PDF (embeddable, for the citation deep-link viewer)",
)
async def report_pdf(
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    url: Annotated[str, Query(min_length=8, description="BSE/NSE filing PDF URL.")],
) -> StreamingResponse:
    _ = firm_id  # auth-gated; the PDFs are public exchange filings
    host = (urlparse(url).hostname or "").lower()
    if not any(host == h or host.endswith("." + h) for h in _PDF_ALLOWED_HOSTS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only BSE/NSE filing URLs can be proxied.",
        )
    client = httpx.AsyncClient(timeout=_PDF_PROXY_TIMEOUT, follow_redirects=True)
    try:
        req = client.build_request(
            "GET", url, headers={"User-Agent": "Mozilla/5.0 (PRISM filings viewer)"}
        )
        resp = await client.send(req, stream=True)
    except httpx.RequestError as exc:
        await client.aclose()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not fetch the filing PDF: {exc}",
        ) from exc
    if resp.status_code != 200:
        await resp.aclose()
        await client.aclose()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"The filing host returned HTTP {resp.status_code}.",
        )

    async def _stream():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        _stream(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "private, max-age=3600",
        },
    )


_FILINGS_TIMEOUT = 10.0
# Strip the trailing legal-form suffix from a security_name before matching
# prism-filings' EXACT-match ``company`` param — its canonical names usually
# carry none ("HDFC Bank Ltd." misses; "HDFC Bank" hits). Deliberately narrow:
# only ``Ltd/Limited/Pvt/Private`` (with an optional trailing dot), which is
# safe to drop on Indian listed names. We do NOT strip Corp/Inc/LLP (real name
# words → over-stripping) nor a trailing "-$" prowess artifact (sometimes part
# of the upstream's actual tag, e.g. "Baba Arts Ltd-$").
_CORP_SUFFIX_RE = re.compile(r"[\s,]*\b(?:ltd|limited|pvt|private)\b\.?\s*$", re.IGNORECASE)


def _canonical_company(name: str) -> str:
    """``"HDFC Bank Ltd."`` → ``"HDFC Bank"`` for prism-filings' exact match.

    Iteratively strips a trailing ``Ltd/Limited/Pvt/Private`` (handles stacked
    forms like ``"… Pvt. Ltd."``). Idempotent; safe on already-clean names.
    """
    prev = None
    out = name.strip()
    while out and out != prev:
        prev = out
        out = _CORP_SUFFIX_RE.sub("", out).strip()
    return out or name.strip()


@router.get(
    "/announcements",
    summary="A company's regulatory announcements (Announcements pane)",
    description=(
        "Thin proxy to the prism-filings service's /filings query, scoped to one "
        "company. ``company`` (the dashboard's security_name) is normalised to "
        "prism-filings' canonical name (its match is exact, suffix-sensitive). "
        "Optional ``regulator`` (RBI/SEBI/BSE/NSE/PIB) and ``filing_type`` "
        "(category) narrow the feed; ``hours`` is the lookback (≤ 720 = 30d). "
        "Returns the upstream JSON ({success, query, meta, filings[]}). A company "
        "with no tagged filings comes back 200 with an empty ``filings`` array."
    ),
)
async def list_announcements(
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    company: Annotated[str, Query(min_length=1, description="Company name (security_name).")],
    regulator: Annotated[str | None, Query(description="RBI|SEBI|BSE|NSE|PIB.")] = None,
    filing_type: Annotated[str | None, Query(description="Filing category (exact).")] = None,
    hours: Annotated[int, Query(ge=1, le=720)] = 720,
    page: Annotated[int, Query(ge=1)] = 1,
    limit: Annotated[int, Query(ge=1, le=100)] = 12,
) -> Any:
    _ = firm_id  # auth-gated only; the upstream is firm-agnostic
    params: dict[str, Any] = {
        "company": _canonical_company(company),
        "hours": hours,
        "page": page,
        "limit": limit,
    }
    if regulator:
        params["regulator"] = regulator
    if filing_type:
        params["filing_type"] = filing_type
    url = f"{settings.PRISM_FILINGS_URL.rstrip('/')}/filings"
    try:
        async with httpx.AsyncClient(timeout=_FILINGS_TIMEOUT) as client:
            resp = await client.get(url, params=params)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Filings service unreachable: {exc}",
        ) from exc
    if resp.status_code >= 400:
        try:
            detail: Any = resp.json().get("detail", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)
    try:
        return resp.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Filings service returned a non-JSON response.",
        ) from exc


# ── Market overview (landing) ───────────────────────────────────────────────
# Declared BEFORE /{security_id} so "indices" / "movers" aren't matched as an
# int security_id. Both read straight from the investment DB (cached in-repo).


@router.get(
    "/indices/latest",
    response_model=list[IndexLatest],
    summary="Latest level + day move + sparkline for each index (landing strip)",
    description=(
        "One row per index in ``indices_list`` (the 5 NSE universes: Nifty 50 / "
        "Next 50 / 100 / 200 / 500). Each carries the latest close (level), the "
        "day-over-day % change, and a short ``spark`` array of recent closes "
        "(oldest → newest) for an inline sparkline."
    ),
)
async def indices_latest(
    session: Annotated[AsyncSession, Depends(get_investment_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    spark_days: Annotated[int, Query(ge=2, le=120, description="Sparkline length.")] = 30,
) -> list[IndexLatest]:
    _ = firm_id
    repo = StockRepository(session)
    return await repo.get_indices_latest(spark_days)


@router.get(
    "/movers",
    response_model=MoversResponse,
    summary="Top gainers / losers / most-active (Nifty 200 universe)",
    description=(
        "Latest-trading-day movers computed over the current Nifty 200 "
        "constituents — restricting the universe keeps the scan fast and the "
        "list institutionally relevant (no illiquid penny stocks). ``kind`` is "
        "``gainers`` | ``losers`` | ``most_active``. Computed once per ~30-min "
        "window and cached (prices are end-of-day)."
    ),
)
async def stock_movers(
    session: Annotated[AsyncSession, Depends(get_investment_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    kind: Annotated[MoverKind, Query(description="gainers | losers | most_active")] = "gainers",
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> MoversResponse:
    _ = firm_id
    repo = StockRepository(session)
    universe, trade_date, movers = await repo.get_movers(kind, limit)
    return MoversResponse(kind=kind, universe=universe, trade_date=trade_date, movers=movers)


@router.get(
    "/top-companies",
    response_model=list[MoverRow],
    summary="Largest companies by market cap (Nifty 200 universe)",
    description=(
        "The biggest Nifty 200 constituents by latest market cap — powers the "
        "BMC dashboard's 'suggested to build' list. Reuses the cached movers "
        "universe (no extra scan); each row carries security_id, symbol, "
        "security_name, sector, and market_cap (₹ crore). Declared before "
        "``/{security_id}`` so the literal isn't matched as an int path."
    ),
)
async def top_companies(
    session: Annotated[AsyncSession, Depends(get_investment_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    limit: Annotated[int, Query(ge=1, le=50)] = 12,
) -> list[MoverRow]:
    _ = firm_id
    repo = StockRepository(session)
    return await repo.get_top_companies(limit)


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


@router.get(
    "/{security_id}/income-statement",
    response_model=IncomeStatementResponse,
    summary="Annual income statement (sequential) over the last ~10 fiscal years",
    description=(
        "Sequential P&L (Revenue → Operating Profit → PBT → PAT) with one column "
        "per fiscal year, values in ₹ crore. Computed subtotals carry "
        "``emphasis`` + an ``info`` formula tooltip. ``basis`` selects "
        "standalone vs consolidated and falls back to whichever is available."
    ),
    responses={404: {"description": "Security not found."}},
)
async def get_income_statement(
    security_id: int,
    session: Annotated[AsyncSession, Depends(get_investment_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    basis: Annotated[FinancialBasis, Query(description="standalone | consolidated")] = "consolidated",
) -> IncomeStatementResponse:
    _ = firm_id
    repo = StockRepository(session)
    sec = await repo.get_security(security_id)
    if sec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Security {security_id} not found.",
        )
    return await repo.get_income_statement(security_id, basis)
