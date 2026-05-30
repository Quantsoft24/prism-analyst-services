"""News router — thin proxy to the external prism-news service (``PRISM_NEWS_URL``).

The teammate-built financial-news service (showtimeapp/NewsRSS, FastAPI on
:8001) is the source of truth for headlines + per-company sentiment + trending.
This router preserves a stable ``/api/v1/news/*`` surface our frontend /news
page speaks, so the browser never talks to the upstream directly. Same
rationale as the BMC proxy:

  * Single CORS / auth surface (PRISM) — when real auth lands, news inherits it.
  * Single audit-log surface — news calls show up in PRISM's logs.
  * No new env var leaked into the browser bundle (the upstream URL stays
    server-side; the frontend only knows ``/api/v1/news``).

Endpoints forwarded (all GET — the upstream is read-only from a client's view):

  GET /news/feed         → upstream /news            (headline feed)
  GET /news/summary      → upstream /news/summary     (per-company verdict)
  GET /news/trending     → upstream /news/trending    (most-mentioned)
  GET /news/compare      → upstream /news/compare      (multi-company)
  GET /news/sources      → upstream /news/sources      (source reliability)
  GET /news/companies    → upstream /news/companies    (directory)
  GET /news/sectors      → upstream /news/sectors      (8 sector codes)
  GET /news/stats        → upstream /stats             (24h rollups)
  GET /news/health       → upstream /health            (ops heartbeat)

The agent reaches the same service through the typed wrapper
(src/integrations/tools/prism_news.py); this router is purely for the UI.
"""

from __future__ import annotations

from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.config import settings
from src.core.auth import get_current_firm_id

router = APIRouter(prefix="/news", tags=["News & Sentiment"])

# Company-sentiment endpoints fire cold OpenAI scoring on first request for a
# fresh company+window. Live measurement showed the cold /news/summary path
# running past 20s while scoring ~24 articles (then it caches → sub-second).
# 40s ceiling keeps watchlist cards / summary calls from 502-ing on a cold
# company; once cached they're instant. List / directory calls stay fast.
_SENTIMENT_TIMEOUT = 40.0
_FAST_TIMEOUT = 15.0
_MAX_HOURS = 240


def _news_url(path: str) -> str:
    return f"{settings.PRISM_NEWS_URL.rstrip('/')}{path}"


def _auth_headers() -> dict[str, str]:
    key = settings.PRISM_NEWS_API_KEY
    return {"X-API-Key": key} if key else {}


def _clamp_hours(hours: int) -> int:
    if hours <= 0:
        return 24
    return min(hours, _MAX_HOURS)


async def _forward(
    path: str, *, params: dict[str, Any], timeout: float,
) -> Any:
    """Call the news service and forward the JSON response. Upstream status
    codes propagate verbatim so the frontend distinguishes 404 / 422 / 5xx the
    same way it would talking directly to the service. Drops None-valued params
    so the upstream sees clean query strings."""
    clean = {k: v for k, v in params.items() if v is not None}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                _news_url(path), params=clean, headers=_auth_headers(),
            )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"News service unreachable: {exc}",
        ) from exc
    if resp.status_code >= 400:
        detail: Any
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)
    try:
        return resp.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="News service returned a non-JSON response.",
        ) from exc


# ── Routes ────────────────────────────────────────────────────────────────────
# `firm_id` is resolved (dev-mode default today) so news calls share PRISM's
# auth surface and show in the audit log, even though the upstream itself is
# firm-agnostic. We don't forward it upstream (the service has no per-firm
# concept) — it just gates access at our layer.


@router.get("/feed", summary="Headline feed; filter by company / sector")
async def news_feed(
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    company: str | None = Query(None, description="CSV company name(s); aliases ok"),
    sector: str | None = Query(
        None, pattern="^(BANKING|TECH|AUTO|PHARMA|ENERGY|FMCG|METALS|REALTY)$"
    ),
    hours: int = Query(24, ge=1, le=_MAX_HOURS),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
) -> Any:
    return await _forward(
        "/news",
        params={
            "company": company,
            "sector": sector,
            "hours": _clamp_hours(hours),
            "page": page,
            "limit": limit,
            "fuzzy": "true",
            "resolve_links": "true",
        },
        timeout=_SENTIMENT_TIMEOUT,  # company filter can warm cold sentiment
    )


@router.get("/summary", summary="Per-company sentiment verdict + trend")
async def news_summary(
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    company: str = Query(..., min_length=1, description="Single company; aliases ok"),
    hours: int = Query(24, ge=1, le=_MAX_HOURS),
) -> Any:
    return await _forward(
        "/news/summary",
        params={"company": company, "hours": _clamp_hours(hours)},
        timeout=_SENTIMENT_TIMEOUT,
    )


@router.get("/trending", summary="Most-mentioned companies in the window")
async def news_trending(
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    hours: int = Query(24, ge=1, le=_MAX_HOURS),
    limit: int = Query(20, ge=1, le=50),
) -> Any:
    return await _forward(
        "/news/trending",
        params={"hours": _clamp_hours(hours), "limit": limit},
        timeout=_FAST_TIMEOUT,
    )


@router.get("/compare", summary="Multi-company side-by-side sentiment")
async def news_compare(
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    companies: str = Query(..., min_length=1, description="CSV, >=1 company"),
    hours: int = Query(48, ge=1, le=_MAX_HOURS),
) -> Any:
    return await _forward(
        "/news/compare",
        params={"companies": companies, "hours": _clamp_hours(hours)},
        timeout=_SENTIMENT_TIMEOUT,
    )


@router.get("/sources", summary="Per-source reliability stats")
async def news_sources(
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    hours: int = Query(24, ge=1, le=_MAX_HOURS),
) -> Any:
    return await _forward(
        "/news/sources", params={"hours": _clamp_hours(hours)}, timeout=_FAST_TIMEOUT
    )


@router.get("/companies", summary="Canonical company directory (for dropdowns)")
async def news_companies(
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> Any:
    return await _forward("/news/companies", params={}, timeout=_FAST_TIMEOUT)


@router.get("/sectors", summary="The 8 sector chip codes")
async def news_sectors(
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> Any:
    return await _forward("/news/sectors", params={}, timeout=_FAST_TIMEOUT)


@router.get("/stats", summary="Last-24h source / sentiment / sector rollups")
async def news_stats(
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> Any:
    return await _forward("/stats", params={}, timeout=_FAST_TIMEOUT)


@router.get("/health", summary="News service ops heartbeat")
async def news_health(
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> Any:
    return await _forward("/health", params={}, timeout=_FAST_TIMEOUT)
