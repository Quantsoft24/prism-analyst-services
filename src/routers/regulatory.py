"""Regulatory Lens — SEBI regulatory intelligence (circulars, regulations,
orders, deadlines, weekly digest).

Direct read-only reads against the SEBI Postgres via ``SebiRepository`` (like
``stocks.py``/``companies.py``, NOT an httpx proxy like ``news.py``). Powers the
frontend Regulatory Lens views:

  * ``GET /api/v1/regulatory/stats``            — dashboard headline stats
  * ``GET /api/v1/regulatory/feed``             — paginated + filterable content
  * ``GET /api/v1/regulatory/content/{id}``     — one document, full detail
  * ``GET /api/v1/regulatory/recent``           — latest N (dashboard)
  * ``GET /api/v1/regulatory/deadlines``        — upcoming compliance deadlines
  * ``GET /api/v1/regulatory/weekly-summary``   — AI weekly digests
  * ``GET /api/v1/regulatory/topics`` / ``/types`` — filter aggregations

If the SEBI DB isn't configured the session dependency raises and these routes
503 — the rest of the app is unaffected.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.principal import Principal, get_current_principal
from src.core.auth import get_current_firm_id
from src.core.database import get_session
from src.core.sebi_database import get_sebi_session, is_sebi_configured
from src.repositories.preferences_repo import PreferencesRepository
from src.repositories.sebi_repo import SebiRepository
from src.schemas.regulatory import (
    AlertsResponse,
    CalendarResponse,
    DeadlinesResponse,
    FeedResponse,
    RegAlert,
    RegDocDetail,
    RegDocSummary,
    RegPersonalization,
    RegStats,
    TopicCount,
    TypeCount,
    WeeklySummariesResponse,
)

router = APIRouter(prefix="/regulatory", tags=["Regulatory Lens"])

FirmDep = Annotated[str, Depends(get_current_firm_id)]
SessionDep = Annotated[AsyncSession, Depends(get_sebi_session)]
PrincipalDep = Annotated[Principal, Depends(get_current_principal)]
PrimaryDep = Annotated[AsyncSession, Depends(get_session)]

# Regulatory personalization lives under this key in user_preferences.prefs.
_PREFS_KEY = "regulatory"


async def _load_personalization(
    principal: Principal, primary: AsyncSession
) -> RegPersonalization:
    """The user's Regulatory state, or defaults when there's no user (dev/anon)."""
    if principal.user_id is None:
        return RegPersonalization()
    prefs = await PreferencesRepository(primary).get(principal.user_id)
    blob = prefs.get(_PREFS_KEY) if isinstance(prefs, dict) else None
    if not isinstance(blob, dict):
        return RegPersonalization()
    try:
        return RegPersonalization.model_validate(blob)
    except Exception:  # noqa: BLE001 — tolerate older/dirty blobs
        return RegPersonalization()


@router.get("/health", summary="Regulatory Lens health / connectivity")
async def health() -> dict:
    """Reports whether the SEBI DB is configured + reachable. No auth (used by
    ops); returns ``configured: false`` instead of erroring when unset."""
    if not is_sebi_configured():
        return {"status": "unconfigured", "configured": False}
    try:
        async for session in get_sebi_session():
            total = (
                await session.execute(text("SELECT COUNT(*) FROM content"))
            ).scalar_one()
            return {"status": "ok", "configured": True, "total_documents": int(total)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "configured": True, "detail": str(exc)}
    return {"status": "error", "configured": True}


@router.get("/stats", response_model=RegStats, summary="Dashboard headline stats")
async def stats(firm_id: FirmDep, session: SessionDep, response: Response) -> RegStats:
    _ = firm_id
    repo = SebiRepository(session)
    response.headers["Cache-Control"] = "public, max-age=600"
    return RegStats(**await repo.stats())


@router.get(
    "/feed",
    response_model=FeedResponse,
    summary="Paginated, filterable SEBI content feed",
)
async def feed(
    firm_id: FirmDep,
    session: SessionDep,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    type: str | None = Query(None, description="Content type, e.g. CIRCULAR/ORDER"),
    severity: str | None = Query(None, description="High/Medium/Low"),
    intent: str | None = Query(None),
    action_required: bool | None = Query(None),
    topic: str | None = Query(None),
    search: str | None = Query(None, description="Title/summary/tags substring"),
    date_from: date | None = Query(None),
    date_to: date | None = Query(None),
) -> FeedResponse:
    _ = firm_id
    repo = SebiRepository(session)
    filters = {
        "type": type,
        "severity": severity,
        "intent": intent,
        "action_required": action_required,
        "topic": topic,
        "search": search,
        "date_from": date_from,
        "date_to": date_to,
    }
    items, total = await repo.feed(filters, page, limit)
    total_pages = (total + limit - 1) // limit if total else 0
    return FeedResponse(
        items=items, total=total, page=page, limit=limit, total_pages=total_pages
    )


@router.get(
    "/recent",
    response_model=list[RegDocSummary],
    summary="Latest documents (dashboard)",
)
async def recent(
    firm_id: FirmDep,
    session: SessionDep,
    limit: int = Query(10, ge=1, le=50),
) -> list[RegDocSummary]:
    _ = firm_id
    return await SebiRepository(session).recent(limit)


@router.get(
    "/deadlines",
    response_model=DeadlinesResponse,
    summary="Upcoming compliance deadlines (from ai_tags.deadlines)",
)
async def deadlines(
    firm_id: FirmDep,
    session: SessionDep,
    limit: int = Query(30, ge=1, le=500),
) -> DeadlinesResponse:
    _ = firm_id
    items = await SebiRepository(session).deadlines(limit)
    return DeadlinesResponse(items=items, total=len(items))


@router.get(
    "/calendar",
    response_model=CalendarResponse,
    summary="Calendar events in a date range (deadlines + board meetings)",
)
async def calendar(
    firm_id: FirmDep,
    session: SessionDep,
    start: date = Query(..., description="Range start (YYYY-MM-DD), inclusive"),
    end: date = Query(..., description="Range end (YYYY-MM-DD), inclusive"),
) -> CalendarResponse:
    _ = firm_id
    if (end - start).days > 366:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Calendar range cannot exceed 1 year.",
        )
    events = await SebiRepository(session).calendar_events(start, end)
    return CalendarResponse(events=events)


@router.get(
    "/weekly-summary",
    response_model=WeeklySummariesResponse,
    summary="AI-generated weekly digests",
)
async def weekly_summary(
    firm_id: FirmDep,
    session: SessionDep,
    limit: int = Query(8, ge=1, le=52),
) -> WeeklySummariesResponse:
    _ = firm_id
    items = await SebiRepository(session).weekly_summaries(limit)
    return WeeklySummariesResponse(items=items)


@router.get("/topics", response_model=list[TopicCount], summary="Topic aggregation")
async def topics(
    firm_id: FirmDep,
    session: SessionDep,
    response: Response,
    limit: int = Query(50, ge=1, le=200),
) -> list[TopicCount]:
    _ = firm_id
    response.headers["Cache-Control"] = "public, max-age=3600"
    return await SebiRepository(session).topics(limit)


@router.get("/types", response_model=list[TypeCount], summary="Content-type counts")
async def types(firm_id: FirmDep, session: SessionDep, response: Response) -> list[TypeCount]:
    _ = firm_id
    response.headers["Cache-Control"] = "public, max-age=3600"
    return await SebiRepository(session).types()


# ── Per-user personalization (bookmarks, tracked terms, alert rules) ─────────


@router.get(
    "/me",
    response_model=RegPersonalization,
    summary="The signed-in user's Regulatory Lens state",
)
async def get_personalization(
    principal: PrincipalDep, primary: PrimaryDep
) -> RegPersonalization:
    """Returns the user's bookmarks / tracked terms / alert rules. Falls back to
    empty defaults when there's no user identity (dev / anonymous)."""
    return await _load_personalization(principal, primary)


@router.put(
    "/me",
    response_model=RegPersonalization,
    summary="Replace the signed-in user's Regulatory Lens state",
)
async def put_personalization(
    body: RegPersonalization, principal: PrincipalDep, primary: PrimaryDep
) -> RegPersonalization:
    if principal.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to save your Regulatory Lens watchlist.",
        )
    # Defensive caps.
    body.bookmarks = body.bookmarks[:500]
    body.tracked = body.tracked[:50]
    repo = PreferencesRepository(primary)
    current = await repo.get(principal.user_id)
    current = dict(current) if isinstance(current, dict) else {}
    current[_PREFS_KEY] = body.model_dump()
    await repo.upsert(principal.user_id, current)
    return body


@router.get(
    "/alerts",
    response_model=AlertsResponse,
    summary="Recent documents matching the user's tracked topics/entities",
)
async def get_alerts(
    principal: PrincipalDep,
    primary: PrimaryDep,
    session: SessionDep,
    limit: int = Query(20, ge=1, le=50),
) -> AlertsResponse:
    pers = await _load_personalization(principal, primary)
    terms = [t.term for t in pers.tracked]
    if not terms:
        return AlertsResponse(items=[], total=0)
    docs = await SebiRepository(session).search_any(terms, limit)
    lowered = [(t, t.lower()) for t in terms]
    items: list[RegAlert] = []
    for d in docs:
        hay = f"{d.title} {d.summary or ''} {' '.join(d.ai_tags.topics)} {' '.join(d.ai_tags.stakeholders)}".lower()
        matched = next((orig for orig, low in lowered if low in hay), None)
        items.append(RegAlert(**d.model_dump(), matched_term=matched))
    return AlertsResponse(items=items, total=len(items))


@router.get(
    "/bookmarks",
    response_model=list[RegDocSummary],
    summary="The user's bookmarked documents (resolved to summaries)",
)
async def get_bookmarks(
    principal: PrincipalDep, primary: PrimaryDep, session: SessionDep
) -> list[RegDocSummary]:
    pers = await _load_personalization(principal, primary)
    return await SebiRepository(session).by_ids(pers.bookmarks)


@router.get(
    "/content/{doc_id}",
    response_model=RegDocDetail,
    summary="One document — full detail (body, metadata, impact tags)",
)
async def get_document(
    doc_id: int, firm_id: FirmDep, session: SessionDep
) -> RegDocDetail:
    _ = firm_id
    doc = await SebiRepository(session).get_doc(doc_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No regulatory document with id {doc_id}.",
        )
    return doc
