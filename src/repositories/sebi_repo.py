"""Read-only data access for the SEBI DB (Regulatory Lens).

Raw parametrized ``text()`` SELECTs against a single external ``content`` table
(plus ``weekly_summaries`` / ``insight_feed``). We deliberately avoid ORM models:
the table is owned externally, has a ``json`` ``ai_tags`` column, and we only
ever read. Every method is SELECT-only.

``ai_tags`` is Postgres ``json`` (not ``jsonb``) — ``->``/``->>`` and
``json_array_elements_text`` all work; ``::text`` casts power ILIKE search.
asyncpg returns ``json`` columns as strings, so we ``json.loads`` them via
``_parse_ai_tags``.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.schemas.regulatory import (
    AiTags,
    CalendarEvent,
    DeadlineItem,
    IntentCount,
    RegDocDetail,
    RegDocSummary,
    SeverityCount,
    TopicCount,
    TypeCount,
    WeeklySummary,
)

# Columns selected for compact list rows vs. the full detail row.
_SUMMARY_COLS = (
    "id, type, sub_type, title, date, summary, sebi_id, sebi_department, ai_tags"
)
_DETAIL_COLS = (
    "id, type, sub_type, title, date, summary, sebi_id, sebi_department, "
    "sebi_url, sebi_section, sebi_sub_section, sebi_info_for, meeting_date, "
    "extracted_text, language, related_content_id, ai_tags, created_at, updated_at"
)

# Guard so json_array_elements_text never sees a non-array (would raise).
def _arr(col: str) -> str:
    return (
        f"json_array_elements_text(CASE WHEN json_typeof({col})='array' "
        f"THEN {col} ELSE '[]'::json END)"
    )


def _parse_ai_tags(raw: Any) -> AiTags:
    """Coerce the ``ai_tags`` json (str | dict | None) into a typed ``AiTags``."""
    if raw is None:
        return AiTags()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return AiTags()
    if not isinstance(raw, dict):
        return AiTags()

    def _list(v: Any) -> list[str]:
        return [str(x) for x in v] if isinstance(v, list) else []

    return AiTags(
        intent=raw.get("intent"),
        topics=_list(raw.get("topics")),
        stakeholders=_list(raw.get("stakeholders")),
        severity=raw.get("severity"),
        action_required=bool(raw.get("action_required")),
        deadlines=_list(raw.get("deadlines")),
    )


def _summary(row: Any) -> RegDocSummary:
    return RegDocSummary(
        id=row.id,
        type=row.type,
        sub_type=row.sub_type,
        title=row.title,
        date=row.date,
        summary=row.summary,
        sebi_id=row.sebi_id,
        sebi_department=row.sebi_department,
        ai_tags=_parse_ai_tags(row.ai_tags),
    )


class SebiRepository:
    """All read-only queries for Regulatory Lens. One instance per request."""

    def __init__(self, session: AsyncSession) -> None:
        self.s = session

    # ── Feed / search ──────────────────────────────────────────────────────
    def _build_filters(self, f: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Build a dynamic WHERE clause + bound params from the filter dict."""
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if f.get("type"):
            clauses.append("type = :type")
            params["type"] = f["type"]
        if f.get("severity"):
            clauses.append("ai_tags->>'severity' = :severity")
            params["severity"] = f["severity"]
        if f.get("intent"):
            clauses.append("ai_tags->>'intent' = :intent")
            params["intent"] = f["intent"]
        if f.get("action_required") is True:
            clauses.append("ai_tags->>'action_required' = 'true'")
        if f.get("topic"):
            clauses.append("ai_tags::text ILIKE :topic")
            params["topic"] = f"%{f['topic']}%"
        if f.get("search"):
            clauses.append("(title ILIKE :q OR summary ILIKE :q OR ai_tags::text ILIKE :q)")
            params["q"] = f"%{f['search']}%"
        if f.get("date_from"):
            clauses.append("date >= :date_from")
            params["date_from"] = f["date_from"]
        if f.get("date_to"):
            # Inclusive end date: < (to + 1 day) so the whole `to` day is
            # included regardless of any time component (keeps the date index
            # usable). CAST so asyncpg infers the param as a date, not interval.
            clauses.append("date < (CAST(:date_to AS date) + interval '1 day')")
            params["date_to"] = f["date_to"]
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    async def feed(
        self, filters: dict[str, Any], page: int, limit: int
    ) -> tuple[list[RegDocSummary], int]:
        where, params = self._build_filters(filters)
        total = (
            await self.s.execute(text(f"SELECT COUNT(*) FROM content{where}"), params)
        ).scalar_one()
        params = {**params, "limit": limit, "offset": (page - 1) * limit}
        rows = (
            await self.s.execute(
                text(
                    f"SELECT {_SUMMARY_COLS} FROM content{where} "
                    "ORDER BY date DESC NULLS LAST, id DESC LIMIT :limit OFFSET :offset"
                ),
                params,
            )
        ).all()
        return [_summary(r) for r in rows], int(total)

    async def get_doc(self, doc_id: int) -> RegDocDetail | None:
        row = (
            await self.s.execute(
                text(f"SELECT {_DETAIL_COLS} FROM content WHERE id = :id"),
                {"id": doc_id},
            )
        ).first()
        if row is None:
            return None
        return RegDocDetail(
            id=row.id,
            type=row.type,
            sub_type=row.sub_type,
            title=row.title,
            date=row.date,
            summary=row.summary,
            sebi_id=row.sebi_id,
            sebi_department=row.sebi_department,
            sebi_url=row.sebi_url,
            sebi_section=row.sebi_section,
            sebi_sub_section=row.sebi_sub_section,
            sebi_info_for=row.sebi_info_for,
            meeting_date=row.meeting_date,
            extracted_text=row.extracted_text,
            language=row.language,
            related_content_id=row.related_content_id,
            created_at=row.created_at,
            updated_at=row.updated_at,
            ai_tags=_parse_ai_tags(row.ai_tags),
        )

    async def by_ids(self, ids: list[int]) -> list[RegDocSummary]:
        """Resolve a list of content ids → summaries (for bookmarks)."""
        if not ids:
            return []
        rows = (
            await self.s.execute(
                text(
                    f"SELECT {_SUMMARY_COLS} FROM content WHERE id = ANY(:ids) "
                    "ORDER BY date DESC NULLS LAST, id DESC"
                ),
                {"ids": ids},
            )
        ).all()
        return [_summary(r) for r in rows]

    async def search_any(self, terms: list[str], limit: int) -> list[RegDocSummary]:
        """Recent docs matching ANY of the given terms (title/summary/tags ILIKE).
        Powers the alert feed from a user's tracked topics/entities."""
        clean = [t.strip() for t in terms if t and t.strip()][:25]
        if not clean:
            return []
        clauses: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        for i, term in enumerate(clean):
            key = f"t{i}"
            clauses.append(f"(title ILIKE :{key} OR summary ILIKE :{key} OR ai_tags::text ILIKE :{key})")
            params[key] = f"%{term}%"
        where = " OR ".join(clauses)
        rows = (
            await self.s.execute(
                text(
                    f"SELECT {_SUMMARY_COLS} FROM content WHERE {where} "
                    "ORDER BY date DESC NULLS LAST, id DESC LIMIT :limit"
                ),
                params,
            )
        ).all()
        return [_summary(r) for r in rows]

    async def recent(self, limit: int) -> list[RegDocSummary]:
        rows = (
            await self.s.execute(
                text(
                    f"SELECT {_SUMMARY_COLS} FROM content "
                    "ORDER BY date DESC NULLS LAST, id DESC LIMIT :limit"
                ),
                {"limit": limit},
            )
        ).all()
        return [_summary(r) for r in rows]

    # ── Aggregations / stats ───────────────────────────────────────────────
    async def types(self) -> list[TypeCount]:
        rows = (
            await self.s.execute(
                text("SELECT type, COUNT(*) c FROM content GROUP BY type ORDER BY c DESC")
            )
        ).all()
        return [TypeCount(type=r.type, count=int(r.c)) for r in rows]

    async def topics(self, limit: int) -> list[TopicCount]:
        topics_arr = _arr("ai_tags->'topics'")
        rows = (
            await self.s.execute(
                text(
                    f"SELECT topic, COUNT(*) c FROM content, {topics_arr} "
                    "AS topic WHERE ai_tags IS NOT NULL GROUP BY topic "
                    "ORDER BY c DESC LIMIT :limit"
                ),
                {"limit": limit},
            )
        ).all()
        return [TopicCount(topic=r.topic, count=int(r.c)) for r in rows]

    async def stats(self) -> dict[str, Any]:
        async def scalar(sql: str) -> int:
            return int((await self.s.execute(text(sql))).scalar_one())

        total = await scalar("SELECT COUNT(*) FROM content")
        this_week = await scalar(
            "SELECT COUNT(*) FROM content WHERE date >= NOW() - INTERVAL '7 days'"
        )
        today = await scalar("SELECT COUNT(*) FROM content WHERE date::date = NOW()::date")
        action_required = await scalar(
            "SELECT COUNT(*) FROM content WHERE ai_tags->>'action_required' = 'true'"
        )
        high_week = await scalar(
            "SELECT COUNT(*) FROM content WHERE date >= NOW() - INTERVAL '7 days' "
            "AND ai_tags->>'severity' = 'High'"
        )
        dl_arr = _arr("c.ai_tags->'deadlines'")
        open_deadlines = await scalar(
            "SELECT COUNT(*) FROM content c WHERE EXISTS (SELECT 1 FROM "
            f"{dl_arr} AS d(value) "
            "WHERE d.value >= to_char(NOW(), 'YYYY-MM-DD'))"
        )
        type_rows = (
            await self.s.execute(
                text("SELECT type, COUNT(*) c FROM content GROUP BY type ORDER BY c DESC")
            )
        ).all()
        sev_rows = (
            await self.s.execute(
                text(
                    "SELECT ai_tags->>'severity' s, COUNT(*) c FROM content "
                    "WHERE ai_tags->>'severity' IS NOT NULL GROUP BY s ORDER BY c DESC"
                )
            )
        ).all()
        intent_rows = (
            await self.s.execute(
                text(
                    "SELECT ai_tags->>'intent' i, COUNT(*) c FROM content "
                    "WHERE ai_tags->>'intent' IS NOT NULL GROUP BY i ORDER BY c DESC LIMIT 10"
                )
            )
        ).all()
        return {
            "total_documents": total,
            "this_week": this_week,
            "today": today,
            "action_required": action_required,
            "high_severity_week": high_week,
            "open_deadlines": open_deadlines,
            "type_counts": [TypeCount(type=r.type, count=int(r.c)) for r in type_rows],
            "severity_counts": [
                SeverityCount(severity=r.s, count=int(r.c)) for r in sev_rows
            ],
            "intent_counts": [
                IntentCount(intent=r.i, count=int(r.c)) for r in intent_rows
            ],
        }

    # ── Deadlines ──────────────────────────────────────────────────────────
    async def deadlines(self, limit: int) -> list[DeadlineItem]:
        dl_arr = _arr("c.ai_tags->'deadlines'")
        rows = (
            await self.s.execute(
                text(
                    "SELECT c.id, c.type, c.title, c.date, d.value AS deadline, "
                    "c.ai_tags->>'severity' AS severity, c.ai_tags->>'intent' AS intent "
                    f"FROM content c, {dl_arr} AS d(value) "
                    "WHERE d.value >= to_char(NOW(), 'YYYY-MM-DD') "
                    "ORDER BY d.value ASC LIMIT :limit"
                ),
                {"limit": limit},
            )
        ).all()
        return [
            DeadlineItem(
                id=r.id,
                type=r.type,
                title=r.title,
                date=r.date,
                deadline=r.deadline,
                severity=r.severity,
                intent=r.intent,
            )
            for r in rows
        ]

    # ── Calendar (month-scoped: deadlines + board meetings, past & future) ──
    async def calendar_events(self, start, end) -> list[CalendarEvent]:
        """All dated events in [start, end] (inclusive) — compliance deadlines
        from ai_tags.deadlines + board meetings from meeting_date. `start`/`end`
        are date objects."""
        start_s, end_s = start.isoformat(), end.isoformat()
        dl_arr = _arr("c.ai_tags->'deadlines'")
        dl_rows = (
            await self.s.execute(
                text(
                    "SELECT c.id, c.type, c.title, d.value AS dt, "
                    "c.ai_tags->>'severity' AS severity "
                    f"FROM content c, {dl_arr} AS d(value) "
                    "WHERE d.value >= :start AND d.value <= :end"
                ),
                {"start": start_s, "end": end_s},
            )
        ).all()
        bm_rows = (
            await self.s.execute(
                text(
                    "SELECT id, type, title, "
                    "to_char(COALESCE(meeting_date, date), 'YYYY-MM-DD') AS dt, "
                    "ai_tags->>'severity' AS severity FROM content "
                    "WHERE type = 'BOARD_MEETING' "
                    "AND COALESCE(meeting_date, date) >= :start_d "
                    "AND COALESCE(meeting_date, date) < (CAST(:end_d AS date) + interval '1 day')"
                ),
                {"start_d": start, "end_d": end},
            )
        ).all()
        events = [
            CalendarEvent(
                id=r.id, type=r.type, title=r.title, date=r.dt,
                kind="deadline", severity=r.severity,
            )
            for r in dl_rows
        ]
        events += [
            CalendarEvent(
                id=r.id, type=r.type, title=r.title, date=r.dt,
                kind="board", severity=r.severity,
            )
            for r in bm_rows
        ]
        events.sort(key=lambda e: e.date)
        return events

    # ── Weekly digest ──────────────────────────────────────────────────────
    async def weekly_summaries(self, limit: int) -> list[WeeklySummary]:
        rows = (
            await self.s.execute(
                text(
                    "SELECT id, week_start_date, week_end_date, generated_at, summary_text "
                    "FROM weekly_summaries ORDER BY week_start_date DESC LIMIT :limit"
                ),
                {"limit": limit},
            )
        ).all()
        return [
            WeeklySummary(
                id=r.id,
                week_start_date=r.week_start_date,
                week_end_date=r.week_end_date,
                generated_at=r.generated_at,
                summary_text=r.summary_text,
            )
            for r in rows
        ]
