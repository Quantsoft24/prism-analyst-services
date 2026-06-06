"""sebi_regulatory — agent tools over the SEBI regulatory corpus (Regulatory Lens).

In-process read-only access to the SEBI Postgres via ``SebiRepository`` (same
data that backs ``/api/v1/regulatory/*``). Lets the research agent search SEBI
circulars/regulations/orders, surface upcoming compliance deadlines, and pull a
single document's summary + AI impact tags.

Every function returns a plain dict on success or a structured ``make_error``
dict on failure (so the agent can recover gracefully). If the SEBI DB isn't
configured, the tools report that instead of raising.
"""

from __future__ import annotations

from typing import Any

from src.core.sebi_database import is_sebi_configured, sebi_session_scope
from src.integrations.tools._errors import make_error
from src.repositories.sebi_repo import SebiRepository

_MAX_TEXT = 4000  # cap a document's body so tool output stays token-bounded


def _unconfigured() -> dict:
    return make_error(
        message="The SEBI regulatory database isn't configured.",
        code="sebi_unconfigured",
        next_action="give_up_gracefully",
        retriable=False,
    )


def _failed(exc: Exception, code: str) -> dict:
    return make_error(
        message="Couldn't reach the SEBI regulatory database.",
        code=code,
        next_action="ask_user_to_retry_later",
        retriable=True,
        detail=str(exc)[:400],
    )


async def sebi_search(
    query: str,
    type: str | None = None,
    severity: str | None = None,
    limit: int = 10,
) -> dict:
    """Search SEBI regulatory documents (circulars, regulations, orders, etc.).

    Args:
        query: free-text search over title, summary and AI tags (e.g. "insider
            trading", "FPI position limits", "mutual fund nomination").
        type: optional content type filter — one of CIRCULAR, REGULATION, ORDER,
            MASTER_CIRCULAR, PRESS_RELEASE, BOARD_MEETING, GUIDELINE, ACT, RULES,
            ADVISORY, GAZETTE_NOTIFICATION, GENERAL_ORDER, MUTUAL_FUND.
        severity: optional impact filter — High, Medium or Low.
        limit: max results (1-25).
    """
    if not query or not query.strip():
        return make_error(
            message="Provide a search query.",
            code="sebi_missing_query",
            next_action="ask_user_to_clarify",
        )
    if not is_sebi_configured():
        return _unconfigured()
    limit = max(1, min(limit, 25))
    try:
        async with sebi_session_scope() as session:
            repo = SebiRepository(session)
            items, total = await repo.feed(
                {"search": query.strip(), "type": type, "severity": severity}, 1, limit
            )
            return {
                "query": query,
                "total": total,
                "returned": len(items),
                "items": [_doc_brief(i.model_dump()) for i in items],
            }
    except Exception as exc:  # noqa: BLE001
        return _failed(exc, "sebi_search_failed")


async def sebi_recent(limit: int = 10) -> dict:
    """Most recent SEBI documents across all types (newest first)."""
    if not is_sebi_configured():
        return _unconfigured()
    limit = max(1, min(limit, 25))
    try:
        async with sebi_session_scope() as session:
            items = await SebiRepository(session).recent(limit)
            return {"returned": len(items), "items": [_doc_brief(i.model_dump()) for i in items]}
    except Exception as exc:  # noqa: BLE001
        return _failed(exc, "sebi_recent_failed")


async def sebi_deadlines(limit: int = 10) -> dict:
    """Upcoming SEBI compliance deadlines (AI-extracted), soonest first."""
    if not is_sebi_configured():
        return _unconfigured()
    limit = max(1, min(limit, 30))
    try:
        async with sebi_session_scope() as session:
            items = await SebiRepository(session).deadlines(limit)
            return {
                "returned": len(items),
                "items": [
                    {
                        "id": d.id,
                        "type": d.type,
                        "title": d.title,
                        "deadline": d.deadline,
                        "severity": d.severity,
                    }
                    for d in items
                ],
            }
    except Exception as exc:  # noqa: BLE001
        return _failed(exc, "sebi_deadlines_failed")


async def sebi_document(doc_id: int) -> dict:
    """Full detail for one SEBI document by id — summary, AI impact tags
    (topics, stakeholders, severity, deadlines) and the (truncated) body text."""
    if not is_sebi_configured():
        return _unconfigured()
    try:
        async with sebi_session_scope() as session:
            doc = await SebiRepository(session).get_doc(doc_id)
            if doc is None:
                return make_error(
                    message=f"No SEBI document with id {doc_id}.",
                    code="sebi_doc_not_found",
                    next_action="ask_user_to_clarify",
                )
            data = doc.model_dump()
            text = data.get("extracted_text") or ""
            data["extracted_text"] = text[:_MAX_TEXT]
            data["extracted_text_truncated"] = len(text) > _MAX_TEXT
            data.pop("ai_tags", None)
            data["impact"] = {
                "intent": doc.ai_tags.intent,
                "severity": doc.ai_tags.severity,
                "action_required": doc.ai_tags.action_required,
                "topics": doc.ai_tags.topics,
                "stakeholders": doc.ai_tags.stakeholders,
                "deadlines": doc.ai_tags.deadlines,
            }
            return data
    except Exception as exc:  # noqa: BLE001
        return _failed(exc, "sebi_document_failed")


def _doc_brief(d: dict[str, Any]) -> dict[str, Any]:
    """Compact a summary doc for tool output (drop heavy nested tags)."""
    tags = d.get("ai_tags") or {}
    return {
        "id": d.get("id"),
        "type": d.get("type"),
        "title": d.get("title"),
        "date": str(d.get("date")) if d.get("date") else None,
        "summary": d.get("summary"),
        "severity": tags.get("severity"),
        "intent": tags.get("intent"),
        "topics": tags.get("topics", []),
        "department": d.get("sebi_department"),
    }


SEBI_REGULATORY_TOOLS = [sebi_search, sebi_recent, sebi_deadlines, sebi_document]
