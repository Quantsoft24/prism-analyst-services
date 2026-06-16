"""Chat / agent invocation endpoints.

``POST /api/v1/chat/run`` runs an agent and streams typed events back over
Server-Sent Events. The wire format (chosen for compatibility with
``EventSource`` and `fetch`-with-ReadableStream both):

  event: meta
  data: {"type":"meta","agent_run_id":"...","session_id":"...","agent_name":"company_intel"}

  event: tool_call
  data: {"type":"tool_call","tool":"lookup_company","args":{"ticker":"TCS"},"call_id":"..."}

  ... and so on ...

  event: final
  data: {"type":"final","answer":"...","cost_usd":0.0008,"input_tokens":1234,"output_tokens":210,"latency_ms":3200}

Frontend reads ``event.type`` from the JSON to dispatch into UI state. The
SSE ``event:`` field is also set so plain ``EventSource`` listeners can
filter without parsing JSON.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from src.agents import build_company_intel_agent
from src.auth.principal import Principal, get_current_principal
from src.core.database import get_session
from src.integrations.firm_state import enabled_integration_names
from src.models.firm import Firm
from src.repositories.conversation_repo import ConversationRepository
from src.schemas.chat import (
    ChatRunRequest,
    ConversationDetail,
    ConversationSummary,
    ConversationTurn,
    ConversationUpdate,
    FeedbackCreate,
    FeedbackRead,
    QuotaRead,
    SharedConversationRead,
    ShareRead,
)
from src.services import rate_limit
from src.services.agent_runner import AgentRunner, ChatEvent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


def _guest_key(principal: Principal, request: Request) -> str | None:
    """Per-browser id for an anonymous caller — the ``X-Guest-Id`` header (set by
    the client) with the client IP as a fallback. ``None`` for signed-in users.

    Critical for isolation: all guests share the ``__anonymous__`` firm, so this
    is the ONLY thing that separates one guest's conversations from another's.
    """
    if not principal.is_anonymous:
        return None
    return request.headers.get("X-Guest-Id") or (
        request.client.host if request.client else None
    )


async def _quota_state(
    session: AsyncSession, principal: Principal, request: Request
) -> tuple[int, int, str | None]:
    """Resolve (messages used today, daily limit, guest_key) for the caller.

    Anonymous callers are identified by the ``X-Guest-Id`` header (a per-browser
    id) falling back to the client IP; signed-in callers by their tier.
    """
    guest_key = _guest_key(principal, request)
    tier: str | None = None
    if not principal.is_anonymous:
        tier = await session.scalar(
            select(Firm.subscription_tier).where(Firm.slug == principal.firm_id)
        )
    limit = rate_limit.cap_for(is_anonymous=principal.is_anonymous, tier=tier)
    used = await rate_limit.used_today(
        session, user_id=principal.user_id, client_key=guest_key
    )
    return used, limit, guest_key


@router.post(
    "/run",
    summary="Run an agent and stream events",
    description=(
        "Invokes the named agent with the provided message. Returns a "
        "Server-Sent Events stream of typed events. The stream always ends "
        "with exactly one ``final`` OR ``error`` event."
    ),
    response_class=EventSourceResponse,
    responses={
        200: {
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "description": (
                            "Stream of ``event: <type>\\ndata: <json>\\n\\n`` blocks."
                        ),
                    }
                }
            }
        }
    },
)
async def run_agent(
    body: ChatRunRequest,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> EventSourceResponse:
    firm_id = principal.firm_id

    # Daily message cap (configurable per tier — config/rate_limits.yml).
    used, limit, guest_key = await _quota_state(session, principal, request)
    if rate_limit.is_enabled() and used >= limit:
        detail = (
            f"You've reached the guest limit of {limit} messages for today. "
            "Sign in to keep going."
            if principal.is_anonymous
            else f"You've reached your daily limit of {limit} messages. It resets tomorrow."
        )
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=detail)

    # Resolve the agent. Only one is registered in Slice 3 — but the
    # Literal on ``body.agent`` constrains valid values at the Pydantic layer.
    if body.agent == "company_intel":
        # Attach only the integrations this firm has enabled (default ON).
        enabled = await enabled_integration_names(firm_id, session)
        agent = build_company_intel_agent(integrations=enabled)
    else:  # pragma: no cover — Pydantic Literal blocks this path
        raise ValueError(f"Unknown agent: {body.agent}")

    runner = AgentRunner(
        agent=agent,
        firm_id=firm_id,
        user_id=principal.user_id,  # attributes the agent_runs row to the user
        client_key=guest_key,  # identifies anonymous callers for the daily cap
        session_id=body.session_id,
    )

    async def event_stream() -> AsyncIterator[dict]:
        async for event in runner.run(body.message):
            yield _serialize_event(event)

    # ``EventSourceResponse`` handles the ``Content-Type``, keep-alive pings,
    # and connection lifecycle. We hand it an async generator of dicts where
    # each dict has the keys ``event`` (SSE event name) and ``data`` (string).
    return EventSourceResponse(event_stream())


@router.get(
    "/conversations",
    response_model=list[ConversationSummary],
    summary="List the current user's recent conversations",
)
async def list_conversations(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    q: Annotated[str | None, Query(max_length=200, description="Search question/answer/title.")] = None,
    archived: bool = Query(False, description="Show only archived conversations."),
) -> list[ConversationSummary]:
    rows = await ConversationRepository(session).list_conversations(
        firm_id=principal.firm_id,
        user_id=principal.user_id,
        client_key=_guest_key(principal, request),
        limit=limit,
        offset=offset,
        q=q,
        archived=archived,
    )
    return [ConversationSummary(**r) for r in rows]


@router.get(
    "/conversations/{session_id}",
    response_model=ConversationDetail,
    summary="Replay one conversation (its ordered turns)",
)
async def get_conversation(
    session_id: str,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> ConversationDetail:
    repo = ConversationRepository(session)
    runs = await repo.get_conversation(
        session_id=session_id,
        firm_id=principal.firm_id,
        user_id=principal.user_id,
        client_key=_guest_key(principal, request),
    )
    if not runs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    # The caller's 👍/👎 per turn, so replay can show the saved rating.
    feedback = await repo.get_feedback_for_runs([r.id for r in runs])
    turns = []
    for r in runs:
        payload = r.result_payload or {}
        fb = feedback.get(r.id)
        turns.append(
            ConversationTurn(
                agent_run_id=r.id,
                user_input=r.user_input,
                final_answer=r.final_answer,
                status=r.status,
                created_at=r.created_at,
                tool_trace=r.tool_trace,
                # Restore the rich view from result_payload (null on legacy rows
                # → Pydantic coerces the stored dicts back into the typed models).
                structured=payload.get("structured"),
                plan=payload.get("plan") or [],
                clarification=payload.get("clarification"),
                feedback=FeedbackRead(**fb) if fb else None,
            )
        )
    return ConversationDetail(session_id=session_id, turns=turns)


@router.patch(
    "/conversations/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Update a conversation (rename / pin / archive)",
)
async def update_conversation(
    session_id: str,
    body: ConversationUpdate,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> None:
    repo = ConversationRepository(session)
    scope = {
        "session_id": session_id,
        "firm_id": principal.firm_id,
        "user_id": principal.user_id,
        "client_key": _guest_key(principal, request),
    }
    ok = True
    if body.title is not None:
        ok = await repo.set_title(**scope, title=body.title.strip()) and ok
    if body.pinned is not None:
        ok = await repo.set_pinned(**scope, pinned=body.pinned) and ok
    if body.archived is not None:
        ok = await repo.set_archived(**scope, archived=body.archived) and ok
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")


@router.delete(
    "/conversations/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Hide a conversation from the user's history (soft delete)",
)
async def delete_conversation(
    session_id: str,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> None:
    hidden = await ConversationRepository(session).hide_conversation(
        session_id=session_id,
        firm_id=principal.firm_id,
        user_id=principal.user_id,
        client_key=_guest_key(principal, request),
    )
    if hidden == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")


@router.post(
    "/conversations/{session_id}/share",
    response_model=ShareRead,
    summary="Create (or get) a read-only public share link for a conversation",
)
async def create_share(
    session_id: str,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> ShareRead:
    share = await ConversationRepository(session).create_or_get_share(
        session_id=session_id,
        firm_id=principal.firm_id,
        user_id=principal.user_id,
        client_key=_guest_key(principal, request),
    )
    if share is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return ShareRead(**share)


@router.delete(
    "/conversations/{session_id}/share",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke a conversation's public share link",
)
async def revoke_share(
    session_id: str,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> None:
    ok = await ConversationRepository(session).revoke_share(
        session_id=session_id,
        firm_id=principal.firm_id,
        user_id=principal.user_id,
        client_key=_guest_key(principal, request),
    )
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")


@router.get(
    "/shared/{token}",
    response_model=SharedConversationRead,
    summary="Fetch a frozen, read-only shared conversation (public — no auth)",
)
async def get_shared(
    token: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> SharedConversationRead:
    snapshot = await ConversationRepository(session).get_shared_snapshot(token)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This shared link is no longer available.",
        )
    turns = []
    for r in snapshot["runs"]:
        payload = r.result_payload or {}
        turns.append(
            ConversationTurn(
                agent_run_id=r.id,
                user_input=r.user_input,
                final_answer=r.final_answer,
                status=r.status,
                created_at=r.created_at,
                tool_trace=r.tool_trace,
                structured=payload.get("structured"),
                plan=payload.get("plan") or [],
                clarification=payload.get("clarification"),
                # feedback intentionally omitted — a public snapshot carries no
                # per-user ratings.
            )
        )
    return SharedConversationRead(
        title=snapshot["title"], shared_at=snapshot["shared_at"], turns=turns
    )


@router.post(
    "/runs/{agent_run_id}/feedback",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Rate one answer (👍/👎 with optional reasons + comment)",
)
async def submit_feedback(
    agent_run_id: uuid.UUID,
    body: FeedbackCreate,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> None:
    ok = await ConversationRepository(session).upsert_feedback(
        agent_run_id=agent_run_id,
        firm_id=principal.firm_id,
        user_id=principal.user_id,
        client_key=_guest_key(principal, request),
        rating=body.rating,
        reasons=body.reasons,
        comment=body.comment,
    )
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Answer not found.")


@router.delete(
    "/runs/{agent_run_id}/feedback",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a rating (toggle 👍/👎 back to neutral)",
)
async def clear_feedback(
    agent_run_id: uuid.UUID,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> None:
    ok = await ConversationRepository(session).clear_feedback(
        agent_run_id=agent_run_id,
        firm_id=principal.firm_id,
        user_id=principal.user_id,
        client_key=_guest_key(principal, request),
    )
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Answer not found.")


@router.get("/quota", response_model=QuotaRead, summary="Today's message quota for the caller")
async def read_quota(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> QuotaRead:
    used, limit, _ = await _quota_state(session, principal, request)
    return QuotaRead(
        limit=limit,
        used=used,
        remaining=max(0, limit - used),
        is_anonymous=principal.is_anonymous,
        enabled=rate_limit.is_enabled(),
    )


def _serialize_event(event: ChatEvent) -> dict[str, str]:
    """Convert a typed event into the dict ``sse-starlette`` expects.

    We set both the SSE ``event:`` field AND duplicate ``type`` in the JSON
    payload so consumers using either ``EventSource.addEventListener('type', ...)``
    or a plain reader that switches on ``parsed.type`` both work.
    """
    payload = event.model_dump(mode="json")
    return {"event": event.type, "data": json.dumps(payload, separators=(",", ":"))}
