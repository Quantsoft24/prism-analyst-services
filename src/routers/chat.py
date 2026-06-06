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
    ConversationTitleUpdate,
    ConversationTurn,
    QuotaRead,
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
) -> list[ConversationSummary]:
    rows = await ConversationRepository(session).list_conversations(
        firm_id=principal.firm_id,
        user_id=principal.user_id,
        client_key=_guest_key(principal, request),
        limit=limit,
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
    runs = await ConversationRepository(session).get_conversation(
        session_id=session_id,
        firm_id=principal.firm_id,
        user_id=principal.user_id,
        client_key=_guest_key(principal, request),
    )
    if not runs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    turns = [
        ConversationTurn(
            agent_run_id=r.id,
            user_input=r.user_input,
            final_answer=r.final_answer,
            status=r.status,
            created_at=r.created_at,
            tool_trace=r.tool_trace,
        )
        for r in runs
    ]
    return ConversationDetail(session_id=session_id, turns=turns)


@router.patch(
    "/conversations/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Rename a conversation",
)
async def rename_conversation(
    session_id: str,
    body: ConversationTitleUpdate,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> None:
    ok = await ConversationRepository(session).set_title(
        session_id=session_id,
        firm_id=principal.firm_id,
        user_id=principal.user_id,
        client_key=_guest_key(principal, request),
        title=body.title.strip(),
    )
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
