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

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from src.agents import build_company_intel_agent
from src.auth.principal import Principal, get_current_principal
from src.core.database import get_session
from src.integrations.firm_state import enabled_integration_names
from src.repositories.conversation_repo import ConversationRepository
from src.schemas.chat import (
    ChatRunRequest,
    ConversationDetail,
    ConversationSummary,
    ConversationTitleUpdate,
    ConversationTurn,
)
from src.services.agent_runner import AgentRunner, ChatEvent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


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
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> EventSourceResponse:
    firm_id = principal.firm_id
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
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
    limit: int = Query(30, ge=1, le=100),
) -> list[ConversationSummary]:
    rows = await ConversationRepository(session).list_conversations(
        firm_id=principal.firm_id, user_id=principal.user_id, limit=limit
    )
    return [ConversationSummary(**r) for r in rows]


@router.get(
    "/conversations/{session_id}",
    response_model=ConversationDetail,
    summary="Replay one conversation (its ordered turns)",
)
async def get_conversation(
    session_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> ConversationDetail:
    runs = await ConversationRepository(session).get_conversation(
        session_id=session_id, firm_id=principal.firm_id, user_id=principal.user_id
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
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> None:
    ok = await ConversationRepository(session).set_title(
        session_id=session_id,
        firm_id=principal.firm_id,
        user_id=principal.user_id,
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
    session: Annotated[AsyncSession, Depends(get_session)],
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> None:
    hidden = await ConversationRepository(session).hide_conversation(
        session_id=session_id, firm_id=principal.firm_id, user_id=principal.user_id
    )
    if hidden == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")


def _serialize_event(event: ChatEvent) -> dict[str, str]:
    """Convert a typed event into the dict ``sse-starlette`` expects.

    We set both the SSE ``event:`` field AND duplicate ``type`` in the JSON
    payload so consumers using either ``EventSource.addEventListener('type', ...)``
    or a plain reader that switches on ``parsed.type`` both work.
    """
    payload = event.model_dump(mode="json")
    return {"event": event.type, "data": json.dumps(payload, separators=(",", ":"))}
