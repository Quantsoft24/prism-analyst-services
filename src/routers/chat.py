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

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from src.agents import build_company_intel_agent
from src.core.auth import get_current_firm_id
from src.schemas.chat import ChatRunRequest
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
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> EventSourceResponse:
    # Resolve the agent. Only one is registered in Slice 3 — but the
    # Literal on ``body.agent`` constrains valid values at the Pydantic layer.
    if body.agent == "company_intel":
        agent = build_company_intel_agent()
    else:  # pragma: no cover — Pydantic Literal blocks this path
        raise ValueError(f"Unknown agent: {body.agent}")

    runner = AgentRunner(
        agent=agent,
        firm_id=firm_id,
        session_id=body.session_id,
    )

    async def event_stream() -> AsyncIterator[dict]:
        async for event in runner.run(body.message):
            yield _serialize_event(event)

    # ``EventSourceResponse`` handles the ``Content-Type``, keep-alive pings,
    # and connection lifecycle. We hand it an async generator of dicts where
    # each dict has the keys ``event`` (SSE event name) and ``data`` (string).
    return EventSourceResponse(event_stream())


def _serialize_event(event: ChatEvent) -> dict[str, str]:
    """Convert a typed event into the dict ``sse-starlette`` expects.

    We set both the SSE ``event:`` field AND duplicate ``type`` in the JSON
    payload so consumers using either ``EventSource.addEventListener('type', ...)``
    or a plain reader that switches on ``parsed.type`` both work.
    """
    payload = event.model_dump(mode="json")
    return {"event": event.type, "data": json.dumps(payload, separators=(",", ":"))}
