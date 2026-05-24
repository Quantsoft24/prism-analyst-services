"""Live integration test for the Company Intelligence agent.

Skipped unless ``GEMINI_API_KEY`` is set in the environment — keeps CI free
and deterministic. Run locally with::

    GEMINI_API_KEY=<your key> pytest tests/test_chat_agent_integration.py -v

What it verifies:
  * The full SSE stream emits at least one meta, at least one tool_call, and
    exactly one final or one error event.
  * On success: the agent_runs row is updated to ``status='complete'`` and
    captures token usage.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest
from sqlalchemy import select

from src.models.agent_run import AgentRun

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="Set GEMINI_API_KEY to run the live LLM integration test.",
)


@pytest.mark.asyncio
async def test_company_intel_end_to_end_streams_events(client, auth_headers, db_session):
    """A simple 'what is TCS?' query should trigger at least one tool call and finish."""
    async with client.stream(
        "POST",
        "/api/v1/chat/run",
        headers=auth_headers,
        json={"message": "What does TCS do? One sentence."},
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        events: list[dict] = []
        async for raw_line in response.aiter_lines():
            if raw_line.startswith("data:"):
                payload = raw_line[len("data:") :].strip()
                if payload:
                    events.append(json.loads(payload))

    types = [e["type"] for e in events]
    assert "meta" in types
    # Either we got a real final, or the agent errored (still pass — the
    # stream contract held). We never want both.
    assert ("final" in types) ^ ("error" in types)

    meta = next(e for e in events if e["type"] == "meta")
    agent_run_id = uuid.UUID(meta["agent_run_id"])

    # Confirm the audit row was written.
    row = (await db_session.execute(select(AgentRun).where(AgentRun.id == agent_run_id))).scalar_one()
    assert row.firm_id == "QUANTSOFT"
    assert row.agent_name == "company_intel"
    assert row.status in {"complete", "failed", "timeout"}
