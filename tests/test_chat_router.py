"""Tests for the ``POST /api/v1/chat/run`` endpoint shape and validation.

We don't exercise the actual LLM here — that's expensive, non-deterministic,
and requires an API key. Those concerns belong in
``test_chat_agent_integration.py`` which we skip unless ``GEMINI_API_KEY``
is set in the environment.

What this file covers:
  * Request validation (Pydantic boundary).
  * 422 for malformed bodies.
  * Auth dependency wires up correctly.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_chat_run_rejects_missing_message(client, auth_headers):
    response = await client.post(
        "/api/v1/chat/run", headers=auth_headers, json={}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_run_rejects_empty_message(client, auth_headers):
    response = await client.post(
        "/api/v1/chat/run", headers=auth_headers, json={"message": ""}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_run_rejects_oversized_message(client, auth_headers):
    huge = "x" * 10_000
    response = await client.post(
        "/api/v1/chat/run", headers=auth_headers, json={"message": huge}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_run_rejects_unknown_agent(client, auth_headers):
    response = await client.post(
        "/api/v1/chat/run",
        headers=auth_headers,
        json={"message": "hi", "agent": "totally_made_up"},
    )
    assert response.status_code == 422
