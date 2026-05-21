"""Tests for BMC endpoints that don't require an LLM (not-found + read paths).

The generation endpoint (POST /run) does 9 LLM calls, so it's exercised
manually / in the live run, not here.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_get_latest_bmc_404_when_none(client, auth_headers):
    # TCS is seeded but has no BMC in the test DB.
    resp = await client.get("/api/v1/bmc/TCS", headers=auth_headers)
    assert resp.status_code == 404
    assert "No BMC generated" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_library_empty_ok(client, auth_headers):
    resp = await client.get("/api/v1/bmc/TCS/library", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_get_specific_version_404(client, auth_headers):
    resp = await client.get("/api/v1/bmc/TCS/3", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_run_bmc_unknown_company_404(client, auth_headers):
    resp = await client.post(
        "/api/v1/bmc/NOSUCHCO/run", headers=auth_headers, json={}
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_company_bmc_tool_unknown_company():
    """The agent read-tool returns a structured note (not an exception) for
    an unknown company, so the agent degrades gracefully. Robust across DBs:
    'NOSUCHCO' resolves to None regardless of seed state."""
    from src.tools.bmc_tools import get_company_bmc

    result = await get_company_bmc("NOSUCHCO")
    assert result["found"] is False
    assert "coverage universe" in result["note"]
