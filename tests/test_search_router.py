"""Tests for the /api/v1/search + /api/v1/filings endpoints.

Validation + unknown-entity paths (no LLM needed). The unknown-ticker search
returns early with a ``note`` before any query embedding, so it's testable
without the embedding provider.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_search_rejects_empty_query(client, auth_headers):
    resp = await client.post("/api/v1/search", headers=auth_headers, json={"query": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_search_rejects_bad_limit(client, auth_headers):
    resp = await client.post(
        "/api/v1/search", headers=auth_headers, json={"query": "revenue", "limit": 999}
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_search_unknown_ticker_returns_note(client, auth_headers):
    resp = await client.post(
        "/api/v1/search",
        headers=auth_headers,
        json={"query": "revenue", "ticker": "NOSUCHCO"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["hits"] == []
    assert "coverage universe" in body["note"]


@pytest.mark.asyncio
async def test_list_filings_unknown_company_404(client, auth_headers):
    resp = await client.get("/api/v1/filings/NOSUCHCO", headers=auth_headers)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_filings_known_company_empty_ok(client, auth_headers):
    """TCS is seeded but has no filings ingested in the test DB → empty list, 200."""
    resp = await client.get("/api/v1/filings/TCS", headers=auth_headers)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
