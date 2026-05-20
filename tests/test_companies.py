"""End-to-end tests for ``/api/v1/companies``.

These hit a real Postgres (seed migration provides the data). No mocks.
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_list_companies_returns_seeded_rows(client, auth_headers):
    """The seed migration inserts 10 NSE companies — verify they list."""
    response = await client.get("/api/v1/companies", headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert "items" in body
    assert "page" in body
    assert body["page"]["total"] >= 10
    tickers = [item["ticker"] for item in body["items"]]
    assert "RELIANCE" in tickers
    assert "TCS" in tickers


@pytest.mark.asyncio
async def test_list_companies_pagination(client, auth_headers):
    """Limit + offset work as documented."""
    page1 = await client.get(
        "/api/v1/companies?limit=3&offset=0", headers=auth_headers
    )
    page2 = await client.get(
        "/api/v1/companies?limit=3&offset=3", headers=auth_headers
    )
    assert page1.status_code == page2.status_code == 200
    p1 = page1.json()
    p2 = page2.json()
    assert len(p1["items"]) == 3
    assert len(p2["items"]) == 3
    p1_tickers = {i["ticker"] for i in p1["items"]}
    p2_tickers = {i["ticker"] for i in p2["items"]}
    assert p1_tickers.isdisjoint(p2_tickers), "pages must not overlap"


@pytest.mark.asyncio
async def test_list_companies_search_by_ticker(client, auth_headers):
    response = await client.get(
        "/api/v1/companies?search=TCS", headers=auth_headers
    )
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) >= 1
    assert any(i["ticker"] == "TCS" for i in items)


@pytest.mark.asyncio
async def test_list_companies_search_by_name(client, auth_headers):
    """Searching by full name should find the company via alias OR name match."""
    response = await client.get(
        "/api/v1/companies?search=Reliance", headers=auth_headers
    )
    assert response.status_code == 200
    items = response.json()["items"]
    assert any(i["ticker"] == "RELIANCE" for i in items)


@pytest.mark.asyncio
async def test_list_companies_filter_by_sector(client, auth_headers):
    response = await client.get(
        "/api/v1/companies?sector=Financials", headers=auth_headers
    )
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) >= 2  # HDFCBANK, ICICIBANK, SBIN
    for item in items:
        assert item["sector"] == "Financials"


@pytest.mark.asyncio
async def test_get_company_by_ticker(client, auth_headers):
    response = await client.get(
        "/api/v1/companies/TCS", headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "TCS"
    assert data["name"] == "Tata Consultancy Services"
    assert data["isin"] == "INE467B01029"
    assert isinstance(data["aliases"], list)


@pytest.mark.asyncio
async def test_get_company_by_uuid(client, auth_headers):
    """First list to get a real UUID, then fetch by UUID."""
    listing = await client.get(
        "/api/v1/companies?search=RELIANCE", headers=auth_headers
    )
    company_id = listing.json()["items"][0]["id"]
    response = await client.get(
        f"/api/v1/companies/{company_id}", headers=auth_headers
    )
    assert response.status_code == 200
    assert response.json()["ticker"] == "RELIANCE"


@pytest.mark.asyncio
async def test_get_company_404(client, auth_headers):
    response = await client.get(
        "/api/v1/companies/NOSUCHTICKER", headers=auth_headers
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_invalid_limit_rejected(client, auth_headers):
    """FastAPI must reject limit > 200 with 422."""
    response = await client.get(
        "/api/v1/companies?limit=9999", headers=auth_headers
    )
    assert response.status_code == 422
