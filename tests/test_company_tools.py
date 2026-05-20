"""Direct tests for the agent-callable company tools.

We test the tool *functions* against a real DB (seed data from migrations).
This is the most valuable layer to test: tool correctness is what determines
whether the agent gives the right answer. The LLM step is exercised in
``test_chat_agent_integration.py``, conditionally.
"""

from __future__ import annotations

import pytest

from src.tools.company_tools import (
    list_covered_sectors,
    lookup_company,
    search_companies,
)


@pytest.mark.asyncio
async def test_lookup_company_found():
    result = await lookup_company("TCS")
    assert result["found"] is True
    assert result["ticker"] == "TCS"
    assert result["name"] == "Tata Consultancy Services"
    assert result["sector"] == "Information Technology"
    assert result["isin"] == "INE467B01029"


@pytest.mark.asyncio
async def test_lookup_company_case_insensitive():
    result = await lookup_company("tcs")
    assert result["found"] is True
    assert result["ticker"] == "TCS"


@pytest.mark.asyncio
async def test_lookup_company_not_found():
    result = await lookup_company("NOSUCHTICKER")
    assert result["found"] is False
    assert result["ticker"] == "NOSUCHTICKER"


@pytest.mark.asyncio
async def test_search_companies_by_name():
    result = await search_companies("Reliance")
    assert result["total"] >= 1
    tickers = [item["ticker"] for item in result["items"]]
    assert "RELIANCE" in tickers


@pytest.mark.asyncio
async def test_search_companies_by_alias():
    """'Tata Consultancy' resolves to TCS via the name alias seeded in migration 0002."""
    result = await search_companies("Tata Consultancy")
    tickers = [item["ticker"] for item in result["items"]]
    assert "TCS" in tickers


@pytest.mark.asyncio
async def test_search_companies_filter_by_sector():
    result = await search_companies("", sector="Financials")
    tickers = {item["ticker"] for item in result["items"]}
    # HDFCBANK, ICICIBANK, SBIN are all in Financials in the seed.
    assert {"HDFCBANK", "ICICIBANK", "SBIN"}.issubset(tickers)
    for item in result["items"]:
        assert item["sector"] == "Financials"


@pytest.mark.asyncio
async def test_search_companies_limit_capped_at_25():
    """Even if caller asks for 999, repo caps at 25."""
    result = await search_companies("", limit=999)
    assert len(result["items"]) <= 25


@pytest.mark.asyncio
async def test_list_covered_sectors_returns_sorted_distinct():
    result = await list_covered_sectors()
    sectors = result["sectors"]
    assert sectors == sorted(sectors)
    assert len(sectors) == len(set(sectors))
    # From the seed data we expect at least these.
    assert "Information Technology" in sectors
    assert "Financials" in sectors
    assert "Energy" in sectors
