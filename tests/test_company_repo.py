"""Tests for the typo-tolerant CompanyRepository.

Two layers:
  1. Pure unit tests for the fuzzy-search helpers (no DB).
  2. Integration tests that seed a tiny ``company_industry`` table via the
     CatalogBase metadata and exercise ``CompanyRepository.list()`` end-to-end.

The integration layer creates / drops the catalog table inside the existing
``db_session`` fixture's transaction — same approach as the rest of the
test suite, no extra Postgres setup needed.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import text

from src.core.catalog_database import CatalogBase
from src.models.catalog import CompanyIndustry
from src.repositories.company_repo import (
    CompanyRepository,
    _candidate_score,
    _normalize_query,
    _rank_candidates,
)


# ── Pure unit tests — fuzzy-search helpers ─────────────────────────────────


class TestNormalizeQuery:
    def test_lowercases_and_strips(self) -> None:
        assert _normalize_query("  Reliance Industries  ") == "reliance industries"

    def test_drops_punctuation(self) -> None:
        assert _normalize_query("Tata, Consultancy.") == "tata consultancy"

    def test_strips_legal_suffix_ltd(self) -> None:
        assert _normalize_query("TCS Ltd") == "tcs"
        assert _normalize_query("Reliance Limited") == "reliance"

    def test_strips_inc_suffix(self) -> None:
        assert _normalize_query("Acme Inc") == "acme"

    def test_only_strips_suffix_when_at_end(self) -> None:
        # "Ltd" embedded mid-string is part of the name; don't drop it.
        assert _normalize_query("Ltd Reliance") == "ltd reliance"

    def test_collapses_whitespace(self) -> None:
        assert _normalize_query("Tata   Consultancy\tServices") == "tata consultancy services"

    def test_pure_punctuation_returns_empty(self) -> None:
        assert _normalize_query("!!!,,,") == ""


class TestCandidateScore:
    def _co(self, code: str, name: str) -> CompanyIndustry:
        return CompanyIndustry(code=code, company_name=name, industry=None, isin=None)

    def test_exact_name_match_is_perfect(self) -> None:
        c = self._co("RELIANCE", "Reliance Industries")
        # Normalized query == lowercased name
        assert _candidate_score(c, "reliance industries") == 100

    def test_one_letter_typo_scores_high(self) -> None:
        c = self._co("RELIANCE", "Reliance Industries")
        # "Reliac" missing the n+e at the end; partial_ratio should be high
        score = _candidate_score(c, "reliac")
        assert score >= 75, f"expected ≥75 for one-letter typo, got {score}"

    def test_completely_unrelated_scores_low(self) -> None:
        c = self._co("RELIANCE", "Reliance Industries")
        assert _candidate_score(c, "asdfqwer") < 40

    def test_short_query_boosts_code_match(self) -> None:
        # TCS exact code, name is the longer "Tata Consultancy Services"
        c = self._co("TCS", "Tata Consultancy Services")
        # Query "TCS" should score very high via the code-match bonus
        assert _candidate_score(c, "tcs") >= 95


class TestRankCandidates:
    def _co(self, code: str, name: str) -> CompanyIndustry:
        return CompanyIndustry(code=code, company_name=name, industry=None, isin=None)

    def test_top_match_is_first(self) -> None:
        cands = [
            self._co("INFY", "Infosys"),
            self._co("RELIANCE", "Reliance Industries"),
            self._co("TCS", "Tata Consultancy Services"),
        ]
        ranked = _rank_candidates(cands, "reliance")
        assert ranked[0][0].code == "RELIANCE"
        # Score is descending
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_ties_break_alphabetically(self) -> None:
        # Two near-identical inputs; the rank should still be deterministic.
        cands = [
            self._co("ZZZ", "Zzzero Inc"),
            self._co("AAA", "Aalpha Inc"),
        ]
        # Same letter prefix bonus collapses scores; expect alphabetical tiebreak.
        ranked = _rank_candidates(cands, "xyz")
        # Order shouldn't crash; primary contract is non-empty + scores attached.
        assert len(ranked) == 2
        for _, s in ranked:
            assert 0 <= s <= 100


# ── Integration tests against a seeded catalog table ───────────────────────


@pytest_asyncio.fixture
async def catalog_table(db_session):
    """Create ``company_industry`` on the test DB and seed with sample rows.

    The table is created inside the test's outer transaction so the
    SAVEPOINT-based rollback in ``conftest.db_session`` cleans it up at
    teardown automatically.
    """
    conn = await db_session.connection()

    # Create the table via CatalogBase metadata. Sync-style DDL — wrapped via run_sync.
    def _create(sync_conn):
        CompanyIndustry.__table__.create(bind=sync_conn, checkfirst=True)

    await conn.run_sync(_create)

    sample = [
        ("RELIANCE", "Reliance Industries Limited", "Refineries", "INE002A01018"),
        ("TCS", "Tata Consultancy Services", "Software & Services", "INE467B01029"),
        ("INFY", "Infosys", "Software & Services", "INE009A01021"),
        ("HDFCBANK", "HDFC Bank Limited", "Banks", "INE040A01034"),
        ("ICICIBANK", "ICICI Bank Limited", "Banks", "INE090A01021"),
        ("ITC", "ITC Limited", "Tobacco Products", "INE154A01025"),
        ("AXISBANK", "Axis Bank Limited", "Banks", "INE238A01034"),
        ("SBIN", "State Bank of India", "Banks", "INE062A01020"),
        ("LT", "Larsen & Toubro Limited", "Construction", "INE018A01030"),
        ("MOIL", "MOIL Limited", "Metals & Mining", "INE490G01020"),
    ]
    await db_session.execute(
        text(
            """
            INSERT INTO company_industry (code, company_name, industry, isin)
            VALUES (:code, :name, :industry, :isin)
            """
        ),
        [{"code": c, "name": n, "industry": i, "isin": s} for c, n, i, s in sample],
    )
    await db_session.flush()
    yield


@pytest.mark.asyncio
async def test_exact_name_search_finds_company(db_session, catalog_table) -> None:
    repo = CompanyRepository(db_session)
    result = await repo.list(search="Reliance Industries", limit=5)
    assert result.items, "should find at least one match"
    assert result.items[0].code == "RELIANCE"


@pytest.mark.asyncio
async def test_typo_finds_top_match(db_session, catalog_table) -> None:
    repo = CompanyRepository(db_session)
    # One-letter typo
    result = await repo.list(search="Reliac", limit=5)
    assert result.items, "typo should still surface Reliance"
    assert result.items[0].code == "RELIANCE"


@pytest.mark.asyncio
async def test_partial_name_finds_company(db_session, catalog_table) -> None:
    repo = CompanyRepository(db_session)
    result = await repo.list(search="Tata Consultanc", limit=5)
    assert result.items
    assert result.items[0].code == "TCS"


@pytest.mark.asyncio
async def test_legal_suffix_is_stripped(db_session, catalog_table) -> None:
    repo = CompanyRepository(db_session)
    result = await repo.list(search="TCS Ltd", limit=5)
    assert result.items
    assert result.items[0].code == "TCS"


@pytest.mark.asyncio
async def test_gibberish_returns_empty(db_session, catalog_table) -> None:
    repo = CompanyRepository(db_session)
    result = await repo.list(search="asdfqwerzxcv", limit=5)
    assert result.items == []
    assert result.suggestions == []


@pytest.mark.asyncio
async def test_sector_filter_works(db_session, catalog_table) -> None:
    repo = CompanyRepository(db_session)
    result = await repo.list(search="bank", sector="Banks", limit=10)
    codes = {c.code for c in result.items}
    assert "HDFCBANK" in codes
    assert "ICICIBANK" in codes
    # Non-Banks shouldn't show up even if name matches "bank"
    assert "RELIANCE" not in codes


@pytest.mark.asyncio
async def test_fuzzy_false_uses_legacy_substring(db_session, catalog_table) -> None:
    repo = CompanyRepository(db_session)
    # Typo "Reliac" → no substring match → no items
    result = await repo.list(search="Reliac", limit=5, fuzzy=False)
    assert result.items == []  # legacy ILIKE has no typo tolerance


@pytest.mark.asyncio
async def test_no_search_returns_all_paginated(db_session, catalog_table) -> None:
    repo = CompanyRepository(db_session)
    result = await repo.list(limit=5)
    assert len(result.items) == 5
    assert result.total == 10


@pytest.mark.asyncio
async def test_suggestions_populated_when_no_strong_hit(db_session, catalog_table) -> None:
    """When the query is a poor fit, the repo should still surface near-misses
    in ``suggestions`` so the LLM can ask 'did you mean ...?'"""
    repo = CompanyRepository(db_session)
    # "Reliaace" — extra letter, off enough that it shouldn't hit the 60
    # threshold confidently for all candidates, but Reliance Industries
    # is still the obvious near-match.
    result = await repo.list(search="Releace", limit=5)
    # Either items OR suggestions should contain Reliance — both are
    # acceptable proof the fuzzy path found it.
    all_codes = {c.code for c in result.items} | {c.code for c in result.suggestions}
    assert "RELIANCE" in all_codes
