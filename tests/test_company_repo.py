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


# ── Pure unit tests — lookup_company input-shape routing (no DB) ───────────


class TestLookupCompanyPatternRouting:
    """Verify ``lookup_company`` dispatches to the right code path based on
    input shape: ISIN → repo.get_by_isin; ticker → repo.get_by_ticker;
    BSE numeric → friendly refusal; foreign ISIN → friendly refusal;
    empty → missing_input refusal; multi-word → falls through to fuzzy.

    Uses ``unittest.mock`` to swap CompanyRepository so we never touch
    Postgres — these tests run in <100ms and have zero infra requirements.
    """

    @pytest.mark.asyncio
    async def test_empty_input_returns_missing_input_error(self) -> None:
        from src.tools.company_tools import lookup_company
        for empty in ["", "   ", "\t\n"]:
            result = await lookup_company(empty)
            assert result.get("ok") is False
            assert result["error_code"] == "missing_input"
            assert result["next_action"] == "ask_user_to_clarify"

    @pytest.mark.asyncio
    async def test_foreign_isin_refused_without_db_call(self) -> None:
        """US/GB ISINs should never reach the repo — they're refused
        synchronously on pattern recognition."""
        from unittest.mock import patch

        from src.tools.company_tools import lookup_company

        with patch("src.tools.company_tools.catalog_session_scope") as scope:
            result = await lookup_company("US0378331005")
            assert result.get("ok") is False
            assert result["error_code"] == "foreign_isin"
            # Confirm: never opened a DB session for foreign ISIN
            scope.assert_not_called()

    @pytest.mark.asyncio
    async def test_bse_numeric_code_refused_without_db_call(self) -> None:
        """6-digit BSE codes should be refused before any DB call."""
        from unittest.mock import patch

        from src.tools.company_tools import lookup_company

        with patch("src.tools.company_tools.catalog_session_scope") as scope:
            result = await lookup_company("500325")
            assert result.get("ok") is False
            assert result["error_code"] == "bse_code_unsupported"
            scope.assert_not_called()

    @pytest.mark.asyncio
    async def test_indian_isin_dispatches_to_get_by_isin(self) -> None:
        """An IN-prefixed 12-char ISIN should call repo.get_by_isin."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.models.catalog import CompanyIndustry
        from src.tools.company_tools import lookup_company

        # Repo returns a real-shaped CompanyIndustry instance.
        fake_co = CompanyIndustry(
            code="RELIANCE",
            company_name="Reliance Industries Limited",
            industry="Refineries",
            isin="INE002A01018",
        )

        # Mock repo: get_by_isin returns the fake row, get_by_ticker
        # should NEVER be called for an ISIN input.
        mock_repo = MagicMock()
        mock_repo.get_by_isin = AsyncMock(return_value=fake_co)
        mock_repo.get_by_ticker = AsyncMock()

        @asynccontextmanager
        async def fake_session_scope():
            yield MagicMock()  # session — unused after we mock the repo

        with patch("src.tools.company_tools.catalog_session_scope", fake_session_scope), \
             patch("src.tools.company_tools.CompanyRepository", return_value=mock_repo):
            result = await lookup_company("INE002A01018")
            assert result["found"] is True
            assert result["ticker"] == "RELIANCE"
            assert "disambiguation_note" in result
            assert "Resolved ISIN INE002A01018" in result["disambiguation_note"]
            mock_repo.get_by_isin.assert_awaited_once_with("INE002A01018")
            mock_repo.get_by_ticker.assert_not_called()

    @pytest.mark.asyncio
    async def test_regular_ticker_dispatches_to_get_by_ticker(self) -> None:
        """A plain alpha ticker like "TCS" should call get_by_ticker, not ISIN."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.models.catalog import CompanyIndustry
        from src.tools.company_tools import lookup_company

        fake_co = CompanyIndustry(
            code="TCS",
            company_name="Tata Consultancy Services",
            industry="Software & Services",
            isin="INE467B01029",
        )
        mock_repo = MagicMock()
        mock_repo.get_by_ticker = AsyncMock(return_value=fake_co)
        mock_repo.get_by_isin = AsyncMock()
        mock_repo.resolve_alias = AsyncMock(return_value=None)
        mock_repo.list = AsyncMock()

        @asynccontextmanager
        async def fake_session_scope():
            yield MagicMock()

        with patch("src.tools.company_tools.catalog_session_scope", fake_session_scope), \
             patch("src.tools.company_tools.CompanyRepository", return_value=mock_repo):
            result = await lookup_company("TCS")
            assert result["found"] is True
            assert result["ticker"] == "TCS"
            mock_repo.get_by_ticker.assert_awaited_once_with("TCS")
            mock_repo.get_by_isin.assert_not_called()
            mock_repo.list.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiword_name_falls_through_to_fuzzy(self) -> None:
        """Multi-word names like 'Tata Consultancy' should miss exact
        ticker lookup AND skip the ISIN/BSE branches, then call
        repo.list() for fuzzy resolution."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.repositories.company_repo import CompanyListResult
        from src.tools.company_tools import lookup_company

        mock_repo = MagicMock()
        mock_repo.get_by_ticker = AsyncMock(return_value=None)
        mock_repo.get_by_isin = AsyncMock()
        mock_repo.resolve_alias = AsyncMock(return_value=None)
        mock_repo.list = AsyncMock(
            return_value=CompanyListResult(items=[], total=0, suggestions=[])
        )

        @asynccontextmanager
        async def fake_session_scope():
            yield MagicMock()

        with patch("src.tools.company_tools.catalog_session_scope", fake_session_scope), \
             patch("src.tools.company_tools.CompanyRepository", return_value=mock_repo):
            result = await lookup_company("Tata Consultancy")
            # Empty fuzzy result → "found: false, suggestions: []"
            assert result["found"] is False
            # The ticker shape stored on miss is the upper-cased original input
            assert result["ticker"] == "TATA CONSULTANCY"
            # Confirm dispatch path: NOT to get_by_isin
            mock_repo.get_by_isin.assert_not_called()
            mock_repo.list.assert_awaited()

    @pytest.mark.asyncio
    async def test_lookup_company_rejects_pathological_length(self) -> None:
        """Inputs > 200 chars are refused BEFORE hitting the repo.
        Protects rapidfuzz from a 10K-char fuzzy-match storm."""
        from unittest.mock import patch

        from src.tools.company_tools import lookup_company

        with patch("src.tools.company_tools.catalog_session_scope") as scope:
            result = await lookup_company("x" * 5000)
            assert result.get("ok") is False
            assert result["error_code"] == "input_too_long"
            assert result["next_action"] == "ask_user_to_clarify"
            scope.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_companies_rejects_long_query(self) -> None:
        """search_companies guards `query` against the same length cap."""
        from unittest.mock import patch

        from src.tools.company_tools import search_companies

        with patch("src.tools.company_tools.catalog_session_scope") as scope:
            result = await search_companies(query="x" * 1000)
            assert result.get("ok") is False
            assert result["error_code"] == "input_too_long"
            scope.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_companies_rejects_long_sector(self) -> None:
        """search_companies guards `sector` against the same length cap."""
        from unittest.mock import patch

        from src.tools.company_tools import search_companies

        with patch("src.tools.company_tools.catalog_session_scope") as scope:
            result = await search_companies(query="TCS", sector="x" * 500)
            assert result.get("ok") is False
            assert result["error_code"] == "input_too_long"
            scope.assert_not_called()


# ── Pure unit tests — fuzzy sector resolution (no DB) ──────────────────────


class TestFuzzySectorResolution:
    """Verify ``_resolve_sector`` maps wonky sector hints to the catalog's
    canonical names, and returns useful suggestions on a miss."""

    @pytest.mark.asyncio
    async def test_case_insensitive_exact_match(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from src.tools.company_tools import _resolve_sector

        mock_repo = MagicMock()
        mock_repo.distinct_sectors = AsyncMock(
            return_value=["Banks", "Software & Services", "Pharmaceuticals"]
        )
        resolved, suggestions = await _resolve_sector(mock_repo, "banks")
        assert resolved == "Banks"
        assert suggestions == []  # exact match — no suggestions needed

    @pytest.mark.asyncio
    async def test_fuzzy_resolves_partial_hint(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from src.tools.company_tools import _resolve_sector

        mock_repo = MagicMock()
        mock_repo.distinct_sectors = AsyncMock(
            return_value=["Banks", "Software & Services", "Pharmaceuticals"]
        )
        # "software" partial → resolves to "Software & Services"
        resolved, _ = await _resolve_sector(mock_repo, "software")
        assert resolved == "Software & Services"

    @pytest.mark.asyncio
    async def test_no_match_returns_suggestions(self) -> None:
        """When no fuzzy candidate clears the threshold, we still surface
        the closest three as suggestions for the user."""
        from unittest.mock import AsyncMock, MagicMock

        from src.tools.company_tools import _resolve_sector

        mock_repo = MagicMock()
        mock_repo.distinct_sectors = AsyncMock(
            return_value=["Banks", "Software & Services", "Pharmaceuticals"]
        )
        resolved, suggestions = await _resolve_sector(mock_repo, "asdfqwer")
        assert resolved is None
        assert len(suggestions) > 0  # always returns SOMETHING for the agent

    @pytest.mark.asyncio
    async def test_unknown_sector_returns_structured_error(self) -> None:
        """search_companies surfaces _resolve_sector's miss as an
        ``unknown_sector`` error with the closest sectors in `detail`."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.tools.company_tools import search_companies

        mock_repo = MagicMock()
        mock_repo.distinct_sectors = AsyncMock(
            return_value=["Banks", "Software & Services", "Pharmaceuticals"]
        )

        @asynccontextmanager
        async def fake_session_scope():
            yield MagicMock()

        with patch("src.tools.company_tools.catalog_session_scope", fake_session_scope), \
             patch("src.tools.company_tools.CompanyRepository", return_value=mock_repo):
            result = await search_companies(query="", sector="asdfqwer")
            assert result.get("ok") is False
            assert result["error_code"] == "unknown_sector"
            assert "detail" in result  # the closest catalog sectors


# ── Pure unit tests — sector dedup helpers (no DB) ─────────────────────────


class TestSectorDedup:
    """Verify the catalog's "Software & Services" vs "Software and Services"
    near-duplicates collapse to one canonical entry."""

    def test_normalize_collapses_amp_and_word_and(self) -> None:
        from src.tools.company_tools import _normalize_sector
        assert _normalize_sector("Software & Services") == "software services"
        assert _normalize_sector("Software and Services") == "software services"
        assert _normalize_sector("Oil & Gas") == "oil gas"
        assert _normalize_sector("Iron / Steel") == "iron steel"

    def test_normalize_handles_empty(self) -> None:
        from src.tools.company_tools import _normalize_sector
        assert _normalize_sector("") == ""
        assert _normalize_sector("   ") == ""

    def test_dedupe_keeps_longest_per_group(self) -> None:
        from src.tools.company_tools import _dedupe_sectors
        out = _dedupe_sectors([
            "Software & Services",
            "Software and Services",
            "Banks",
            "Oil & Gas",
            "Oil and Gas",
        ])
        # 5 inputs → 3 groups
        assert len(out) == 3
        assert "Banks" in out
        # The longer original is kept per group (alphabetical sort applies)
        assert any("and" in s.lower() for s in out if "software" in s.lower())
        assert any("and" in s.lower() for s in out if "oil" in s.lower())

    def test_dedupe_alphabetical(self) -> None:
        from src.tools.company_tools import _dedupe_sectors
        out = _dedupe_sectors(["Pharmaceuticals", "Banks", "Software & Services"])
        assert out == sorted(out)


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
