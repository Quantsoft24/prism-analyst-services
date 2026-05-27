"""Tests for the agent runner's structured-error wiring, freshness emission,
and structured-final-answer parsing.

These are pure unit tests — no DB, no ADK. They exercise the helpers the
runner uses to translate ADK events into PRISM's typed SSE stream. The
end-to-end ADK integration is covered by tests/test_chat_agent_integration.py.
"""

from __future__ import annotations

import pytest

from src.integrations.tools._errors import (
    extract_error_message,
    is_error,
    is_retriable,
    make_error,
)
from src.schemas.chat import FinalAnswer
from src.services.agent_runner import (
    _freshness_source_label,
    _split_structured_answer,
    _summarize_tool_response,
)

# ── make_error / is_error / is_retriable ───────────────────────────────────


class TestErrorHelpers:
    def test_make_error_builds_canonical_shape(self) -> None:
        err = make_error(
            message="stock-chat is down",
            code="stock_chat_unreachable",
            next_action="ask_user_to_retry_later",
            retriable=True,
        )
        assert err == {
            "ok": False,
            "error": "stock-chat is down",
            "error_code": "stock_chat_unreachable",
            "next_action": "ask_user_to_retry_later",
            "retriable": True,
        }

    def test_make_error_truncates_long_detail(self) -> None:
        long = "x" * 1000
        err = make_error(
            message="m", code="c", next_action="give_up_gracefully", detail=long
        )
        assert len(err["detail"]) == 500

    def test_make_error_omits_detail_when_none(self) -> None:
        err = make_error(message="m", code="c", next_action="give_up_gracefully")
        assert "detail" not in err

    def test_is_error_on_new_shape(self) -> None:
        assert is_error({"ok": False, "error": "boom"}) is True

    def test_is_error_on_legacy_bare_error(self) -> None:
        # Older tools still return this — runner must detect them.
        assert is_error({"error": "old style"}) is True

    def test_is_error_on_success_shape(self) -> None:
        assert is_error({"items": [], "total": 0}) is False
        assert is_error({"ok": True}) is False
        assert is_error({}) is False

    def test_is_error_on_non_dict(self) -> None:
        assert is_error(None) is False
        assert is_error("oops") is False

    def test_is_retriable_only_when_flag_set(self) -> None:
        assert is_retriable({"ok": False, "retriable": True}) is True
        assert is_retriable({"ok": False, "retriable": False}) is False
        assert is_retriable({"ok": False}) is False

    def test_extract_error_message_returns_none_on_success(self) -> None:
        assert extract_error_message({"items": []}) is None

    def test_extract_error_message_picks_up_new_shape(self) -> None:
        assert extract_error_message({"ok": False, "error": "boom"}) == "boom"

    def test_extract_error_message_picks_up_legacy_shape(self) -> None:
        assert extract_error_message({"error": "legacy boom"}) == "legacy boom"


# ── _summarize_tool_response ───────────────────────────────────────────────


class TestSummarizeToolResponse:
    def test_items_array(self) -> None:
        assert (
            _summarize_tool_response({"items": [1, 2, 3], "total": 10})
            == "3 of 10 item(s)"
        )

    def test_items_with_suggestions_chip(self) -> None:
        summary = _summarize_tool_response(
            {"items": [], "total": 0, "suggestions": [{"ticker": "RELIANCE"}]}
        )
        assert "near-match" in summary

    def test_filings_array(self) -> None:
        assert (
            _summarize_tool_response({"filings": [1, 2], "total": 5})
            == "2 of 5 filing(s)"
        )

    def test_bmc_blocks(self) -> None:
        assert "block" in _summarize_tool_response({"blocks": [1] * 9})

    def test_lookup_hit(self) -> None:
        assert "found" in _summarize_tool_response(
            {"found": True, "name": "Reliance", "ticker": "RELIANCE"}
        )

    def test_lookup_miss_with_suggestions(self) -> None:
        summary = _summarize_tool_response(
            {"found": False, "suggestions": [{"ticker": "RELIANCE"}]}
        )
        assert "suggestion" in summary

    def test_lookup_miss_clean(self) -> None:
        assert _summarize_tool_response({"found": False}) == "not found"

    def test_filings_answer(self) -> None:
        summary = _summarize_tool_response(
            {"answer": "Reliance reported strong growth in Q4...", "evidence": []}
        )
        assert summary.startswith("answer:")

    def test_compute_result(self) -> None:
        assert (
            _summarize_tool_response({"result": 12.5, "unit": "%"}) == "= 12.5%"
        )


# ── _freshness_source_label ────────────────────────────────────────────────


class TestFreshnessSourceLabel:
    @pytest.mark.parametrize(
        "tool,expected",
        [
            ("stock_filings_read", "filings catalog"),
            ("stock_filings_lookup", "filings catalog"),
            ("stock_technicals", "market data"),
            ("bmc_get", "business model canvas"),
            ("bmc_generate", "business model canvas"),
            ("web_search", "web search"),
        ],
    )
    def test_known_tools(self, tool, expected) -> None:
        assert _freshness_source_label(tool) == expected

    def test_unknown_tool_falls_back_to_name(self) -> None:
        assert _freshness_source_label("custom_tool") == "custom_tool"


# ── _split_structured_answer ───────────────────────────────────────────────


class TestSplitStructuredAnswer:
    def test_plain_prose_returns_unchanged(self) -> None:
        prose, structured = _split_structured_answer("Hello world.")
        assert prose == "Hello world."
        assert structured is None

    def test_empty_string(self) -> None:
        prose, structured = _split_structured_answer("")
        assert prose == ""
        assert structured is None

    def test_well_formed_meta_block(self) -> None:
        raw = (
            "TCS is a software company.\n\n"
            '<answer_meta>{"confidence":"high","data_freshness":"2026-Q4",'
            '"citations":[{"label":"TCS Q4 FY24, p.5","source_kind":"filing"}]}'
            "</answer_meta>"
        )
        prose, structured = _split_structured_answer(raw)
        assert prose == "TCS is a software company."
        assert isinstance(structured, FinalAnswer)
        assert structured.confidence == "high"
        assert structured.data_freshness == "2026-Q4"
        assert len(structured.citations) == 1
        assert structured.citations[0].label == "TCS Q4 FY24, p.5"

    def test_malformed_json_falls_back_to_prose(self) -> None:
        raw = "Answer here.\n<answer_meta>{not json}</answer_meta>"
        prose, structured = _split_structured_answer(raw)
        assert prose == raw  # untouched
        assert structured is None

    def test_block_with_only_confidence(self) -> None:
        raw = 'Short answer.\n<answer_meta>{"confidence":"low"}</answer_meta>'
        prose, structured = _split_structured_answer(raw)
        assert structured is not None
        assert structured.confidence == "low"
        assert structured.citations == []

    def test_block_not_at_end_is_ignored(self) -> None:
        # Anchor-to-end means a tag mid-response isn't accidentally parsed.
        raw = "Mid <answer_meta>{}</answer_meta> tail text"
        prose, structured = _split_structured_answer(raw)
        assert prose == raw
        assert structured is None

    def test_invalid_confidence_value_drops_block(self) -> None:
        # Pydantic Literal validation should reject and we fall back to prose.
        raw = 'A.\n<answer_meta>{"confidence":"reckless"}</answer_meta>'
        prose, structured = _split_structured_answer(raw)
        assert prose == raw
        assert structured is None
