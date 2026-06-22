"""Tests for the agent runner's structured-error wiring, freshness emission,
structured-final-answer parsing, and the agentic-experience helpers
(token chunker, initial-plan thought heuristic, auto-chart from time-series).

These are pure unit tests — no DB, no ADK. They exercise the helpers the
runner uses to translate ADK events into PRISM's typed SSE stream. The
end-to-end ADK integration is covered by tests/test_chat_agent_integration.py.
"""

from __future__ import annotations

import asyncio

import pytest

from src.integrations.tools._errors import (
    extract_error_message,
    is_error,
    is_retriable,
    make_error,
)
from src.schemas.chat import ChartEvent, FinalAnswer
from src.services.agent_runner import (
    _freshness_source_label,
    _initial_plan_thought,
    _is_stall_response,
    _split_structured_answer,
    _summarize_tool_response,
    _TokenChunker,
    _trim_response_for_rescue,
    _try_emit_chart,
    _validate_structured_freshness,
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

    # ── financials_query shape — new operation/status branches (2026-06) ──

    def test_financials_lookup_summary(self) -> None:
        resp = {
            "status": "ok", "operation": "lookup", "value": 65446.5, "period": "FY2024",
            "field": {"key": "net_profit", "label": "Net Profit (PAT)", "unit": "₹ cr"},
        }
        assert _summarize_tool_response(resp) == "Net Profit (PAT) FY2024"

    def test_financials_trend_summary(self) -> None:
        resp = {
            "status": "ok", "operation": "trend",
            "series": [{"period": "FY2022", "value": 1}, {"period": "FY2023", "value": 2}],
        }
        assert _summarize_tool_response(resp) == "trend · 2 pts"

    def test_financials_compare_summary(self) -> None:
        # The service tags compares as operation:"lookup" but carries `comparison`.
        resp = {
            "status": "ok", "operation": "lookup",
            "comparison": [{"name": "TCS", "value": 19.1}, {"name": "Infosys", "value": 16.4}],
        }
        assert _summarize_tool_response(resp) == "compared 2 cos"

    def test_financials_rank_summary(self) -> None:
        resp = {"status": "ok", "operation": "rank", "ranking": [{"name": "A"}, {"name": "B"}, {"name": "C"}]}
        assert _summarize_tool_response(resp) == "ranked top 3"

    def test_financials_statement_summary(self) -> None:
        resp = {"status": "ok", "operation": "statement", "line_items": [{"key": "x"}, {"key": "y"}]}
        assert _summarize_tool_response(resp) == "statement · 2 items"

    def test_financials_needs_clarification_counts_suggestions(self) -> None:
        resp = {
            "status": "needs_clarification", "needs_clarification": True,
            "suggestions": [{"key": "cash", "label": "Cash"}, {"key": "rf", "label": "Reserves"}],
        }
        summary = _summarize_tool_response(resp)
        assert "needs clarification" in summary
        assert "2 suggestion" in summary

    def test_financials_no_data(self) -> None:
        resp = {"status": "no_data", "operation": "lookup", "answer": "No value."}
        assert _summarize_tool_response(resp) == "no data"


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
        # Behavior updated 2026-05-28: on JSON parse failure, we still
        # defensively STRIP the meta tags from the prose so the raw block
        # never reaches the user as visible text. Old behavior leaked the
        # raw tags through.
        raw = "Answer here.\n<answer_meta>{not json}</answer_meta>"
        prose, structured = _split_structured_answer(raw)
        assert structured is None
        assert "<answer_meta>" not in prose
        assert "</answer_meta>" not in prose
        assert prose.strip() == "Answer here."

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

    def test_invalid_confidence_value_coerced_not_rejected(self) -> None:
        # Behavior updated 2026-05-28: rather than reject the whole payload
        # over a vocabulary mismatch, unknown confidence values are coerced
        # to "medium" (the schema default). The structured payload survives;
        # the user gets a Report tab instead of bare prose.
        raw = 'A.\n<answer_meta>{"confidence":"reckless"}</answer_meta>'
        prose, structured = _split_structured_answer(raw)
        assert structured is not None
        assert structured.confidence == "medium"
        assert "<answer_meta>" not in prose

    def test_nested_json_block_parses(self) -> None:
        # REGRESSION (2026-05-28): the previous regex used ``\{.*?\}`` which
        # truncated nested JSON at the FIRST inner ``}``, leaving the block
        # in the prose. The current regex is anchored end-of-string and
        # captures everything between the tags. The exact JSON below is the
        # shape we observed in production from Gemini Flash.
        raw = (
            'Reliance reported PAT of ₹80,787 crore.\n\n'
            '<answer_meta>{ "confidence": "high", "data_freshness": "2025-03-31", '
            '"citations": [ {"label": "Reliance Annual PnL 2025", '
            '"source_kind": "filing", "as_of": "2025-03-31"} ] }</answer_meta>'
        )
        prose, structured = _split_structured_answer(raw)
        assert structured is not None
        assert "<answer_meta>" not in prose
        assert "</answer_meta>" not in prose
        assert structured.confidence == "high"
        assert structured.data_freshness == "2025-03-31"
        assert len(structured.citations) == 1
        assert structured.citations[0].label == "Reliance Annual PnL 2025"

    def test_unknown_source_kind_is_coerced_not_rejected(self) -> None:
        # PRODUCTION BUG (2026-05-28): the LLM wrote ``source_kind: "financials"``
        # which isn't in the strict Literal. Previously the whole structured
        # payload was rejected and the raw meta block leaked into the prose.
        # Now we coerce to "tool" (the safe default) and the payload survives.
        raw = (
            "Profit was strong.\n"
            '<answer_meta>{"confidence":"high",'
            '"citations":[{"label":"X","source_kind":"financials","as_of":"2025-03-31"}]}'
            '</answer_meta>'
        )
        prose, structured = _split_structured_answer(raw)
        assert structured is not None
        assert "<answer_meta>" not in prose
        assert structured.citations[0].source_kind == "tool"
        assert structured.citations[0].label == "X"

    def test_unknown_confidence_coerced_to_medium(self) -> None:
        raw = 'A.\n<answer_meta>{"confidence":"unsure"}</answer_meta>'
        prose, structured = _split_structured_answer(raw)
        assert structured is not None
        assert structured.confidence == "medium"

    def test_unknown_section_kind_coerced(self) -> None:
        raw = (
            'A.\n<answer_meta>{"sections":'
            '[{"title":"Note","body":"hi","kind":"weird"}]}</answer_meta>'
        )
        prose, structured = _split_structured_answer(raw)
        assert structured is not None
        assert structured.sections[0].kind == "summary"

    def test_sections_missing_required_fields_dropped(self) -> None:
        raw = (
            'A.\n<answer_meta>{"sections":'
            '[{"title":"OK","body":"yes","kind":"summary"},'
            ' {"title":"MissingBody"},'
            ' {"body":"MissingTitle"}]}</answer_meta>'
        )
        prose, structured = _split_structured_answer(raw)
        assert structured is not None
        assert len(structured.sections) == 1
        assert structured.sections[0].title == "OK"

    def test_malformed_kpis_filtered(self) -> None:
        # KPIs require both label + value. Anything missing should be dropped
        # silently (don't reject the whole payload).
        raw = (
            'A.\n<answer_meta>{"kpis":'
            '[{"label":"PAT","value":"₹80,787 cr"},'
            ' {"label":"NoValue"},'
            ' "not even a dict",'
            ' {"value":"NoLabel"}]}</answer_meta>'
        )
        prose, structured = _split_structured_answer(raw)
        assert structured is not None
        assert len(structured.kpis) == 1
        assert structured.kpis[0].label == "PAT"

    def test_trailing_sources_line_stripped_from_prose(self) -> None:
        # Even if Gemini ignores the prompt's "don't add a Sources line"
        # rule, the parser strips it as a backstop. The right-pane Sources
        # tab is the canonical attribution surface.
        raw = (
            "Reliance reported PAT of ₹80,787 crore.\n\n"
            "Sources: financials_query\n"
            '<answer_meta>{"confidence":"high"}</answer_meta>'
        )
        prose, _ = _split_structured_answer(raw)
        assert "Sources:" not in prose
        assert "Reliance" in prose

    def test_trailing_sources_line_with_bold_marker_stripped(self) -> None:
        raw = "Text here.\n\n**Sources:** financials_query, web_search"
        prose, _ = _split_structured_answer(raw)
        assert "Sources" not in prose

    def test_trailing_sources_bullet_stripped(self) -> None:
        raw = "Text.\n\n- Sources: financials_query"
        prose, _ = _split_structured_answer(raw)
        assert "Sources" not in prose

    def test_malformed_meta_block_never_leaks_to_prose(self) -> None:
        # Even when the JSON is unparseable, the user must never see raw
        # ``<answer_meta>...</answer_meta>`` tags as visible text.
        raw = (
            "Answer text.\n"
            "<answer_meta>{ not valid json at all !!! }</answer_meta>"
        )
        prose, structured = _split_structured_answer(raw)
        assert structured is None
        assert "<answer_meta>" not in prose
        assert "</answer_meta>" not in prose
        assert prose.strip() == "Answer text."

    def test_meta_only_response_yields_empty_prose_but_keeps_structured(self) -> None:
        # PRODUCTION REGRESSION (2026-05-28 evening): Gemini Flash, after the
        # "<answer_meta> required" prompt change, sometimes emits ONLY the
        # meta block with no prose before it. The parser must still extract
        # the structured payload (so the Report tab is populated); the
        # runner's empty-prose fallback handles promoting a section body
        # to prose. This test pins the parser's contract for that case.
        raw = (
            '<answer_meta>{"confidence":"high","data_freshness":"2025-03-31",'
            '"citations":[{"label":"Reliance Annual PnL 2025",'
            '"source_kind":"filing","as_of":"2025-03-31"}],'
            '"sections":[{"title":"Executive summary","kind":"summary",'
            '"body":"Reliance reported PAT of ₹80,787 crore for FY25."}]}'
            '</answer_meta>'
        )
        prose, structured = _split_structured_answer(raw)
        # No prose was emitted before the block — the user got nothing in
        # the chat thread. The runner's layered fallback (in agent_runner.py)
        # is responsible for promoting sections[0].body to prose in that case.
        assert prose == ""
        assert structured is not None
        assert structured.confidence == "high"
        assert len(structured.sections) == 1
        assert "Reliance" in structured.sections[0].body
        assert len(structured.citations) == 1

    def test_kpis_and_sections_round_trip(self) -> None:
        # Newly wired (2026-05-28): kpis + sections must flow through.
        # Previously they were silently dropped — Report tab stayed empty.
        raw = (
            'TCS profit was strong.\n'
            '<answer_meta>{'
            '"confidence":"high",'
            '"data_freshness":"2025-03-31",'
            '"kpis":[{"label":"PAT","value":"₹80,787 cr","unit":"cr","cite_label":"src 1"}],'
            '"sections":[{"title":"Executive summary","kind":"summary","body":"PAT ₹80,787 cr."}],'
            '"citations":[{"label":"Reliance Q4 FY25","source_kind":"filing"}]'
            '}</answer_meta>'
        )
        prose, structured = _split_structured_answer(raw)
        assert structured is not None
        assert len(structured.kpis) == 1
        assert structured.kpis[0].label == "PAT"
        assert structured.kpis[0].value == "₹80,787 cr"
        assert len(structured.sections) == 1
        assert structured.sections[0].kind == "summary"
        assert "PAT" in structured.sections[0].body


# ── _TokenChunker (T1.2: smooth token streaming) ─────────────────────────────


class TestTokenChunker:
    """Server-side re-chunking gives the UI a smooth typing cadence even when
    ADK drops the whole answer in one big text part."""

    @staticmethod
    async def _collect(text: str) -> list[str]:
        return [chunk async for chunk in _TokenChunker.stream(text)]

    def test_empty_text_yields_nothing(self) -> None:
        chunks = asyncio.run(self._collect(""))
        assert chunks == []

    def test_short_text_passes_through_unchanged(self) -> None:
        # Anything <= _PASSTHROUGH (120) is one chunk, no splitting.
        text = "Reliance reported ₹80,787 cr PAT for FY25."
        chunks = asyncio.run(self._collect(text))
        assert chunks == [text]

    def test_long_text_is_chunked_at_word_boundaries(self) -> None:
        # 600 chars of lorem-ish — must split but never break a word.
        text = (
            "Reliance Industries delivered ₹80,787 crore in profit after tax for FY25, "
            "up 12% YoY despite a softer refining margin environment. The board "
            "recommended a final dividend of ₹10 per share and reaffirmed its capex "
            "guidance for the New Energy business. Jio added 12.4 million net subscribers "
            "in Q4 and ARPU expanded to ₹181.7. Retail gross revenue grew 17% YoY led by "
            "fashion and grocery. Management noted that O2C margins should normalise in "
            "H1 FY26 as crack spreads recover from the seasonal low."
        )
        chunks = asyncio.run(self._collect(text))
        assert len(chunks) > 1
        # Concatenation must reproduce the original byte-for-byte.
        assert "".join(chunks) == text
        # No chunk should be empty.
        assert all(len(c) > 0 for c in chunks)
        # No chunk except possibly the last should end mid-word — i.e. each
        # non-final chunk should end with whitespace OR with a punctuation
        # character that's a sane line break, OR be exactly _TARGET_LEN
        # (hard-cut fallback).
        for chunk in chunks[:-1]:
            ends_clean = chunk.endswith((" ", "\n", ",", ".", ";")) or len(chunk) == _TokenChunker._TARGET_LEN
            assert ends_clean, f"chunk ends mid-word: {chunk!r}"

    def test_chunks_have_delay_between_them(self) -> None:
        # The streaming feel comes from sleeping between chunks. We assert
        # the total time is at least (n_chunks - 1) * _DELAY_S (minus a
        # generous fudge for event-loop variance).
        text = "x" * 400  # forces multiple chunks via hard-cut path
        loop = asyncio.new_event_loop()
        try:
            start = loop.time()
            chunks = loop.run_until_complete(self._collect(text))
            elapsed = loop.time() - start
        finally:
            loop.close()
        # 400 chars / ~80 per chunk → ~5 chunks, ~4 sleeps × 0.035s = ~0.14s
        # Be lenient: at least 60% of the theoretical minimum.
        expected_min = (len(chunks) - 1) * _TokenChunker._DELAY_S * 0.6
        assert elapsed >= expected_min, f"too fast: {elapsed:.3f}s for {len(chunks)} chunks"


# ── _initial_plan_thought (T2.2: synthetic plan) ─────────────────────────────


class TestInitialPlanThought:
    """The opening plan thought must be honest — never name a specific tool
    or commit to a path, since the LLM hasn't chosen yet."""

    def test_empty_input_returns_safe_default(self) -> None:
        assert _initial_plan_thought("") == "Let me work on this."

    def test_comparison_triggers_compare_phrasing(self) -> None:
        assert "side by side" in _initial_plan_thought("compare TCS and Infosys")
        assert "side by side" in _initial_plan_thought("TCS vs Wipro")

    def test_filings_keywords_route_to_filings_phrasing(self) -> None:
        assert "filings" in _initial_plan_thought("what did Reliance say in the annual report")
        assert "filings" in _initial_plan_thought("disclosures of HDFC Bank")
        assert "filings" in _initial_plan_thought("board meeting outcomes for TCS")

    def test_numeric_keywords_route_to_financial_phrasing(self) -> None:
        assert "financial data" in _initial_plan_thought("what is the profit of reliance in 2025")
        assert "financial data" in _initial_plan_thought("5-year revenue CAGR of Infosys")
        assert "financial data" in _initial_plan_thought("debt to equity of Vedanta")

    def test_market_keywords_route_to_market_phrasing(self) -> None:
        assert "market data" in _initial_plan_thought("current price of TCS")
        assert "market data" in _initial_plan_thought("RSI for HDFC Bank")

    def test_bmc_keywords_route_to_canvas_phrasing(self) -> None:
        assert "business model canvas" in _initial_plan_thought("show me the business model of TCS")
        assert "business model canvas" in _initial_plan_thought("BMC of Reliance")

    def test_no_specific_tool_named(self) -> None:
        # Phrasings must NEVER promise a specific tool — the agent might
        # choose differently and we'd have lied. This sanity-checks all
        # the templates above.
        forbidden = ("resolve_company", "stock_filings_read", "financials_query",
                     "bmc_get", "stock_technicals", "web_search")
        for msg in [
            "compare TCS and Infosys", "what is the profit of reliance",
            "current price of TCS", "BMC of Reliance",
            "annual report of HDFC Bank", "show me banks",
            "anything random here",
        ]:
            out = _initial_plan_thought(msg)
            for tool in forbidden:
                assert tool not in out, f"plan thought {out!r} names tool {tool!r}"


# ── _try_emit_chart (T2.1: auto-chart from time-series) ──────────────────────


class TestTryEmitChart:
    """Conservative auto-chart: a rendered chart must always be 'right' or
    not appear at all. False negatives are fine, false positives are not."""

    def test_non_financials_tool_no_chart(self) -> None:
        assert _try_emit_chart("stock_technicals", "c1", {"rows": [
            {"period_end": "2024-01-01", "value": 1},
            {"period_end": "2024-02-01", "value": 2},
            {"period_end": "2024-03-01", "value": 3},
        ]}) is None

    def test_non_dict_response_no_chart(self) -> None:
        assert _try_emit_chart("financials_query", "c1", None) is None
        assert _try_emit_chart("financials_query", "c1", "boom") is None

    def test_too_few_rows_no_chart(self) -> None:
        rows = [
            {"period_end": "2024-03-31", "revenue": 100},
            {"period_end": "2025-03-31", "revenue": 110},
        ]
        assert _try_emit_chart("financials_query", "c1", {"rows": rows}) is None

    def test_missing_period_end_no_chart(self) -> None:
        rows = [
            {"revenue": 100}, {"revenue": 110}, {"revenue": 120},
        ]
        assert _try_emit_chart("financials_query", "c1", {"rows": rows}) is None

    def test_no_numeric_column_no_chart(self) -> None:
        rows = [
            {"period_end": "2023-03-31", "note": "n1"},
            {"period_end": "2024-03-31", "note": "n2"},
            {"period_end": "2025-03-31", "note": "n3"},
        ]
        assert _try_emit_chart("financials_query", "c1", {"rows": rows}) is None

    def test_meta_columns_are_not_chart_targets(self) -> None:
        # company_id is numeric but is in _NON_CHART_COLUMNS — must not chart.
        rows = [
            {"period_end": "2023-03-31", "company_id": 4193},
            {"period_end": "2024-03-31", "company_id": 4193},
            {"period_end": "2025-03-31", "company_id": 4193},
        ]
        assert _try_emit_chart("financials_query", "c1", {"rows": rows}) is None

    def test_happy_path_emits_well_formed_chart(self) -> None:
        rows = [
            {"period_end": "2023-03-31", "revenue": 90791.0},
            {"period_end": "2024-03-31", "revenue": 100472.0},
            {"period_end": "2025-03-31", "revenue": 110180.0},
        ]
        chart = _try_emit_chart("financials_query", "call-xyz", {"rows": rows})
        assert chart is not None
        assert isinstance(chart, ChartEvent)
        assert chart.call_id == "call-xyz"
        assert chart.kind == "line"
        assert len(chart.points) == 3
        # x-axis sorted ascending by period_end
        assert chart.points[0].x == "2023-03-31"
        assert chart.points[-1].x == "2025-03-31"
        assert chart.points[-1].y == 110180.0
        assert chart.delta_kind == "pos"
        assert "21" in chart.current_delta or "+21" in chart.current_delta

    def test_negative_trend_marked_neg(self) -> None:
        rows = [
            {"period_end": "2023-03-31", "value": 100.0},
            {"period_end": "2024-03-31", "value": 80.0},
            {"period_end": "2025-03-31", "value": 60.0},
        ]
        chart = _try_emit_chart("financials_query", "c1", {"rows": rows})
        assert chart is not None
        assert chart.delta_kind == "neg"

    def test_flat_trend_marked_neutral(self) -> None:
        rows = [
            {"period_end": "2023-03-31", "value": 100.0},
            {"period_end": "2024-03-31", "value": 100.0},
            {"period_end": "2025-03-31", "value": 100.0},
        ]
        chart = _try_emit_chart("financials_query", "c1", {"rows": rows})
        assert chart is not None
        assert chart.delta_kind == "neutral"

    def test_rows_in_random_order_get_sorted(self) -> None:
        # The underlying response order may not be chronological — we MUST
        # sort by period_end for the chart to make sense.
        rows = [
            {"period_end": "2025-03-31", "revenue": 110180.0},
            {"period_end": "2023-03-31", "revenue": 90791.0},
            {"period_end": "2024-03-31", "revenue": 100472.0},
        ]
        chart = _try_emit_chart("financials_query", "c1", {"rows": rows})
        assert chart is not None
        assert [p.x for p in chart.points] == ["2023-03-31", "2024-03-31", "2025-03-31"]

    def test_bool_is_not_treated_as_numeric(self) -> None:
        # Python's bool is a subclass of int; we explicitly filter it out so
        # is_listed=True/True/False never charts.
        rows = [
            {"period_end": "2023-03-31", "is_listed": True},
            {"period_end": "2024-03-31", "is_listed": True},
            {"period_end": "2025-03-31", "is_listed": False},
        ]
        assert _try_emit_chart("financials_query", "c1", {"rows": rows}) is None


# ── _validate_structured_freshness (anti-hallucination guard) ────────────────


class TestValidateStructuredFreshness:
    """Defence-in-depth against Gemini fabricating a `data_freshness` date
    from training data when no tool actually returned one this turn."""

    @staticmethod
    def _payload(freshness: str | None) -> FinalAnswer:
        return FinalAnswer(
            text="prose",
            citations=[],
            confidence="high",
            data_freshness=freshness,
        )

    def test_none_structured_returns_none(self) -> None:
        assert _validate_structured_freshness(None, {"2025-03-31"}) is None

    def test_unset_freshness_preserved(self) -> None:
        before = self._payload(None)
        after = _validate_structured_freshness(before, {"2025-03-31"})
        assert after is before
        assert after.data_freshness is None

    def test_matching_freshness_preserved(self) -> None:
        before = self._payload("2025-03-31")
        after = _validate_structured_freshness(before, {"2025-03-31", "live"})
        # Same instance + same value — no unnecessary copy on the happy path.
        assert after is before
        assert after.data_freshness == "2025-03-31"

    def test_fabricated_freshness_dropped_to_none(self) -> None:
        # Gemini wrote "2025-05-15" but no tool emitted that — drop it.
        before = self._payload("2025-05-15")
        after = _validate_structured_freshness(before, {"2025-03-31"})
        assert after is not None
        assert after.data_freshness is None
        # The rest of the payload must be preserved verbatim.
        assert after.confidence == "high"
        assert after.text == "prose"

    def test_fabricated_when_no_tool_emitted_freshness(self) -> None:
        # Common case: only `list_covered_sectors` ran; observed is empty.
        before = self._payload("2025-05-15")
        after = _validate_structured_freshness(before, set())
        assert after is not None
        assert after.data_freshness is None

    def test_live_label_preserved_when_tool_emitted_it(self) -> None:
        # `stock_technicals` emits the literal string "live" — must pass.
        before = self._payload("live")
        after = _validate_structured_freshness(before, {"live"})
        assert after.data_freshness == "live"


# ── _trim_response_for_rescue (audit log + rescue prompt sizing) ─────────────


class TestTrimResponseForRescue:
    """Tool responses can be huge (200 rows, multi-page PDF text). We trim
    them so the audit row + rescue-call context stay bounded."""

    def test_non_dict_passes_through(self) -> None:
        assert _trim_response_for_rescue("hello") == "hello"
        assert _trim_response_for_rescue(42) == 42
        assert _trim_response_for_rescue(None) is None

    def test_short_dict_unchanged(self) -> None:
        resp = {"rows": [{"x": 1}, {"x": 2}], "sql": "SELECT 1"}
        out = _trim_response_for_rescue(resp)
        assert out == resp

    def test_long_list_truncated_with_marker(self) -> None:
        rows = [{"i": i} for i in range(50)]
        out = _trim_response_for_rescue({"rows": rows}, max_rows=20)
        # 20 real rows + 1 marker entry
        assert len(out["rows"]) == 21
        assert out["rows"][:20] == rows[:20]
        assert "+30 more" in str(out["rows"][20])

    def test_long_string_cropped(self) -> None:
        big = "x" * 5000
        out = _trim_response_for_rescue({"answer": big}, max_str=2000)
        assert len(out["answer"]) == 2003  # 2000 + "..."
        assert out["answer"].endswith("...")

    def test_keys_preserved(self) -> None:
        resp = {"a": 1, "b": [1, 2, 3], "c": "short"}
        out = _trim_response_for_rescue(resp)
        assert set(out.keys()) == {"a", "b", "c"}


# ── _rescue_empty_synthesis (Pro single-shot when prose is empty) ────────────


class TestRescueEmptySynthesis:
    """The rescue path fires when the orchestrator skips prose synthesis.
    These tests cover the deterministic guards (no key, no trace, empty
    response) — the actual Pro call is mocked at the litellm boundary."""

    @staticmethod
    async def _run(monkeypatch, trace, **kwargs):
        # Default-stub the gemini key so the API-key guard doesn't fire
        # unless the test wants it to. ``gemini_api_keys`` is a computed
        # property (reads GEMINI_API_KEY / GEMINI_API_KEY_1..4); set the
        # underlying field directly instead of trying to assign the prop.
        if "api_keys" in kwargs:
            keys = kwargs.pop("api_keys")
        else:
            keys = ["test-key"]
        from src.services import agent_runner as ar
        monkeypatch.setattr(ar.settings, "GEMINI_API_KEY", keys[0] if keys else "")
        for i, k in enumerate(["GEMINI_API_KEY_1", "GEMINI_API_KEY_2",
                               "GEMINI_API_KEY_3", "GEMINI_API_KEY_4",
                               "GEMINI_API_KEY_5", "GEMINI_API_KEY_6",
                               "GEMINI_API_KEY_7", "GEMINI_API_KEY_8"]):
            monkeypatch.setattr(ar.settings, k, keys[i + 1] if len(keys) > i + 1 else "")
        # Force the router-off path so these tests deterministically exercise the
        # direct single-shot fallback (the composer tries the quality tier first).
        import src.services.model_router as mr

        def _no_router():
            raise RuntimeError("router disabled in test")

        monkeypatch.setattr(mr, "get_router", _no_router)
        return await ar._compose_final_answer("the question", trace)

    def test_empty_trace_returns_none(self, monkeypatch) -> None:
        # No tools ran → nothing to compose from → skip rescue.
        result = asyncio.run(self._run(monkeypatch, []))
        assert result is None

    def test_no_api_key_returns_none(self, monkeypatch) -> None:
        # Can't call Pro without a key — caller falls back to deterministic.
        result = asyncio.run(self._run(
            monkeypatch,
            [{"tool": "financials_query", "args": {}, "response": {"rows": []}}],
            api_keys=[],
        ))
        assert result is None

    def test_trace_without_response_returns_none(self, monkeypatch) -> None:
        # Tool was called but we never recorded the response (shouldn't
        # happen in real flow, but defend against it).
        result = asyncio.run(self._run(
            monkeypatch,
            [{"tool": "financials_query", "args": {"q": "x"}}],
        ))
        assert result is None

    def test_successful_rescue_returns_pro_text(self, monkeypatch) -> None:
        # Mock litellm.acompletion to return a canned answer.
        captured: dict = {}

        async def fake_acompletion(**kwargs):
            captured.update(kwargs)
            return _make_litellm_response(
                "Reliance reported PAT of ₹80,787 cr in FY25 [Reliance | 2025-03-31]."
            )

        import litellm
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        trace = [{
            "tool": "financials_query",
            "args": {"question": "Reliance profit FY25"},
            "response": {"rows": [{"pat": 80787}]},
        }]
        result = asyncio.run(self._run(monkeypatch, trace))

        assert result is not None
        assert "Reliance" in result
        assert "80,787" in result
        # Sanity: Pro model selected, low temp, short prompt.
        assert captured["model"] == "gemini/gemini-2.5-pro"
        assert captured["temperature"] == 0.2
        # The original user question must reach the model.
        joined = " ".join(m["content"] for m in captured["messages"])
        assert "the question" in joined

    def test_rescue_strips_accidental_meta_block(self, monkeypatch) -> None:
        # Pro might still try to emit a meta tail; we strip it so the
        # prose we splice in is clean.
        async def fake_acompletion(**kwargs):
            return _make_litellm_response(
                "Reliance PAT was ₹80,787 cr.\n<answer_meta>{\"x\":1}</answer_meta>"
            )

        import litellm
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        result = asyncio.run(self._run(
            monkeypatch,
            [{"tool": "financials_query", "args": {}, "response": {"rows": [{}]}}],
        ))
        assert result is not None
        assert "<answer_meta>" not in result
        assert result.startswith("Reliance PAT")

    def test_rescue_litellm_exception_returns_none(self, monkeypatch) -> None:
        # Network error / quota / anything — caller falls back cleanly.
        async def fake_acompletion(**kwargs):
            raise RuntimeError("upstream down")

        import litellm
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        result = asyncio.run(self._run(
            monkeypatch,
            [{"tool": "financials_query", "args": {}, "response": {"rows": [{}]}}],
        ))
        assert result is None

    def test_rescue_empty_pro_response_returns_none(self, monkeypatch) -> None:
        # If Pro itself emits empty text, we don't pretend — fall back.
        async def fake_acompletion(**kwargs):
            return _make_litellm_response("")

        import litellm
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        result = asyncio.run(self._run(
            monkeypatch,
            [{"tool": "financials_query", "args": {}, "response": {"rows": [{}]}}],
        ))
        assert result is None

    def test_composer_only_uses_last_6_tools(self, monkeypatch) -> None:
        # Bound the prompt size by keeping at most 6 recent tool entries.
        captured_messages: list = []

        async def fake_acompletion(**kwargs):
            captured_messages.extend(kwargs["messages"])
            return _make_litellm_response("ok")

        import litellm
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        trace = [
            {"tool": f"tool_{i}", "args": {}, "response": {"r": i}}
            for i in range(8)
        ]
        asyncio.run(self._run(monkeypatch, trace))

        user_msg = next(m["content"] for m in captured_messages if m["role"] == "user")
        # Only the last 6 tools' names should appear in the user message.
        for i in range(2, 8):
            assert f"tool_{i}" in user_msg
        assert "tool_0" not in user_msg
        assert "tool_1" not in user_msg


# ── _is_stall_response (catches "I will re-run" type final answers) ─────────


class TestIsStallResponse:
    """Stall detection: prose that promises future tool calls but doesn't
    actually fire them. The detector must catch real stalls while leaving
    legitimate analyst answers alone — even ones that happen to mention
    a partial retry."""

    # ── Empty / no-stall-phrase cases (should NOT trigger) ──────────────

    def test_empty_string_is_not_stall(self) -> None:
        assert _is_stall_response("") is False

    def test_normal_answer_is_not_stall(self) -> None:
        prose = (
            "Reliance Industries reported PAT of ₹80,787 crore in FY25, "
            "up 12% YoY. Margins held steady at ~9.7%."
        )
        assert _is_stall_response(prose) is False

    def test_refusal_is_not_stall(self) -> None:
        # Refusals don't mention re-running — they're terminal.
        prose = "PRISM is a research analyst for Indian listed companies — I can't help with that."
        assert _is_stall_response(prose) is False

    def test_clarification_question_is_not_stall(self) -> None:
        prose = "Did you mean Reliance Industries (RELIANCE) or Reliance Power (RPOWER)?"
        assert _is_stall_response(prose) is False

    # ── Pure stall cases (SHOULD trigger) ────────────────────────────────

    def test_will_rerun_is_stall(self) -> None:
        # The exact failure pattern from production.
        prose = (
            "I am still retrieving the data for the comparison. The "
            "initial query did not return all the necessary information. "
            "I will re-run the query to gather the 5-year revenue CAGR "
            "and the net profit margin for FY25 for TCS, Infosys, Wipro, "
            "and HCLTech."
        )
        assert _is_stall_response(prose) is True

    def test_let_me_try_again_is_stall(self) -> None:
        prose = "The tool returned an unexpected shape. Let me try again with a clearer query."
        assert _is_stall_response(prose) is True

    def test_still_investigating_is_stall(self) -> None:
        prose = "Still investigating the financial details — one moment."
        assert _is_stall_response(prose) is True

    def test_let_me_gather_more_is_stall(self) -> None:
        prose = "I have partial data. Let me gather more from the filings."
        assert _is_stall_response(prose) is True

    # ── False-positive guard (substantive content overrides stall phrases) ─

    def test_stall_phrase_with_real_numbers_is_not_stall(self) -> None:
        # A real answer that ALSO mentions a retry → stays as-is.
        prose = (
            "I had to re-run the query once. TCS reported ₹2.46L crore "
            "in revenue for FY25 with a 24.6% EBIT margin."
        )
        assert _is_stall_response(prose) is False

    def test_stall_phrase_with_rupee_is_not_stall(self) -> None:
        prose = "Initial query did not return all rows, but the partial result shows ₹80,787 cr PAT."
        assert _is_stall_response(prose) is False

    def test_stall_phrase_with_fy_label_is_not_stall(self) -> None:
        prose = "Let me try again with FY24 included; FY25 already showed 12% growth."
        assert _is_stall_response(prose) is False

    def test_stall_phrase_with_percent_is_not_stall(self) -> None:
        prose = "I'll re-run for FY26, but FY25 net margin was 24.6%."
        assert _is_stall_response(prose) is False

    def test_stall_phrase_with_crore_unit_is_not_stall(self) -> None:
        prose = "Let me gather more data; TCS revenue was 2.46 lakh crore in FY25."
        assert _is_stall_response(prose) is False

    # ── "Give up" family (added 2026-05-29 from production wire log) ──

    def test_i_do_not_have_access_is_stall(self) -> None:
        # The exact failure pattern from the wire log:
        prose = (
            "I do not have access to comparative 5-year CAGR metrics or "
            "FY25 full-year financials for all requested companies."
        )
        assert _is_stall_response(prose) is True

    def test_i_am_unable_to_provide_is_stall(self) -> None:
        prose = (
            "I am unable to provide a complete side-by-side growth and "
            "margin analysis at this time."
        )
        assert _is_stall_response(prose) is True

    def test_currently_not_available_is_stall(self) -> None:
        prose = (
            "Comparative metrics for the requested peer set are currently "
            "not available through the available financial query tools."
        )
        assert _is_stall_response(prose) is True

    def test_could_not_be_fulfilled_is_stall(self) -> None:
        prose = (
            "The current request for comparative metrics could not be "
            "fulfilled as the automated query did not return a complete "
            "dataset."
        )
        assert _is_stall_response(prose) is True

    def test_my_current_database_does_not_is_stall(self) -> None:
        prose = (
            "While some historical data is available, my current database "
            "does not support a comprehensive multi-company cross-comparison."
        )
        assert _is_stall_response(prose) is True

    def test_give_up_phrase_with_real_numbers_is_not_stall(self) -> None:
        # If the model gave a partial-but-substantive answer, leave it alone.
        prose = (
            "TCS posted a net profit margin of 18.76% in FY24. I do not "
            "have access to FY25 figures for the full peer set."
        )
        assert _is_stall_response(prose) is False


def _make_litellm_response(content: str):
    """Tiny shim to construct the shape litellm.acompletion returns."""
    class _Msg:
        def __init__(self, c: str) -> None:
            self.content = c

    class _Choice:
        def __init__(self, c: str) -> None:
            self.message = _Msg(c)

    class _Resp:
        def __init__(self, c: str) -> None:
            self.choices = [_Choice(c)]

    return _Resp(content)


class TestMergeFilingCitations:
    """Page-exact filing citations parsed from inline [Company | p.N] markers in
    the prose, joined to selected_filings' pdf_link (the synthesise=true path)."""

    def _trace(self):
        # synthesise=true: evidence is empty; pdf_link lives in selected_filings.
        return [{
            "tool": "stock_filings_read",
            "response": {
                "selected_filings": [
                    {"company_name": "ITC Ltd", "pdf_link": "https://www.bseindia.com/x.pdf"},
                ],
                "evidence": [],
            },
        }]

    def test_evidence_is_primary_source(self):
        # synthesise=false: evidence carries citation+page+pdf_url directly — used
        # verbatim, no prose parsing or company-name join needed.
        from src.schemas.chat import Citation, FinalAnswer
        from src.services.agent_runner import _merge_filing_citations

        trace = [{"tool": "stock_filings_read", "response": {"selected_filings": [], "evidence": [
            {"citation": "[ITC Ltd | p.5]", "page": "5",
             "pdf_url": "https://www.bseindia.com/itc.pdf", "company_name": "ITC Ltd"},
            {"citation": "[ITC Ltd | p.42]", "page": "42",
             "pdf_url": "https://www.bseindia.com/itc.pdf", "company_name": "ITC Ltd"},
        ]}}]
        s = FinalAnswer(text="answer the agent composed [ITC Ltd | p.5]",
                        citations=[Citation(label="Web", url="https://ex.com", source_kind="web")])
        out = _merge_filing_citations(s, trace)
        fil = [c for c in out.citations if c.source_kind == "filing"]
        assert {c.page for c in fil} == {5, 42}
        assert all(c.url == "https://www.bseindia.com/itc.pdf" for c in fil)
        assert any(c.source_kind == "web" for c in out.citations)  # non-filing kept

    def test_builds_pageexact_citations_from_inline_markers(self):
        from src.schemas.chat import Citation, FinalAnswer
        from src.services.agent_runner import _merge_filing_citations

        # spaced initials ("I T C Ltd.") + "p. 5" / "p.17" variants must all match.
        s = FinalAnswer(
            text="Net Zero by 2050 [I T C Ltd. | p. 5]. Packaging shift [ITC Ltd | p.17].",
            citations=[Citation(label="Web note", url="https://ex.com", source_kind="web")],
        )
        out = _merge_filing_citations(s, self._trace())
        fil = [c for c in out.citations if c.source_kind == "filing"]
        assert {c.page for c in fil} == {5, 17}
        assert all(c.url == "https://www.bseindia.com/x.pdf" for c in fil)
        assert any(c.source_kind == "web" for c in out.citations)  # non-filing kept

    def test_enriches_citation_with_page_in_label(self):
        # The LLM's <answer_meta> citations sometimes carry the page in the LABEL
        # ("ITC Ltd p. 7") rather than inline in the prose — enrich those too.
        from src.schemas.chat import Citation, FinalAnswer
        from src.services.agent_runner import _merge_filing_citations

        s = FinalAnswer(text="Sustainability points.", citations=[
            Citation(label="ITC Ltd p. 7", source_kind="filing"),
            Citation(label="ITC Ltd p. 9", source_kind="filing"),
        ])
        out = _merge_filing_citations(s, self._trace())
        assert {c.page for c in out.citations} == {7, 9}
        assert all(c.url == "https://www.bseindia.com/x.pdf" for c in out.citations)

    def test_adds_inline_pagecite_alongside_pageless_cite(self):
        from src.schemas.chat import Citation, FinalAnswer
        from src.services.agent_runner import _merge_filing_citations

        # a page-less LLM cite is left alone; the inline page-cite is added w/ url+page.
        s = FinalAnswer(
            text="Sustainability agenda [ITC Ltd | p.5].",
            citations=[Citation(label="ITC Annual Report 2025", source_kind="filing")],
        )
        out = _merge_filing_citations(s, self._trace())
        assert any(c.page == 5 and c.url for c in out.citations)

    def test_multi_company_matches_by_substring(self):
        from src.schemas.chat import Citation, FinalAnswer
        from src.services.agent_runner import _merge_filing_citations

        trace = [{"tool": "stock_filings_read", "response": {"selected_filings": [
            {"company_name": "ITC Ltd", "pdf_link": "https://www.bseindia.com/itc.pdf"},
            {"company_name": "HDFC Bank Ltd", "pdf_link": "https://www.bseindia.com/hdfc.pdf"},
        ], "evidence": []}}]
        s = FinalAnswer(text="cmp", citations=[
            Citation(label="ITC Ltd Annual Report 2025, p. 5", source_kind="filing"),
            Citation(label="HDFC Bank Ltd FY25, p. 12", source_kind="filing"),
        ])
        out = _merge_filing_citations(s, trace)
        by_url = {c.url: c.page for c in out.citations}
        assert by_url.get("https://www.bseindia.com/itc.pdf") == 5
        assert by_url.get("https://www.bseindia.com/hdfc.pdf") == 12

    def test_single_pdf_fallback_on_company_mismatch(self):
        from src.schemas.chat import FinalAnswer
        from src.services.agent_runner import _merge_filing_citations

        # inline company doesn't match the filing's name, but only ONE PDF was
        # read → attribute the cite to it.
        s = FinalAnswer(text="A point [Some Other Co | p.9].")
        out = _merge_filing_citations(s, self._trace())
        fil = [c for c in out.citations if c.source_kind == "filing"]
        assert len(fil) == 1 and fil[0].page == 9

    def test_noop_without_filings(self):
        from src.schemas.chat import FinalAnswer
        from src.services.agent_runner import _merge_filing_citations

        s = FinalAnswer(text="x")
        assert _merge_filing_citations(s, []) is s            # no tool results → unchanged
        assert _merge_filing_citations(None, self._trace()) is None


class TestComposerIntent:
    """The composer answers the user's underlying question — even on a turn
    whose message is just a clarification pick ("Company — security_id N")."""

    def test_detects_clarification_pick(self):
        from src.services.agent_runner import _looks_like_clarification_pick as f

        assert f("Reliance Industries Ltd. — security_id 2228")
        assert f("security_id: 1192")
        assert not f("What were Reliance's 2025 board meetings?")

    def test_tool_questions_only_question_arg_deduped(self):
        from src.services.agent_runner import _tool_questions as f

        trace = [
            {"tool": "resolve_company", "args": {"query": "Reliance"}},  # excluded
            {"tool": "stock_filings_read",
             "args": {"question": "board meeting outcomes 2025", "security_id": 2228}},
            {"tool": "stock_filings_read", "args": {"question": "board meeting outcomes 2025"}},
        ]
        assert f(trace) == ["board meeting outcomes 2025"]

    def test_pick_turn_surfaces_tool_question_in_prompt(self, monkeypatch):
        captured: dict = {}

        async def fake_acompletion(**kwargs):
            captured.update(kwargs)
            return _make_litellm_response("answer")

        import litellm
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        import src.services.model_router as mr

        def _no_router():
            raise RuntimeError("router off in test")

        monkeypatch.setattr(mr, "get_router", _no_router)

        from src.services import agent_runner as ar
        monkeypatch.setattr(ar.settings, "GEMINI_API_KEY", "k")
        for k in ["GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3",
                  "GEMINI_API_KEY_4", "GEMINI_API_KEY_5", "GEMINI_API_KEY_6",
                  "GEMINI_API_KEY_7", "GEMINI_API_KEY_8"]:
            monkeypatch.setattr(ar.settings, k, "")

        trace = [{
            "tool": "stock_filings_read",
            "args": {"question": "board meeting outcomes 2025", "security_id": 2228},
            "response": {"evidence": [{"quote": "approved dividend"}]},
        }]
        asyncio.run(ar._compose_final_answer(
            "Reliance Industries Ltd. — security_id 2228", trace))

        umsg = next(m["content"] for m in captured["messages"] if m["role"] == "user")
        assert "THE QUESTION TO ANSWER" in umsg
        assert "board meeting outcomes 2025" in umsg
