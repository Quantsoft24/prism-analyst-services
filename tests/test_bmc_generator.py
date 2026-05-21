"""Unit tests for BMC generator pure helpers — no DB/LLM.

The grounded end-to-end generation (9 LLM calls) is validated manually /
in the live run. Here we test the deterministic parsing + citation-mapping
logic that determines whether evidence links are correct.
"""

from __future__ import annotations

import pytest

from src.agents.bmc.blocks import BMC_BLOCKS, BMC_BLOCKS_BY_ID
from src.agents.bmc.reconciler import parse_contradictions
from src.services.bmc.generator import (
    _compute_confidence,
    _extract_markers,
    _parse_block_json,
)


def test_nine_canonical_blocks():
    assert len(BMC_BLOCKS) == 9
    ids = {b.block_id for b in BMC_BLOCKS}
    assert ids == {
        "key_partners", "key_activities", "value_propositions",
        "customer_relationships", "customer_segments", "key_resources",
        "channels", "cost_structure", "revenue_streams",
    }


def test_block_orders_are_unique_and_complete():
    orders = sorted(b.order for b in BMC_BLOCKS)
    assert orders == list(range(9))  # 0..8, no gaps/dupes


def test_every_block_has_query_and_focus():
    for b in BMC_BLOCKS:
        assert b.retrieval_query.strip()
        assert b.focus.strip()


def test_blocks_by_id_lookup():
    assert BMC_BLOCKS_BY_ID["revenue_streams"].title == "Revenue Streams"


# ── JSON parsing ──────────────────────────────────────────────────────────


def test_parse_clean_json():
    raw = '{"bullets": ["a [1]"], "evidence_missing": false, "confidence": 0.8}'
    parsed = _parse_block_json(raw)
    assert parsed["bullets"] == ["a [1]"]
    assert parsed["confidence"] == 0.8


def test_parse_json_with_markdown_fence():
    raw = '```json\n{"bullets": ["x [2]"], "confidence": 0.5}\n```'
    parsed = _parse_block_json(raw)
    assert parsed["bullets"] == ["x [2]"]


def test_parse_invalid_json_raises():
    with pytest.raises(Exception):
        _parse_block_json("not json at all")


# ── Citation marker extraction ──────────────────────────────────────────────


def test_extract_markers_single():
    assert _extract_markers(["Serves BFSI clients [1]."]) == {1}


def test_extract_markers_multiple_across_bullets():
    bullets = [
        "Revenue grew on BFSI demand [1][3].",
        "Margins held at 25% [2].",
        "No citation here.",
    ]
    assert _extract_markers(bullets) == {1, 2, 3}


def test_extract_markers_grouped_form():
    """The LLM frequently emits grouped citations like [1, 2, 3] — these MUST
    be extracted, else evidence links silently disappear (regression guard for
    the bug found in TCS v1)."""
    bullets = [
        "Large internal workforce delivers core services [1, 2, 3, 5, 6].",
        "Credit risk managed via provisions [3,4].",
    ]
    assert _extract_markers(bullets) == {1, 2, 3, 4, 5, 6}


def test_extract_markers_none():
    assert _extract_markers(["uncited claim", "another"]) == set()


def test_extract_markers_dedupes():
    assert _extract_markers(["a [1]", "b [1]", "c [1]"]) == {1}


# ── Deterministic confidence floor (P3-2b) ──────────────────────────────────


def test_confidence_anchored_by_evidence_not_pure_llm():
    """A block the LLM is 100% sure of but with only 1 citation must NOT score
    near 1.0 — the deterministic evidence component anchors it down."""
    high_llm_thin_evidence = _compute_confidence(llm_confidence=1.0, cited_count=1)
    # deterministic = 1/3 ≈ .333 → 0.6*.333 + 0.4*1.0 = 0.6
    assert high_llm_thin_evidence == pytest.approx(0.6, abs=0.01)


def test_confidence_rises_with_evidence():
    c1 = _compute_confidence(llm_confidence=0.8, cited_count=1)
    c3 = _compute_confidence(llm_confidence=0.8, cited_count=3)
    assert c3 > c1
    # 3+ citations saturates the deterministic component at 1.0.
    assert _compute_confidence(llm_confidence=0.8, cited_count=5) == _compute_confidence(
        llm_confidence=0.8, cited_count=3
    )


def test_confidence_clamps_llm_input():
    # Out-of-range LLM self-rating is clamped, never escapes [0,1].
    assert 0.0 <= _compute_confidence(llm_confidence=5.0, cited_count=0) <= 1.0
    assert 0.0 <= _compute_confidence(llm_confidence=-1.0, cited_count=0) <= 1.0


# ── Reconciler parsing ──────────────────────────────────────────────────────


def test_parse_contradictions_valid():
    raw = '{"contradictions": [{"block_a": "revenue_streams", "block_b": "customer_segments", "issue": "x"}]}'
    out = parse_contradictions(raw)
    assert len(out) == 1
    assert out[0]["block_a"] == "revenue_streams"


def test_parse_contradictions_empty_and_malformed():
    assert parse_contradictions('{"contradictions": []}') == []
    assert parse_contradictions("not json") == []
    # Missing required keys → dropped.
    assert parse_contradictions('{"contradictions": [{"block_a": "x"}]}') == []


def test_parse_contradictions_strips_fences():
    raw = '```json\n{"contradictions": [{"block_a":"a","block_b":"b","issue":"i"}]}\n```'
    assert len(parse_contradictions(raw)) == 1
