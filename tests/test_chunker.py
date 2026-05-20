"""Unit tests for the section-aware chunker — pure, no DB/network/LLM."""

from __future__ import annotations

import pytest

from src.services.ingestion.chunker import Chunker, _detect_section
from src.services.ingestion.parser.base import ParsedDocument, ParsedPage


@pytest.fixture
def chunker():
    # Small budget so tests exercise splitting without giant fixtures.
    return Chunker(target_tokens=40, overlap_tokens=8, tokenizer_name="cl100k_base")


def _doc(*page_texts: str) -> ParsedDocument:
    return ParsedDocument(
        pages=[ParsedPage(page_number=i + 1, text=t) for i, t in enumerate(page_texts)],
        parser_backend="test",
    )


# ── Section detection ────────────────────────────────────────────────────


def test_detect_section_recognizes_mda():
    assert _detect_section("MANAGEMENT DISCUSSION AND ANALYSIS") == "mda"
    assert _detect_section("## Management's Discussion") == "mda"


def test_detect_section_recognizes_financial_statements():
    assert _detect_section("Balance Sheet") == "balance_sheet"
    assert _detect_section("Statement of Profit and Loss") == "profit_loss"
    assert _detect_section("Cash Flow Statement") == "cash_flow"
    assert _detect_section("Notes to the Financial Statements") == "notes"


def test_detect_section_ignores_body_text():
    # A long sentence is not a heading even if it contains keywords.
    body = (
        "During the year the management discussion covered several topics "
        "including the balance sheet strength and cash flow generation across "
        "all business segments in considerable detail."
    )
    assert _detect_section(body) is None


def test_detect_section_ignores_blank():
    assert _detect_section("") is None
    assert _detect_section("   ") is None


# ── Chunking behavior ──────────────────────────────────────────────────────


def test_empty_document_yields_no_chunks(chunker):
    assert chunker.chunk_document(_doc()) == []
    assert chunker.chunk_document(_doc("")) == []


def test_short_document_single_chunk(chunker):
    chunks = chunker.chunk_document(_doc("Revenue grew strongly this quarter."))
    assert len(chunks) == 1
    assert chunks[0].section == "general"
    assert chunks[0].page_number == 1
    assert chunks[0].token_count > 0


def test_section_switch_flushes_and_tags(chunker):
    text = (
        "MANAGEMENT DISCUSSION AND ANALYSIS\n\n"
        "Revenue rose on strong demand.\n\n"
        "Balance Sheet\n\n"
        "Total assets increased to record levels."
    )
    chunks = chunker.chunk_document(_doc(text))
    sections = {c.section for c in chunks}
    assert "mda" in sections
    assert "balance_sheet" in sections
    # The MD&A content must be tagged mda, the balance-sheet content balance_sheet.
    mda_chunk = next(c for c in chunks if "Revenue rose" in c.text)
    assert mda_chunk.section == "mda"
    bs_chunk = next(c for c in chunks if "Total assets" in c.text)
    assert bs_chunk.section == "balance_sheet"


def test_long_content_splits_into_multiple_chunks(chunker):
    # Many paragraphs well over the 40-token budget → multiple chunks.
    paras = [f"Paragraph number {i} with several words to consume tokens here." for i in range(20)]
    chunks = chunker.chunk_document(_doc("\n\n".join(paras)))
    assert len(chunks) > 1
    # chunk_index is monotonic and contiguous.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_oversized_block_kept_whole(chunker):
    # A single block bigger than the budget (e.g. a table) is not split.
    big_table = "[TABLE]\n" + "\n".join(f"Row {i} | value {i} | metric {i}" for i in range(30))
    chunks = chunker.chunk_document(_doc(big_table))
    table_chunks = [c for c in chunks if "[TABLE]" in c.text]
    assert len(table_chunks) == 1
    assert table_chunks[0].token_count > chunker._target


def test_overlap_carries_context_forward():
    # With overlap, consecutive chunks should share some trailing text.
    chunker = Chunker(target_tokens=30, overlap_tokens=15, tokenizer_name="cl100k_base")
    paras = [f"Distinct paragraph {i} carrying unique words alpha bravo charlie." for i in range(8)]
    chunks = chunker.chunk_document(_doc("\n\n".join(paras)))
    assert len(chunks) >= 2
    # At least one later chunk should contain text that also ended a prior one.
    # (Soft check — overlap is best-effort by token budget.)
    joined = [c.text for c in chunks]
    assert any(
        any(prev_tail in later for prev_tail in joined[i].split("\n\n"))
        for i, later in enumerate(joined[1:], start=0)
    )


def test_page_numbers_preserved_across_pages(chunker):
    chunks = chunker.chunk_document(_doc("Page one content here.", "Page two content here."))
    pages = {c.page_number for c in chunks}
    assert pages == {1, 2}


def test_markdown_preferred_over_text():
    """When a page has markdown (docling), the chunker uses it."""
    doc = ParsedDocument(
        pages=[ParsedPage(page_number=1, text="plain text", markdown="# Heading\n\nMarkdown body content.")],
        parser_backend="docling",
    )
    chunks = Chunker(target_tokens=100).chunk_document(doc)
    assert any("Markdown body" in c.text for c in chunks)
    assert all("plain text" not in c.text for c in chunks)
