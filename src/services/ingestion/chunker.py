"""Section-aware, token-budgeted chunker.

Turns a ``ParsedDocument`` into ``Chunk`` records ready for embedding + storage.

Design choices (retrieval quality is won here):
  * **Section detection first.** Financial filings have stable section
    headers (MD&A, Balance Sheet, Notes, ...). We classify each block to a
    section so retrieval can boost by intent ("margins" → MD&A). Detection
    uses heading keywords; works on both docling markdown and plain text.
  * **Respect paragraph/sentence boundaries.** We never split mid-sentence.
    Chunks accumulate whole paragraphs until the token budget is hit.
  * **Token-budgeted with overlap.** ~512 tokens/chunk with ~64 overlap so a
    fact near a boundary is retrievable from either side.
  * **Never split tables.** A ``[TABLE]`` block (from pdfplumber) or a docling
    markdown table is kept whole even if it exceeds the budget — splitting a
    table destroys its meaning.

This is deterministic + pure (no LLM, no I/O), so it's fully unit-testable.
A future "contextual retrieval" upgrade (prepend an LLM-written context blurb
to each chunk) slots in as a post-processing step without changing this core.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.config import settings
from src.models.filing import FILING_SECTIONS
from src.services.ingestion.parser.base import ParsedDocument


@dataclass(slots=True)
class Chunk:
    """A single chunk ready to become a ``FilingChunk`` row."""

    chunk_index: int
    text: str
    section: str
    page_number: int | None
    token_count: int


# ── Section detection ───────────────────────────────────────────────────────
# Ordered keyword → section mapping. First match wins. Patterns are matched
# case-insensitively against heading-like lines. Tuned for Indian filings.
_SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("mda", re.compile(r"management('s)?\s+discussion|md&a|management discussion and analysis", re.I)),
    ("balance_sheet", re.compile(r"balance\s+sheet|statement of (assets|financial position)", re.I)),
    ("profit_loss", re.compile(r"profit\s+(and|&)\s+loss|statement of profit|income statement", re.I)),
    ("cash_flow", re.compile(r"cash\s+flow", re.I)),
    ("notes", re.compile(r"notes?\s+to\s+(the\s+)?(accounts|financial statements)", re.I)),
    ("auditors_report", re.compile(r"auditor('s|s')?\s+report|independent auditor", re.I)),
    ("directors_report", re.compile(r"director('s|s')?\s+report|board('s)?\s+report", re.I)),
    ("risk_factors", re.compile(r"risk\s+factors?|principal risks", re.I)),
    ("related_party", re.compile(r"related\s+part(y|ies)\s+transaction", re.I)),
    ("segment_reporting", re.compile(r"segment\s+(reporting|information|results)", re.I)),
]


def _detect_section(line: str) -> str | None:
    """Return a section id if ``line`` looks like a section heading, else None."""
    stripped = line.strip()
    # Headings are short-ish and often markdown headers or all-caps.
    if not stripped or len(stripped) > 120:
        return None
    is_heading_like = (
        stripped.startswith("#")
        or stripped.isupper()
        or len(stripped.split()) <= 10
    )
    if not is_heading_like:
        return None
    for section, pattern in _SECTION_PATTERNS:
        if pattern.search(stripped):
            return section
    return None


class Chunker:
    """Token-budgeted, section-aware chunker. Stateless; construct once."""

    def __init__(
        self,
        target_tokens: int | None = None,
        overlap_tokens: int | None = None,
        tokenizer_name: str | None = None,
    ) -> None:
        self._target = target_tokens or settings.CHUNK_TARGET_TOKENS
        self._overlap = overlap_tokens or settings.CHUNK_OVERLAP_TOKENS
        self._tokenizer_name = tokenizer_name or settings.CHUNK_TOKENIZER
        self._encoder = None  # lazy — tiktoken import is mildly heavy

    # ── Tokenization ────────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        if self._encoder is None:
            import tiktoken

            self._encoder = tiktoken.get_encoding(self._tokenizer_name)
        return len(self._encoder.encode(text))

    # ── Public API ──────────────────────────────────────────────────────

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        """Chunk a parsed document into section-tagged, token-budgeted chunks."""
        chunks: list[Chunk] = []
        idx = 0
        current_section = "general"

        for page in doc.pages:
            # Prefer markdown (docling) for structure; fall back to text.
            source = page.markdown or page.text
            if not source:
                continue

            blocks = self._split_into_blocks(source)
            buffer: list[str] = []
            buffer_tokens = 0

            def flush() -> None:
                nonlocal idx, buffer, buffer_tokens
                if not buffer:
                    return
                text = "\n\n".join(buffer).strip()
                if text:
                    chunks.append(
                        Chunk(
                            chunk_index=idx,
                            text=text,
                            section=current_section,
                            page_number=page.page_number,
                            token_count=buffer_tokens,
                        )
                    )
                    idx += 1
                buffer = []
                buffer_tokens = 0

            for block in blocks:
                # Section heading? switch section + flush the current buffer so
                # chunks never straddle a section boundary.
                detected = _detect_section(block)
                if detected and detected in FILING_SECTIONS:
                    flush()
                    current_section = detected
                    continue

                block_tokens = self._count_tokens(block)

                # Tables / oversized blocks: keep whole as their own chunk.
                if block_tokens > self._target:
                    flush()
                    chunks.append(
                        Chunk(
                            chunk_index=idx,
                            text=block.strip(),
                            section=current_section,
                            page_number=page.page_number,
                            token_count=block_tokens,
                        )
                    )
                    idx += 1
                    continue

                # Would exceed budget → flush, then carry overlap forward.
                if buffer_tokens + block_tokens > self._target and buffer:
                    overlap_text = self._tail_for_overlap(buffer)
                    flush()
                    if overlap_text:
                        buffer.append(overlap_text)
                        buffer_tokens += self._count_tokens(overlap_text)

                buffer.append(block)
                buffer_tokens += block_tokens

            flush()

        return chunks

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _split_into_blocks(text: str) -> list[str]:
        """Split text into paragraph/table blocks on blank lines, keeping
        ``[TABLE]`` markers (from pdfplumber) and markdown tables intact."""
        # Normalize newlines, split on blank-line boundaries.
        normalized = re.sub(r"\r\n?", "\n", text)
        raw_blocks = re.split(r"\n\s*\n", normalized)
        return [b.strip() for b in raw_blocks if b.strip()]

    def _tail_for_overlap(self, buffer: list[str]) -> str:
        """Take the last ~overlap_tokens worth of text from the buffer to
        prepend to the next chunk, preserving cross-boundary context."""
        if self._overlap <= 0:
            return ""
        # Walk blocks from the end, accumulating until we hit the overlap budget.
        acc: list[str] = []
        tokens = 0
        for block in reversed(buffer):
            t = self._count_tokens(block)
            if tokens + t > self._overlap and acc:
                break
            acc.insert(0, block)
            tokens += t
        return "\n\n".join(acc)
