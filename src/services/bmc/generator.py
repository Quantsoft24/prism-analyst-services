"""BMCGenerator — produces a filing-grounded Business Model Canvas.

Phase 2 Lite algorithm (per the approved plan):

    for each of the 9 blocks (bounded concurrency):
        1. retrieve top-K filing chunks for this block's query, scoped to the company
        2. one LLM call: summarize the block in 2-4 cited bullets using ONLY those chunks
        3. parse JSON → bullets + which chunk markers were cited
    persist analysis + blocks + evidence as a new version

Grounding contract (what makes this different from every other BMC tool):
  * The LLM sees ONLY retrieved filing chunks, numbered [1], [2], ...
  * It must cite the chunk markers it used; we map those back to real
    ``filing_chunks`` rows → ``bmc_evidence`` with page numbers.
  * If a block has no supporting chunks, it's marked ``evidence_missing`` —
    we never fabricate. (The old PRISM_ANALYST hallucinated here.)

Concurrency is bounded (default 3) so 9 blocks don't blow free-tier RPM.
The router's multi-key fallback absorbs the rest.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.bmc.blocks import BMC_BLOCKS, BMCBlockDef
from src.models.bmc import BMCAnalysis, BMCBlock, BMCEvidence
from src.models.company import Company
from src.repositories.bmc_repo import BMCRepository
from src.repositories.company_repo import CompanyRepository
from src.services.model_router import get_router
from src.services.retrieval import HybridRetrievalService, RetrievedChunk

logger = logging.getLogger(__name__)

_BLOCK_CONCURRENCY = 3
_CHUNKS_PER_BLOCK = 6
_LLM_TIER = "quality"  # block summaries want the better tier; volume is low (9/canvas)


@dataclass(slots=True)
class BMCGenerationResult:
    bmc_id: uuid.UUID | None
    ticker: str
    status: str            # 'complete' | 'partial' | 'failed'
    version: int | None
    blocks_ok: int
    blocks_total: int
    detail: str = ""


_SYSTEM_PROMPT = """\
You are PRISM's Business Model Canvas analyst for Indian listed companies.
You will be given a company name and a set of NUMBERED excerpts from its
filings. Summarize ONE Business Model Canvas block.

HARD RULES:
- Use ONLY the provided excerpts. Do NOT use outside knowledge.
- Every bullet MUST cite the excerpt(s) it draws from using [n] markers.
- If the excerpts contain no relevant information for this block, return an
  empty bullets list and set "evidence_missing": true. NEVER invent facts.
- 2 to 4 short bullets. Each bullet one sentence. No preamble.
- Indian context: ₹ amounts, FY ending 31 March.

Respond with STRICT JSON only, no markdown fences:
{
  "bullets": ["Serves BFSI clients across North America [1].", ...],
  "key_insights": ["optional 1-2 word tags"],
  "evidence_missing": false,
  "confidence": 0.0-1.0
}
confidence reflects how well the excerpts support the block (more relevant
excerpts + specific figures = higher).
"""


class BMCGenerator:
    """Generates + persists a grounded BMC for one company."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._companies = CompanyRepository(session)
        self._bmc = BMCRepository(session)
        self._retriever = HybridRetrievalService(session)

    async def generate(
        self, firm_id: str, ticker: str, *, fiscal_period: str | None = None
    ) -> BMCGenerationResult:
        ticker = ticker.strip().upper()
        company = await self._companies.get_by_ticker(ticker)
        if company is None:
            return BMCGenerationResult(None, ticker, "failed", None, 0, len(BMC_BLOCKS),
                                       detail=f"{ticker} not in coverage universe.")

        # Generate all blocks with bounded concurrency.
        sem = asyncio.Semaphore(_BLOCK_CONCURRENCY)

        async def _one(block_def: BMCBlockDef) -> _BlockResult:
            async with sem:
                return await self._generate_block(company, block_def)

        block_results = await asyncio.gather(*[_one(b) for b in BMC_BLOCKS])

        # Persist as a new version.
        version = await self._bmc.next_version(firm_id, ticker)
        confidences = [r.confidence for r in block_results if r.status == "ok"]
        overall = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
        blocks_ok = sum(1 for r in block_results if r.status == "ok")
        status = (
            "complete" if blocks_ok == len(BMC_BLOCKS)
            else "partial" if blocks_ok > 0
            else "failed"
        )

        analysis = BMCAnalysis(
            firm_id=firm_id,
            ticker=ticker,
            company_id=company.id,
            version=version,
            fiscal_period=fiscal_period,
            status=status,
            overall_confidence=overall,
            model=f"prism-{_LLM_TIER}",
        )
        await self._bmc.add(analysis)

        for r in block_results:
            block = BMCBlock(
                bmc_id=analysis.id,
                block_id=r.block_def.block_id,
                title=r.block_def.title,
                order=r.block_def.order,
                summary_bullets=r.bullets,
                key_insights=r.key_insights,
                confidence=r.confidence,
                status=r.status,
            )
            self._session.add(block)
            await self._session.flush()
            for ev in r.evidence:
                self._session.add(BMCEvidence(bmc_block_id=block.id, **ev))
        await self._session.flush()

        return BMCGenerationResult(
            bmc_id=analysis.id, ticker=ticker, status=status, version=version,
            blocks_ok=blocks_ok, blocks_total=len(BMC_BLOCKS),
            detail=f"v{version} · overall confidence {overall}",
        )

    # ── Per-block generation ──────────────────────────────────────────────

    async def _generate_block(self, company: Company, block_def: BMCBlockDef) -> _BlockResult:
        # 1. Retrieve evidence for this block, scoped to the company.
        try:
            chunks = await self._retriever.retrieve(
                block_def.retrieval_query, company_id=company.id, limit=_CHUNKS_PER_BLOCK
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("BMC retrieval failed for %s/%s: %s", company.ticker, block_def.block_id, exc)
            return _BlockResult(block_def, [], None, 0.0, "failed", [])

        if not chunks:
            return _BlockResult(block_def, [], None, 0.0, "evidence_missing", [])

        # 2. LLM summarize using only the numbered chunks.
        numbered = "\n\n".join(
            f"[{i}] (p.{c.page_number}) {c.text[:1200]}" for i, c in enumerate(chunks, start=1)
        )
        user_prompt = (
            f"Company: {company.name} ({company.exchange}:{company.ticker})\n"
            f"Block: {block_def.title}\n"
            f"What this block captures: {block_def.focus}\n\n"
            f"Filing excerpts:\n{numbered}"
        )
        try:
            raw = await get_router().acomplete(
                _LLM_TIER,
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_json=True,
            )
            parsed = _parse_block_json(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BMC LLM failed for %s/%s: %s", company.ticker, block_def.block_id, exc)
            return _BlockResult(block_def, [], None, 0.0, "failed", [])

        if parsed.get("evidence_missing") or not parsed.get("bullets"):
            return _BlockResult(block_def, [], parsed.get("key_insights"), 0.0, "evidence_missing", [])

        bullets: list[str] = parsed["bullets"]
        confidence = float(parsed.get("confidence", 0.5))

        # 3. Map cited [n] markers back to real chunks → evidence rows.
        cited_markers = _extract_markers(bullets)
        evidence: list[dict] = []
        for marker_num in sorted(cited_markers):
            if 1 <= marker_num <= len(chunks):
                c: RetrievedChunk = chunks[marker_num - 1]
                evidence.append(
                    {
                        "marker": f"[{marker_num}]",
                        "chunk_id": c.chunk_id,
                        "filing_id": c.filing_id,
                        "page_number": c.page_number,
                        "excerpt": c.text[:500],
                    }
                )

        return _BlockResult(
            block_def, bullets, parsed.get("key_insights"), confidence, "ok", evidence
        )


@dataclass(slots=True)
class _BlockResult:
    block_def: BMCBlockDef
    bullets: list[str]
    key_insights: list | None
    confidence: float
    status: str
    evidence: list[dict]


def _parse_block_json(raw: str) -> dict:
    """Parse the model's JSON, tolerating accidental markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    return json.loads(text)


def _extract_markers(bullets: list[str]) -> set[int]:
    """Pull the integers out of [n] citation markers across all bullets."""
    markers: set[int] = set()
    for b in bullets:
        for m in re.findall(r"\[(\d+)\]", b):
            markers.add(int(m))
    return markers
