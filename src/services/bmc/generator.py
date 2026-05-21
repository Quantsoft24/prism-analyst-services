"""BMCGenerator — produces a filing-grounded Business Model Canvas.

Phase 3 (agentic) algorithm:

    for each of the 9 blocks (bounded concurrency):
        1. retrieve top-K filing chunks for this block's query (WE control
           retrieval + numbering so citations map exactly to chunks)
        2. run the block's ADK SUB-AGENT on the numbered excerpts. The agent
           writes 2-4 cited bullets and calls the NRE ``compute_*`` tools for
           any derived figure (growth %, margin, ...) — math is deterministic.
        3. parse JSON → bullets + cited markers → evidence rows
        4. confidence = deterministic evidence-anchored floor blended with the
           agent's self-rating (NOT pure LLM self-rating)
    run the CrossBlockReconciler over all blocks → store contradictions
    persist analysis + blocks + evidence as a new version

Grounding contract (unchanged from Lite, the thing that beats every competitor):
  * The agent sees ONLY retrieved filing chunks, numbered [1], [2], ...
  * Cited markers map back to real ``filing_chunks`` → ``bmc_evidence`` w/ page.
  * No evidence → ``evidence_missing``. We never fabricate.

Concurrency is bounded (default 3) so 9 sub-agents don't blow free-tier RPM;
the router's multi-key fallback absorbs bursts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.bmc import (
    BMC_BLOCKS,
    BMCBlockDef,
    build_bmc_block_agent,
    build_bmc_reconciler_agent,
    parse_contradictions,
)
from src.config import settings
from src.models.bmc import BMCAnalysis, BMCBlock, BMCEvidence
from src.models.company import Company
from src.repositories.bmc_repo import BMCRepository
from src.repositories.company_repo import CompanyRepository
from src.services.bmc.agent_exec import run_agent_to_text
from src.services.retrieval import HybridRetrievalService, RetrievedChunk

logger = logging.getLogger(__name__)

_MODEL_LABEL = "prism-fast+quality"  # blocks → fast tier, reconciler → quality tier


@dataclass(slots=True)
class BMCGenerationResult:
    bmc_id: uuid.UUID | None
    ticker: str
    status: str            # 'complete' | 'partial' | 'failed'
    version: int | None
    blocks_ok: int
    blocks_total: int
    contradictions: int = 0
    detail: str = ""


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

        # 1. Generate all blocks via sub-agents, bounded concurrency.
        sem = asyncio.Semaphore(max(1, settings.BMC_BLOCK_CONCURRENCY))

        async def _one(block_def: BMCBlockDef) -> _BlockResult:
            async with sem:
                return await self._generate_block(company, block_def)

        block_results = await asyncio.gather(*[_one(b) for b in BMC_BLOCKS])

        # 2. CrossBlockReconciler — review the assembled canvas for contradictions.
        contradictions = await self._reconcile(company, block_results)

        # 3. Persist as a new version.
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
            model=_MODEL_LABEL,
            contradictions=contradictions,
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
            contradictions=len(contradictions),
            detail=f"v{version} · overall confidence {overall} · {len(contradictions)} contradiction(s)",
        )

    # ── Per-block generation (agentic) ────────────────────────────────────

    async def _generate_block(self, company: Company, block_def: BMCBlockDef) -> _BlockResult:
        # Retrieve evidence for this block, scoped to the company. WE control
        # the chunk set + numbering so citations stay exact.
        try:
            chunks = await self._retriever.retrieve(
                block_def.retrieval_query,
                company_id=company.id,
                limit=settings.BMC_CHUNKS_PER_BLOCK,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("BMC retrieval failed for %s/%s: %s", company.ticker, block_def.block_id, exc)
            return _BlockResult(block_def, [], None, 0.0, "failed", [])

        if not chunks:
            return _BlockResult(block_def, [], None, 0.0, "evidence_missing", [])

        cap = settings.BMC_CHUNK_CHAR_CAP
        numbered = "\n\n".join(
            f"[{i}] (p.{c.page_number}) {c.text[:cap]}" for i, c in enumerate(chunks, start=1)
        )
        user_message = (
            f"Company: {company.name} ({company.exchange}:{company.ticker})\n\n"
            f"Filing excerpts:\n{numbered}"
        )

        # Run the per-block sub-agent (it may call NRE compute_* tools).
        try:
            agent = build_bmc_block_agent(block_def)
            raw = await run_agent_to_text(agent, user_message)
            parsed = _parse_block_json(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BMC sub-agent failed for %s/%s: %s", company.ticker, block_def.block_id, exc)
            return _BlockResult(block_def, [], None, 0.0, "failed", [])

        if parsed.get("evidence_missing") or not parsed.get("bullets"):
            return _BlockResult(block_def, [], parsed.get("key_insights"), 0.0, "evidence_missing", [])

        bullets: list[str] = parsed["bullets"]

        # Map cited [n] markers back to real chunks → evidence rows.
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

        confidence = _compute_confidence(
            llm_confidence=float(parsed.get("confidence", 0.5)),
            cited_count=len(evidence),
        )
        return _BlockResult(block_def, bullets, parsed.get("key_insights"), confidence, "ok", evidence)

    # ── Reconciler ─────────────────────────────────────────────────────────

    async def _reconcile(self, company: Company, results: list[_BlockResult]) -> list[dict]:
        """Run the CrossBlockReconciler over the OK blocks. Best-effort — a
        reconciler failure must never fail the whole canvas. Skipped entirely
        when ``BMC_RECONCILER_ENABLED`` is off (saves 1 quality-tier call/BMC)."""
        if not settings.BMC_RECONCILER_ENABLED:
            return []
        ok_blocks = [r for r in results if r.status == "ok" and r.bullets]
        if len(ok_blocks) < 2:
            return []
        rendered = "\n\n".join(
            f"{r.block_def.title} ({r.block_def.block_id}):\n"
            + "\n".join(f"- {b}" for b in r.bullets)
            for r in ok_blocks
        )
        message = f"Company: {company.name}\n\nBlocks:\n{rendered}"
        try:
            raw = await run_agent_to_text(build_bmc_reconciler_agent(), message)
            return parse_contradictions(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BMC reconciler failed for %s: %s", company.ticker, exc)
            return []


@dataclass(slots=True)
class _BlockResult:
    block_def: BMCBlockDef
    bullets: list[str]
    key_insights: list | None
    confidence: float
    status: str
    evidence: list[dict]


# ── Helpers ──────────────────────────────────────────────────────────────


def _compute_confidence(*, llm_confidence: float, cited_count: int) -> float:
    """Blend the agent's self-rating with a DETERMINISTIC, evidence-anchored
    score so confidence isn't pure LLM self-assessment (the plan's "deterministic
    floor"). More cited sources → higher deterministic component; we weight it
    above the self-rating so a confident-but-thinly-sourced block can't score high.

    deterministic = min(1, cited_count / 3)   (0, .33, .67, 1.0 for 0..3+ cites)
    final = 0.6 * deterministic + 0.4 * llm_confidence
    """
    llm = max(0.0, min(1.0, llm_confidence))
    deterministic = min(1.0, cited_count / 3.0)
    return round(0.6 * deterministic + 0.4 * llm, 3)


def _parse_block_json(raw: str) -> dict:
    """Parse the model's JSON, tolerating accidental markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    return json.loads(text)


def _extract_markers(bullets: list[str]) -> set[int]:
    """Pull every integer out of citation markers across all bullets.

    Handles BOTH single markers (``[2]``) and grouped markers the LLM often
    emits (``[1, 2, 3]`` / ``[3,4]``). Missing the grouped form silently
    drops evidence links — which breaks the "show your work" contract — so we
    parse any ``[...]`` containing digits + commas.
    """
    markers: set[int] = set()
    for b in bullets:
        for group in re.findall(r"\[([\d,\s]+)\]", b):
            for num in re.findall(r"\d+", group):
                markers.add(int(num))
    return markers
