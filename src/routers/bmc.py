"""Business Model Canvas endpoints.

  POST /api/v1/bmc/{ticker}/run        — generate a NEW grounded canvas version
  GET  /api/v1/bmc/{ticker}            — latest canvas (9 blocks + citations)
  GET  /api/v1/bmc/{ticker}/library    — all versions (headers)
  GET  /api/v1/bmc/{ticker}/{version}  — a specific version

Generation is synchronous here (9 grounded LLM calls, ~15-40s). For Phase 2
Lite that's acceptable for an explicit user action. Phase 3 can move it to a
streamed/async job if needed. The agent-facing path stays the FAST read tool
(``get_company_bmc``) — see ``src/tools/bmc_tools.py``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.auth import get_current_firm_id
from src.core.database import get_session
from src.models.bmc import BMCAnalysis
from src.repositories.bmc_repo import BMCRepository
from src.agents.bmc.block_chat import build_bmc_block_chat_agent
from src.agents.bmc.blocks import BMC_BLOCKS_BY_ID
from src.schemas.bmc import (
    BMCBlockRead,
    BMCChatRequest,
    BMCChatResponse,
    BMCContradiction,
    BMCEvidenceRead,
    BMCRead,
    BMCRunRequest,
    BMCVersionSummary,
)
from src.services.bmc import BMCGenerator
from src.services.bmc.agent_exec import run_agent_to_text
from src.services.bmc.exporter import EXPORT_FORMATS, export_bmc, filename_for

router = APIRouter(prefix="/bmc", tags=["Business Model Canvas"])


def _to_read(analysis: BMCAnalysis) -> BMCRead:
    """Serialize an analysis (with blocks + evidence) into the API shape,
    blocks ordered for the 3x3 grid."""
    ordered = sorted(analysis.blocks, key=lambda b: b.order)
    return BMCRead(
        id=analysis.id,
        ticker=analysis.ticker,
        company_id=analysis.company_id,
        version=analysis.version,
        fiscal_period=analysis.fiscal_period,
        status=analysis.status,
        overall_confidence=analysis.overall_confidence,
        model=analysis.model,
        created_at=analysis.created_at,
        blocks=[
            BMCBlockRead(
                block_id=b.block_id,
                title=b.title,
                order=b.order,
                summary_bullets=b.summary_bullets,
                key_insights=b.key_insights,
                confidence=b.confidence,
                status=b.status,
                evidence=[BMCEvidenceRead.model_validate(e) for e in b.evidence],
            )
            for b in ordered
        ],
        contradictions=[
            BMCContradiction(**c) for c in (analysis.contradictions or [])
        ],
    )


@router.post(
    "/{ticker}/run",
    response_model=BMCRead,
    summary="Generate a new filing-grounded Business Model Canvas",
    description=(
        "Runs grounded generation across all 9 Osterwalder blocks for the "
        "company, citing the filing chunks each claim draws from. Creates a "
        "new version (never overwrites — enables temporal diffing later). "
        "Synchronous; takes ~15-40s. Requires the company to have ingested "
        "filings, else most blocks return 'evidence_missing'."
    ),
    responses={404: {"description": "Company not in coverage universe."}},
)
async def run_bmc(
    ticker: str,
    body: BMCRunRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> BMCRead:
    generator = BMCGenerator(session)
    result = await generator.generate(firm_id, ticker, fiscal_period=body.fiscal_period)

    if result.status == "failed" and result.bmc_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result.detail)

    # Reload with blocks + evidence for the response.
    analysis = await BMCRepository(session).get_by_id(result.bmc_id)  # type: ignore[arg-type]
    assert analysis is not None
    return _to_read(analysis)


@router.get(
    "/{ticker}",
    response_model=BMCRead,
    summary="Latest Business Model Canvas for a company",
    responses={404: {"description": "No canvas generated yet."}},
)
async def get_latest_bmc(
    ticker: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> BMCRead:
    analysis = await BMCRepository(session).get_latest(firm_id, ticker.upper())
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No BMC generated for {ticker.upper()} yet. POST /bmc/{ticker}/run to create one.",
        )
    return _to_read(analysis)


@router.post(
    "/{ticker}/blocks/{block_id}/chat",
    response_model=BMCChatResponse,
    summary="Drill-down chat about one BMC block",
    description=(
        "Ask a follow-up question about a specific block. The answer is grounded "
        "ONLY in that block's evidence excerpts (with NRE math), so it stays "
        "consistent with the canvas. Stateless — pass the prior thread in `history`."
    ),
    responses={404: {"description": "No canvas or block not found."}},
)
async def chat_about_block(
    ticker: str,
    block_id: str,
    body: BMCChatRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> BMCChatResponse:
    analysis = await BMCRepository(session).get_latest(firm_id, ticker.upper())
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No BMC generated for {ticker.upper()} yet.",
        )
    block = next((b for b in analysis.blocks if b.block_id == block_id), None)
    if block is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Block {block_id!r} not found in {ticker.upper()}'s canvas.",
        )

    block_def = BMC_BLOCKS_BY_ID.get(block_id)
    focus = block_def.focus if block_def else block.title

    # Numbered evidence excerpts backing this block.
    numbered_evidence = "\n\n".join(
        f"[{i}] (p.{ev.page_number}) {ev.excerpt}"
        for i, ev in enumerate(block.evidence, start=1)
    ) or "(no evidence excerpts on file for this block)"

    # Render prior thread + the block's current bullets for context.
    bullets = "\n".join(f"- {b}" for b in block.summary_bullets)
    history = "\n".join(f"{m.role}: {m.content}" for m in body.history)
    message = (
        f"Block: {block.title}\n"
        f"Current bullets:\n{bullets}\n\n"
        f"Evidence excerpts:\n{numbered_evidence}\n\n"
        + (f"Conversation so far:\n{history}\n\n" if history else "")
        + f"Analyst question: {body.message}"
    )

    agent = build_bmc_block_chat_agent(block.title, focus)
    answer = await run_agent_to_text(agent, message)
    return BMCChatResponse(answer=answer or "I couldn't find an answer in this block's evidence.")


@router.get(
    "/{ticker}/export",
    summary="Export the latest canvas as JSON / XLSX / PDF",
    description="Download a self-documenting export (includes the Sources audit trail).",
    responses={404: {"description": "No canvas generated yet."}, 400: {"description": "Bad format."}},
)
async def export_bmc_endpoint(
    ticker: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
    format: str = "pdf",
) -> Response:
    fmt = format.lower()
    if fmt not in EXPORT_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format {format!r}. Valid: {list(EXPORT_FORMATS)}",
        )
    analysis = await BMCRepository(session).get_latest(firm_id, ticker.upper())
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No BMC generated for {ticker.upper()} yet.",
        )
    bmc = _to_read(analysis)
    data = export_bmc(bmc, fmt)
    media_type = EXPORT_FORMATS[fmt][0]
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename_for(bmc, fmt)}"'},
    )


@router.get(
    "/{ticker}/library",
    response_model=list[BMCVersionSummary],
    summary="All canvas versions for a company (headers only)",
)
async def list_bmc_versions(
    ticker: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> list[BMCVersionSummary]:
    versions = await BMCRepository(session).list_versions(firm_id, ticker.upper())
    return [BMCVersionSummary.model_validate(v) for v in versions]


@router.get(
    "/{ticker}/{version}",
    response_model=BMCRead,
    summary="A specific canvas version",
    responses={404: {"description": "Version not found."}},
)
async def get_bmc_version(
    ticker: str,
    version: int,
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> BMCRead:
    analysis = await BMCRepository(session).get_version(firm_id, ticker.upper(), version)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"BMC version {version} for {ticker.upper()} not found.",
        )
    return _to_read(analysis)
