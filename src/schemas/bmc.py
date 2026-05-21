"""Pydantic schemas for the Business Model Canvas API."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class BMCRunRequest(BaseModel):
    """Body for ``POST /api/v1/bmc/{ticker}/run``."""

    fiscal_period: str | None = Field(
        default=None,
        description="Optional FY period the canvas should anchor to, e.g. 'Q4-FY26'.",
    )


class BMCEvidenceRead(BaseModel):
    """A citation backing a block — links to the source filing chunk."""

    model_config = ConfigDict(from_attributes=True)

    marker: str
    chunk_id: uuid.UUID | None = None
    filing_id: uuid.UUID | None = None
    page_number: int | None = None
    excerpt: str


class BMCBlockRead(BaseModel):
    """One Osterwalder block."""

    model_config = ConfigDict(from_attributes=True)

    block_id: str
    title: str
    order: int
    summary_bullets: list[str]
    key_insights: list[str] | None = None
    confidence: float
    status: str
    evidence: list[BMCEvidenceRead] = Field(default_factory=list)


class BMCContradiction(BaseModel):
    """A cross-block conflict flagged by the CrossBlockReconciler (Phase 3)."""

    block_a: str
    block_b: str
    issue: str


class BMCRead(BaseModel):
    """A full canvas — header + 9 blocks (ordered for the 3x3 grid)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    ticker: str
    company_id: uuid.UUID
    version: int
    fiscal_period: str | None = None
    status: str
    overall_confidence: float | None = None
    model: str | None = None
    created_at: datetime
    blocks: list[BMCBlockRead] = Field(default_factory=list)
    contradictions: list[BMCContradiction] = Field(default_factory=list)


class BMCVersionSummary(BaseModel):
    """Lightweight header for the version library (no blocks)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    ticker: str
    version: int
    fiscal_period: str | None = None
    status: str
    overall_confidence: float | None = None
    created_at: datetime


class BMCChatMessage(BaseModel):
    """One turn in a per-block drill-down conversation."""

    role: str  # "user" | "assistant"
    content: str


class BMCChatRequest(BaseModel):
    """Body for ``POST /bmc/{ticker}/blocks/{block_id}/chat``.

    Stateless: the frontend holds the thread and sends it each turn (no
    per-user persistence until auth lands).
    """

    message: str = Field(min_length=1, max_length=2000)
    history: list[BMCChatMessage] = Field(default_factory=list)


class BMCChatResponse(BaseModel):
    answer: str
