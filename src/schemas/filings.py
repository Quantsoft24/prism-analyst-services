"""Pydantic schemas for the filings + search API."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class FilingRead(BaseModel):
    """A filing record — metadata + parse status."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    company_id: uuid.UUID
    filing_type: str
    fiscal_period: str | None = None
    title: str | None = None
    source_url: str
    filed_at: date | None = None
    parsed_status: str
    created_at: datetime


class SearchRequest(BaseModel):
    """Body for ``POST /api/v1/search`` — hybrid retrieval over filings."""

    query: str = Field(min_length=1, max_length=1000)
    ticker: str | None = Field(default=None, description="Scope to one NSE/BSE ticker.")
    section: str | None = Field(default=None, description="Filter to a filing section.")
    limit: int = Field(default=10, ge=1, le=50)


class SearchHit(BaseModel):
    """One retrieval result — citation-shaped."""

    filing_id: uuid.UUID
    section: str
    page: int | None = None
    text: str
    fused_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


class SearchResponse(BaseModel):
    query: str
    count: int
    hits: list[SearchHit]
    note: str | None = None
