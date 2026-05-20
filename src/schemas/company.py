"""Pydantic schemas for the ``Company`` ORM model — public API shapes."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CompanyAliasRead(BaseModel):
    """Alternate name / code that resolves to a company (BSE code, full name, ...)."""

    model_config = ConfigDict(from_attributes=True)

    kind: str = Field(description="Alias kind — 'name' | 'ticker' | 'bse_code' | ...")
    value: str


class CompanyRead(BaseModel):
    """Compact list-view representation. Use for `/companies` listings.

    Stable wire format — third-party API consumers can depend on these fields.
    Add new fields here freely (additive); never repurpose existing field names.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    ticker: str
    name: str
    exchange: str
    sector: str | None = None
    industry: str | None = None
    country: str
    isin: str | None = None
    is_active: bool


class CompanyDetail(CompanyRead):
    """Full detail-view representation — adds descriptive fields + aliases.

    Used by ``GET /api/v1/companies/{id_or_ticker}``.
    """

    legal_name: str | None = None
    cin: str | None = None
    pan: str | None = None
    website: str | None = None
    description: str | None = None
    aliases: list[CompanyAliasRead] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
