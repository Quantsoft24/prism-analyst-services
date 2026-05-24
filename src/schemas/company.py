"""Pydantic schemas for the public ``/api/v1/companies`` API.

Backed by the catalog DB (``company_industry`` on stock_chat Postgres). Same
wire-format keys as before (frontend unaffected), but some fields are now
``null`` because the catalog doesn't track them (legal_name, cin, pan,
website, description, aliases). ``id`` is a *deterministic* uuid5 derived
from the company's ISBN (or code) so callers that key by id stay stable
across requests.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

_ID_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000001")


def synthetic_company_id(*, isin: str | None, code: str | None) -> uuid.UUID:
    """Deterministic uuid5 from ISIN (preferred — globally unique) or code.
    Stable across requests so the frontend can cache by id."""
    key = (isin or code or "").strip().upper()
    return uuid.uuid5(_ID_NAMESPACE, f"company_industry:{key}")


class CompanyAliasRead(BaseModel):
    """Kept for back-compat with the old detail shape. The catalog doesn't
    track aliases, so this list is always empty in responses today."""

    model_config = ConfigDict(from_attributes=True)

    kind: str
    value: str


class CompanyRead(BaseModel):
    """Compact list-view representation."""

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
    """Full detail-view representation.

    Catalog-only fields (legal_name, cin, pan, website, description,
    aliases) come back as ``null`` / empty list. Kept in the schema so
    third-party API consumers don't break on field absence.
    """

    legal_name: str | None = None
    cin: str | None = None
    pan: str | None = None
    website: str | None = None
    description: str | None = None
    aliases: list[CompanyAliasRead] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
