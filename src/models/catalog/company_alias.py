"""ORM mapping for the ``company_aliases`` table on the catalog DB.

This table is populated by ``scripts/setup_company_aliases.py`` — a one-time
migration that also supports periodic refreshes. Each row maps a short form /
abbreviation / nickname to a canonical ``company_industry.code`` ticker.

The table lives on the shared catalog Postgres (stock_chat DB), NOT on PRISM's
own Neon DB. Like ``CompanyIndustry``, it uses ``CatalogBase`` so PRISM's
Alembic chain never touches it.

Query patterns:
  * Exact match on ``alias_norm`` (B-tree index) — O(1) for "RIL" → RELIANCE
  * pg_trgm similarity on ``alias_norm`` (GIN index) — O(log n) for "Relianse"
  * Reverse lookup on ``code`` (B-tree index) — "which aliases does RELIANCE have?"
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from src.core.catalog_database import CatalogBase


class CompanyAlias(CatalogBase):
    """One row per alias → canonical NSE ticker mapping."""

    __tablename__ = "company_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alias: Mapped[str] = mapped_column(String, nullable=False)
    alias_norm: Mapped[str] = mapped_column(String, nullable=False, index=True)
    code: Mapped[str] = mapped_column(String, nullable=False, index=True)
    source: Mapped[str] = mapped_column(String, nullable=False, default="algo")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.8)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<CompanyAlias {self.alias!r} → {self.code} ({self.source})>"
