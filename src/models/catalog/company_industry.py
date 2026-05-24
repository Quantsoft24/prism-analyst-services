"""Read-only ORM mapping over the external ``company_industry`` table.

Owned by the stock-chat service (lives in the shared stock_chat Postgres,
4,773 rows). PRISM reads it via the catalog secondary engine to replace its
own retired ``companies`` table. We declare a synthetic PK on ``code`` for
SQLAlchemy identity-map purposes only — the actual table has no explicit PK
constraint, but ``code`` is the natural unique identifier.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.core.catalog_database import CatalogBase


class CompanyIndustry(CatalogBase):
    """One row per NSE/BSE-listed company → industry classification."""

    __tablename__ = "company_industry"

    # SQLAlchemy requires a PK declaration; ``code`` is the natural unique key
    # (scrip code / NSE symbol). Read-only — we never INSERT here.
    code: Mapped[str] = mapped_column(String, primary_key=True)
    industry: Mapped[str | None] = mapped_column(String, index=True)
    company_name: Mapped[str | None] = mapped_column(String, index=True)
    isin: Mapped[str | None] = mapped_column(String, index=True)
    industry_url: Mapped[str | None] = mapped_column(Text)
    industry_rank: Mapped[int | None] = mapped_column(Integer)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    def __repr__(self) -> str:
        return f"<CompanyIndustry {self.code} {self.company_name!r}>"
