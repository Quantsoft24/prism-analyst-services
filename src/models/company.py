"""Company metadata for BSE/NSE-listed (and later, MCA-21 unlisted) entities.

This table is **global** — a company like RELIANCE is one row, shared across
every firm. Per-firm watchlists, BMC analyses, and reports reference it by
``company_id`` and add their own ``firm_id`` for tenancy.
"""

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin, UUIDPKMixin


class Company(UUIDPKMixin, TimestampMixin, Base):
    """A public-markets entity. Currently India-only (BSE/NSE). Year 2 adds
    US/EM as additional rows distinguished by ``exchange``.
    """

    __tablename__ = "companies"
    __table_args__ = (
        UniqueConstraint("exchange", "ticker", name="uq_companies_exchange_ticker"),
    )

    ticker: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    legal_name: Mapped[str | None] = mapped_column(String(512))

    # Exchange identification — "NSE", "BSE", "NYSE", etc.
    exchange: Mapped[str] = mapped_column(String(16), nullable=False, server_default="NSE")
    isin: Mapped[str | None] = mapped_column(String(12), unique=True, index=True)
    cin: Mapped[str | None] = mapped_column(String(32), index=True)  # India Corporate ID
    pan: Mapped[str | None] = mapped_column(String(10))

    # Classification — kept as plain strings in Slice 1; in S4 we'll wire
    # in a proper GICS/NIC sector hierarchy table.
    sector: Mapped[str | None] = mapped_column(String(128), index=True)
    industry: Mapped[str | None] = mapped_column(String(128))
    country: Mapped[str] = mapped_column(String(2), nullable=False, server_default="IN")

    # Lightweight descriptive fields — feed into BMC, RAG, and company cards.
    website: Mapped[str | None] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)

    is_active: Mapped[bool] = mapped_column(default=True, server_default="true", nullable=False)

    aliases: Mapped[list[CompanyAlias]] = relationship(
        back_populates="company", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Company {self.exchange}:{self.ticker} {self.name!r}>"


class CompanyAlias(UUIDPKMixin, TimestampMixin, Base):
    """Alternate names / tickers — analysts say "Tata Consultancy", "TCS",
    "TCS.NS", "532540" (BSE code). All resolve to one ``Company`` row.
    """

    __tablename__ = "company_aliases"
    __table_args__ = (
        UniqueConstraint("kind", "value", name="uq_company_aliases_kind_value"),
    )

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)  # name | ticker | bse_code | ...
    value: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    company: Mapped[Company] = relationship(back_populates="aliases")
