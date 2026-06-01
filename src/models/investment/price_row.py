"""Read-only ORM mapping over the external ``prices_and_securities`` table.

Lives on the investment RDS (21.5M rows). Daily end-of-day bars keyed by
``(security_id, trade_date)`` — which is also the table's PRIMARY KEY, so
per-security date-range scans are fully index-backed. Read-only.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from sqlalchemy import BigInteger, Date, Integer, Numeric
from sqlalchemy.orm import Mapped, mapped_column

from src.core.investment_database import InvestmentBase


class PriceRow(InvestmentBase):
    """One daily OHLC / volume / value / market-cap bar for a security."""

    __tablename__ = "prices_and_securities"

    # Composite PK (security_id, trade_date) — matches the real table.
    security_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[Decimal | None] = mapped_column(Numeric)
    high: Mapped[Decimal | None] = mapped_column(Numeric)
    low: Mapped[Decimal | None] = mapped_column(Numeric)
    close: Mapped[Decimal | None] = mapped_column(Numeric)
    trade_volume: Mapped[int | None] = mapped_column(BigInteger)
    trade_value: Mapped[Decimal | None] = mapped_column(Numeric)
    market_cap: Mapped[Decimal | None] = mapped_column(Numeric)

    def __repr__(self) -> str:
        return f"<PriceRow {self.security_id} {self.trade_date} close={self.close}>"
