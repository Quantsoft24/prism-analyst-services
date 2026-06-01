"""Read-only ORM mapping over the external ``master_securities`` table.

Lives on the investment RDS (8,230 rows: 5,153 BSE + 3,077 NSE). One row per
listed security (a dual-listed company has two rows — one per exchange — with
distinct ``security_id``s). Read-only; PRISM never writes here.
"""

from __future__ import annotations

from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.core.investment_database import InvestmentBase


class MasterSecurity(InvestmentBase):
    """One row per NSE/BSE-listed security."""

    __tablename__ = "master_securities"

    security_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    security_name: Mapped[str | None] = mapped_column(String)
    symbol: Mapped[str | None] = mapped_column(String)
    isin: Mapped[str | None] = mapped_column(String)
    exchange: Mapped[str | None] = mapped_column(String)
    sector: Mapped[str | None] = mapped_column(String, index=True)
    prowess_code: Mapped[int | None] = mapped_column(Integer)
    basic_industry: Mapped[str | None] = mapped_column(Text, index=True)
    industry: Mapped[str | None] = mapped_column(Text)
    macro_economic_indicator: Mapped[str | None] = mapped_column(Text)

    def __repr__(self) -> str:
        return f"<MasterSecurity {self.security_id} {self.symbol!r} {self.exchange}>"
