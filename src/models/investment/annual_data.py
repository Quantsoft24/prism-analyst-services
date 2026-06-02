"""Read-only ORM mapping for ``annual_data`` on the investment RDS.

Long/EAV fundamentals table (~25M rows). One row = one line-item value for one
security, fiscal period, basis, and statement:

  ``(security_id, date 'YYYY-MM', variable, value ₹crore, data_type, financial_type)``

  * ``date``          — fiscal period end as ``'YYYY-MM'`` (mostly March FYs,
    2000-03 → 2026-03). Parsed to a real month-end date by ``portfolio.lag``.
  * ``data_type``     — ``'consolidated'`` | ``'standalone'`` (the basis).
  * ``financial_type``— ``'asset'`` | ``'capital and liabilities'`` |
    ``'profit_and_loss'``.
  * ``variable``      — the line-item name (68 distinct).
  * ``value``         — ₹ crore.

No declared PK on the source; the natural key is the full 5-tuple. Read-only.
Point-in-time correctness (the 6-month reporting lag) is applied centrally in
``src.portfolio.lag`` — never read this table without it for screening/backtest.
"""

from __future__ import annotations

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.core.investment_database import InvestmentBase


class AnnualData(InvestmentBase):
    """One annual line-item value (EAV)."""

    __tablename__ = "annual_data"

    security_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[str] = mapped_column(String, primary_key=True)  # 'YYYY-MM'
    variable: Mapped[str] = mapped_column(Text, primary_key=True)
    data_type: Mapped[str] = mapped_column(Text, primary_key=True)
    financial_type: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[float | None] = mapped_column(Float)

    def __repr__(self) -> str:
        return (
            f"<AnnualData sec={self.security_id} {self.date} {self.variable!r} "
            f"{self.data_type} {self.value}>"
        )
