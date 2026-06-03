"""Read-only ORM mappings for the index tables on the investment RDS.

Three tables back the Systematic Portfolio Builder's universe + benchmark:

  * ``indices_list``       — 5 NSE universes (Nifty 50 / Next 50 / 100 / 200 / 500).
  * ``index_constituent``  — dated membership snapshots ``(index_id, security_id,
    date)``. Each ``date`` is a constituent list as-of a rebalance/announcement;
    point-in-time membership for date D = the snapshot with ``max(date) <= D``.
  * ``index_data``         — per-index daily series (level + valuation + risk).
    The benchmark NAV is built from ``daily_return`` (cumulative product).

All owned externally — strictly read-only (``InvestmentBase``).
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import Date, Float, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.core.investment_database import InvestmentBase


class IndicesList(InvestmentBase):
    """One row per index (the Universe dropdown)."""

    __tablename__ = "indices_list"

    index_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    index_name: Mapped[str | None] = mapped_column(Text)
    exchange: Mapped[str | None] = mapped_column(Text)
    index_prowess_code: Mapped[str | None] = mapped_column(Text)

    def __repr__(self) -> str:
        return f"<IndicesList {self.index_id} {self.index_name!r}>"


class IndexConstituent(InvestmentBase):
    """A dated membership snapshot row: ``security_id`` was in ``index_id`` as of
    ``date``. Point-in-time membership = the latest snapshot ``date <= D``."""

    __tablename__ = "index_constituent"

    # No declared PK on the source table; the natural key is the full triple.
    index_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    security_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)

    def __repr__(self) -> str:
        return f"<IndexConstituent idx={self.index_id} sec={self.security_id} {self.date}>"


class IndexData(InvestmentBase):
    """One daily bar for an index — level, valuation, and risk stats. The
    benchmark total-return NAV is ``∏(1 + daily_return)``."""

    __tablename__ = "index_data"

    index_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    marketcap: Mapped[float | None] = mapped_column(Float)
    free_float_marketcap: Mapped[float | None] = mapped_column(Float)
    daily_return: Mapped[float | None] = mapped_column(Float)
    pe: Mapped[float | None] = mapped_column(Float)
    pb: Mapped[float | None] = mapped_column(Float)
    yield_: Mapped[float | None] = mapped_column("yield", Float)
    trading_volume: Mapped[float | None] = mapped_column(Float)
    num_companies: Mapped[float | None] = mapped_column(Float)
    beta: Mapped[float | None] = mapped_column(Float)
    alpha: Mapped[float | None] = mapped_column(Float)
    rsquare: Mapped[float | None] = mapped_column(Float)

    def __repr__(self) -> str:
        return f"<IndexData idx={self.index_id} {self.trade_date} close={self.close}>"
