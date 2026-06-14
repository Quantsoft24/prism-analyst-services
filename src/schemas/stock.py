"""Pydantic schemas for the ``/api/v1/stocks`` API (Stock Dashboard).

Backed by the read-only investment DB. Price numbers are emitted as plain
floats (not Decimal strings) so the frontend charting library can consume them
directly; ``date`` fields serialize as ISO ``YYYY-MM-DD`` strings, which
lightweight-charts accepts as time values.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict

# Time-range filters offered by the dashboard. ``MAX`` = full history.
StockRange = Literal["5D", "1M", "6M", "1Y", "3Y", "5Y", "MAX"]


class SecurityRead(BaseModel):
    """Compact search-index item (one row of the client-side search list)."""

    model_config = ConfigDict(from_attributes=True)

    security_id: int
    security_name: str | None = None
    symbol: str | None = None
    isin: str | None = None
    exchange: str | None = None
    sector: str | None = None


class SecurityDetail(SecurityRead):
    """Full master row used for the dashboard header."""

    industry: str | None = None
    basic_industry: str | None = None
    macro_economic_indicator: str | None = None


class PricePoint(BaseModel):
    """One daily bar. ``time`` is the ISO trade date (chart x-axis)."""

    time: date
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    trade_volume: int | None = None
    trade_value: float | None = None
    market_cap: float | None = None


class PriceSeriesResponse(BaseModel):
    """A security's price series for a range, plus its latest bar."""

    security: SecurityDetail
    range: StockRange
    latest: PricePoint | None = None
    points: list[PricePoint]


# ── Market overview (landing) ───────────────────────────────────────────────


class IndexLatest(BaseModel):
    """One index's latest level + day move + a short sparkline (recent closes,
    ascending). Powers the indices strip on the dashboard landing."""

    index_id: int
    index_name: str | None = None
    trade_date: date | None = None
    level: float | None = None        # latest close
    change_pct: float | None = None   # day-over-day % change
    spark: list[float] = []           # recent closes, oldest → newest


MoverKind = Literal["gainers", "losers", "most_active"]


class MoverRow(BaseModel):
    """One row in the top-movers list (a single security's latest-day move)."""

    security_id: int
    security_name: str | None = None
    symbol: str | None = None
    exchange: str | None = None
    sector: str | None = None
    close: float | None = None
    prev_close: float | None = None
    change_pct: float | None = None   # (close / prev_close - 1) * 100
    trade_value: float | None = None  # ₹ (turnover) — drives "most active"
    market_cap: float | None = None   # ₹ crore


class MoversResponse(BaseModel):
    """Top gainers / losers / most-active for the latest trading day, computed
    over a fixed index universe (Nifty 200 constituents)."""

    kind: MoverKind
    universe: str                     # e.g. "NIFTY 200"
    trade_date: date | None = None
    movers: list[MoverRow]


# ── Annual financials (Balance Sheet) ───────────────────────────────────────

FinancialBasis = Literal["standalone", "consolidated"]


class FinancialNode(BaseModel):
    """One line item in a statement tree. ``values`` maps fiscal-year (the
    ``YYYY-MM`` date string) → amount in ₹ crore (``None`` where not reported)."""

    key: str
    label: str
    level: int
    values: dict[str, float | None]
    children: list["FinancialNode"] = []


class BalanceSheetResponse(BaseModel):
    """A security's balance sheet over the last ~10 fiscal years.

    ``sections`` are the top-level trees (Total assets, Capital & Liabilities).
    ``basis`` is the resolved standalone/consolidated view; ``available_bases``
    tells the UI which toggle options actually have data for this security.
    """

    security_id: int
    basis: FinancialBasis
    available_bases: list[FinancialBasis]
    years: list[str]
    sections: list[FinancialNode]


# ── Annual financials (Income Statement) ────────────────────────────────────


class IncomeRow(BaseModel):
    """One line in the sequential income statement. ``values`` maps fiscal-year
    → amount in ₹ crore. ``emphasis`` flags computed subtotals (Operating
    Profit / PBT / PAT); ``sign`` is a display operator for input rows
    (``minus`` deducts, ``plus`` adds); ``info`` is the formula tooltip."""

    key: str
    label: str
    emphasis: bool = False
    sign: Literal["plus", "minus"] | None = None
    info: str | None = None
    values: dict[str, float | None]


class IncomeStatementResponse(BaseModel):
    """A security's income statement over the last ~10 fiscal years (sequential
    rows: Revenue → … → PAT). Same basis/years contract as the balance sheet."""

    security_id: int
    basis: FinancialBasis
    available_bases: list[FinancialBasis]
    years: list[str]
    rows: list[IncomeRow]
