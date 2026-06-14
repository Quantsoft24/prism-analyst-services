"""Unit tests for the Stock Dashboard top-movers ranking.

These exercise ``StockRepository.get_movers`` ranking/limit/filter logic in
isolation (no DB) by stubbing the expensive ``_movers_universe`` DB pass — the
SQL itself runs against the investment RDS, which isn't available in CI.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.repositories.stock_repo import StockRepository
from src.schemas.stock import MoverRow


def _row(sid: int, sym: str, change: float | None, value: float | None) -> MoverRow:
    return MoverRow(
        security_id=sid,
        security_name=sym,
        symbol=sym,
        exchange="NSE",
        sector=None,
        close=100.0,
        prev_close=99.0,
        change_pct=change,
        trade_value=value,
        market_cap=None,
    )


@pytest.mark.asyncio
async def test_get_movers_ranks_filters_and_limits() -> None:
    repo = StockRepository(session=None)  # type: ignore[arg-type]  # session unused on this path
    universe = [
        _row(1, "A", 5.0, 100.0),
        _row(2, "B", -3.0, 300.0),
        _row(3, "C", 1.0, 200.0),
        _row(4, "D", None, 50.0),  # no change_pct → excluded from gainers/losers
    ]

    async def fake_universe() -> tuple[date, list[MoverRow]]:
        return date(2026, 5, 31), universe

    repo._movers_universe = fake_universe  # type: ignore[method-assign]

    label, trade_date, gainers = await repo.get_movers("gainers", 2)
    assert label == "NIFTY 200"
    assert trade_date == date(2026, 5, 31)
    assert [r.symbol for r in gainers] == ["A", "C"]  # change desc, limited to 2

    _, _, losers = await repo.get_movers("losers", 5)
    assert [r.symbol for r in losers] == ["B", "C", "A"]  # change asc, D (null) dropped

    _, _, active = await repo.get_movers("most_active", 2)
    assert [r.symbol for r in active] == ["B", "C"]  # trade_value desc
