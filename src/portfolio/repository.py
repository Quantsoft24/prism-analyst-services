"""Read-only data access for the Systematic Portfolio Builder (investment RDS).

Every read here is point-in-time correct: index membership uses the dated
``index_constituent`` snapshots, and fundamentals apply the 6-month reporting
lag (``src.portfolio.lag``). Strictly SELECT-only — the RDS is owned externally.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.portfolio.constants import (
    ANNUAL_DATA_LAG_MONTHS,
    BASES,
    DEFAULT_BASIS,
    Basis,
)
from src.portfolio.lag import LAG_USABLE_SQL


@dataclass(frozen=True)
class Universe:
    index_id: int
    index_name: str
    exchange: str | None


@dataclass(frozen=True)
class MarketSnap:
    """Latest price bar at-or-before the as-of date (₹ crore market cap)."""

    close: float | None
    market_cap: float | None
    trade_date: date | None


class PortfolioRepository:
    """Point-in-time read queries over the investment RDS."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ── Universe (indices_list / index_constituent) ──────────────────────────

    async def list_universes(self) -> list[Universe]:
        rows = await self.session.execute(
            text("SELECT index_id, index_name, exchange FROM indices_list ORDER BY index_id")
        )
        return [Universe(r.index_id, r.index_name, r.exchange) for r in rows]

    async def members_as_of(self, index_id: int, as_of: date) -> list[int]:
        """Point-in-time constituents: the security_ids in the latest membership
        snapshot with ``date <= as_of``. Empty if the index didn't exist yet."""
        rows = await self.session.execute(
            text(
                """
                WITH snap AS (
                    SELECT max(date) AS d FROM index_constituent
                    WHERE index_id = :iid AND date <= :as_of
                )
                SELECT ic.security_id
                FROM index_constituent ic JOIN snap ON ic.date = snap.d
                WHERE ic.index_id = :iid
                """
            ),
            {"iid": index_id, "as_of": as_of},
        )
        return [r.security_id for r in rows]

    async def latest_membership_date(self, index_id: int, as_of: date) -> date | None:
        return await self.session.scalar(
            text(
                "SELECT max(date) FROM index_constituent "
                "WHERE index_id = :iid AND date <= :as_of"
            ),
            {"iid": index_id, "as_of": as_of},
        )

    # ── Reference (master_securities) ────────────────────────────────────────

    async def sectors(self, security_ids: list[int]) -> dict[int, str | None]:
        if not security_ids:
            return {}
        rows = await self.session.execute(
            text("SELECT security_id, sector FROM master_securities WHERE security_id IN :sids")
            .bindparams(bindparam("sids", expanding=True)),
            {"sids": security_ids},
        )
        return {r.security_id: r.sector for r in rows}

    # ── Market data (prices_and_securities) ──────────────────────────────────

    async def market_snapshot(
        self, security_ids: list[int], as_of: date
    ) -> dict[int, MarketSnap]:
        """Latest close + market_cap at-or-before ``as_of`` per security."""
        if not security_ids:
            return {}
        rows = await self.session.execute(
            text(
                """
                SELECT DISTINCT ON (security_id)
                    security_id, close, market_cap, trade_date
                FROM prices_and_securities
                WHERE security_id IN :sids AND trade_date <= :as_of
                ORDER BY security_id, trade_date DESC
                """
            ).bindparams(bindparam("sids", expanding=True)),
            {"sids": security_ids, "as_of": as_of},
        )
        return {
            r.security_id: MarketSnap(
                close=float(r.close) if r.close is not None else None,
                market_cap=float(r.market_cap) if r.market_cap is not None else None,
                trade_date=r.trade_date,
            )
            for r in rows
        }

    # ── Fundamentals (annual_data) — lagged, per-name basis ──────────────────

    async def fundamentals_snapshot(
        self,
        security_ids: list[int],
        as_of: date,
        variables: list[str],
        *,
        prefer_basis: Basis = DEFAULT_BASIS,
        lag_months: int = ANNUAL_DATA_LAG_MONTHS,
    ) -> dict[int, tuple[Basis, dict[str, float]]]:
        """Latest *usable* (point-in-time, lagged) annual value per security for
        each requested ``variable``.

        Returns ``{security_id: (resolved_basis, {variable: value})}``. Basis is
        resolved per name: ``prefer_basis`` if that basis yields any of the
        requested variables, else the other basis (documented fallback). Missing
        variables are simply absent from the inner dict (caller counts coverage —
        never zero-fills).
        """
        if not security_ids or not variables:
            return {}

        # One pass: latest-usable value per (security, variable, basis) across
        # BOTH bases; choose the basis per security in Python.
        rows = await self.session.execute(
            text(
                f"""
                SELECT DISTINCT ON (security_id, data_type, variable)
                    security_id, data_type, variable, value
                FROM annual_data
                WHERE security_id IN :sids
                  AND data_type IN :bases
                  AND variable IN :vars
                  AND value IS NOT NULL
                  AND {LAG_USABLE_SQL}
                ORDER BY security_id, data_type, variable, date DESC
                """
            ).bindparams(
                bindparam("sids", expanding=True),
                bindparam("bases", expanding=True),
                bindparam("vars", expanding=True),
            ),
            {
                "sids": security_ids,
                "bases": list(BASES),
                "vars": variables,
                "as_of": as_of,
                "lag_months": lag_months,
            },
        )

        # security_id -> basis -> {variable: value}
        by_sec: dict[int, dict[str, dict[str, float]]] = {}
        for r in rows:
            by_sec.setdefault(r.security_id, {}).setdefault(r.data_type, {})[
                r.variable
            ] = float(r.value)

        fallback: Basis = "standalone" if prefer_basis == "consolidated" else "consolidated"
        out: dict[int, tuple[Basis, dict[str, float]]] = {}
        for sid, by_basis in by_sec.items():
            if by_basis.get(prefer_basis):
                out[sid] = (prefer_basis, by_basis[prefer_basis])
            elif by_basis.get(fallback):
                out[sid] = (fallback, by_basis[fallback])
        return out

    async def price_history(
        self, security_ids: list[int], as_of: date, n_rows: int = 280
    ) -> dict[int, list[tuple[date, float | None, float | None]]]:
        """The last ``n_rows`` daily bars at-or-before ``as_of`` per security,
        ascending by date — for momentum / volatility / ADV. Returns
        ``{security_id: [(trade_date, close, trade_value), …]}``."""
        if not security_ids:
            return {}
        rows = await self.session.execute(
            text(
                """
                SELECT security_id, trade_date, close, trade_value
                FROM (
                    SELECT security_id, trade_date, close, trade_value,
                           row_number() OVER (
                               PARTITION BY security_id ORDER BY trade_date DESC
                           ) AS rn
                    FROM prices_and_securities
                    WHERE security_id IN :sids AND trade_date <= :as_of
                ) t
                WHERE rn <= :n
                ORDER BY security_id, trade_date ASC
                """
            ).bindparams(bindparam("sids", expanding=True)),
            {"sids": security_ids, "as_of": as_of, "n": n_rows},
        )
        out: dict[int, list[tuple[date, float | None, float | None]]] = {}
        for r in rows:
            out.setdefault(r.security_id, []).append(
                (
                    r.trade_date,
                    float(r.close) if r.close is not None else None,
                    float(r.trade_value) if r.trade_value is not None else None,
                )
            )
        return out

    async def fundamentals_series(
        self,
        security_ids: list[int],
        as_of: date,
        variables: list[str],
        *,
        prefer_basis: Basis = DEFAULT_BASIS,
        n_years: int = 6,
        lag_months: int = ANNUAL_DATA_LAG_MONTHS,
    ) -> dict[int, tuple[Basis, dict[str, list[tuple[str, float]]]]]:
        """Up to ``n_years`` of *usable* (lagged) annual values per security for
        each ``variable``, ascending by fiscal period — for growth/CAGR factors.

        Returns ``{security_id: (resolved_basis, {variable: [(period, value), …]})}``
        with the basis resolved per name (prefer → fallback), same as
        ``fundamentals_snapshot``."""
        if not security_ids or not variables:
            return {}
        rows = await self.session.execute(
            text(
                f"""
                SELECT security_id, data_type, variable, date, value
                FROM (
                    SELECT security_id, data_type, variable, date, value,
                           row_number() OVER (
                               PARTITION BY security_id, data_type, variable
                               ORDER BY date DESC
                           ) AS rn
                    FROM annual_data
                    WHERE security_id IN :sids
                      AND data_type IN :bases
                      AND variable IN :vars
                      AND value IS NOT NULL
                      AND {LAG_USABLE_SQL}
                ) t
                WHERE rn <= :n_years
                ORDER BY security_id, data_type, variable, date ASC
                """
            ).bindparams(
                bindparam("sids", expanding=True),
                bindparam("bases", expanding=True),
                bindparam("vars", expanding=True),
            ),
            {
                "sids": security_ids,
                "bases": list(BASES),
                "vars": variables,
                "as_of": as_of,
                "lag_months": lag_months,
                "n_years": n_years,
            },
        )
        # security_id -> basis -> variable -> [(period, value)]
        by_sec: dict[int, dict[str, dict[str, list[tuple[str, float]]]]] = {}
        for r in rows:
            by_sec.setdefault(r.security_id, {}).setdefault(r.data_type, {}).setdefault(
                r.variable, []
            ).append((r.date, float(r.value)))

        fallback: Basis = "standalone" if prefer_basis == "consolidated" else "consolidated"
        out: dict[int, tuple[Basis, dict[str, list[tuple[str, float]]]]] = {}
        for sid, by_basis in by_sec.items():
            if by_basis.get(prefer_basis):
                out[sid] = (prefer_basis, by_basis[prefer_basis])
            elif by_basis.get(fallback):
                out[sid] = (fallback, by_basis[fallback])
        return out

    async def latest_trade_date(self) -> date | None:
        """Most recent trading day in the price table — the de-facto 'today'."""
        return await self.session.scalar(
            text("SELECT max(trade_date) FROM prices_and_securities")
        )

    async def securities_meta(
        self, security_ids: list[int]
    ) -> dict[int, tuple[str | None, str | None, str | None]]:
        """``{security_id: (symbol, security_name, sector)}`` for display."""
        if not security_ids:
            return {}
        rows = await self.session.execute(
            text(
                "SELECT security_id, symbol, security_name, sector "
                "FROM master_securities WHERE security_id IN :sids"
            ).bindparams(bindparam("sids", expanding=True)),
            {"sids": security_ids},
        )
        return {r.security_id: (r.symbol, r.security_name, r.sector) for r in rows}

    # ── Backtest bulk reads ──────────────────────────────────────────────────

    async def benchmark_series(
        self, index_id: int, start: date, end: date
    ) -> list[tuple[date, float | None]]:
        """The benchmark trading-day axis + daily total return for an index —
        ``[(trade_date, daily_return_fraction), …]`` ascending. NAV = ∏(1 + r).

        ``index_data.daily_return`` is stored in **percent** (verified: 0.43 ==
        +0.43%, matching the close ratio); we normalise to a fraction here so all
        callers get clean returns."""
        rows = await self.session.execute(
            text(
                "SELECT trade_date, daily_return FROM index_data "
                "WHERE index_id = :iid AND trade_date BETWEEN :start AND :end "
                "ORDER BY trade_date ASC"
            ),
            {"iid": index_id, "start": start, "end": end},
        )
        return [
            (r.trade_date, float(r.daily_return) / 100.0 if r.daily_return is not None else None)
            for r in rows
        ]

    async def closes_panel(
        self, security_ids: list[int], start: date, end: date
    ) -> dict[int, list[tuple[date, float | None]]]:
        """Adjusted daily closes per security over ``[start, end]`` for the
        backtest return matrix — ``{security_id: [(trade_date, close), …]}``
        ascending. Prices are corporate-action adjusted (owner-confirmed)."""
        if not security_ids:
            return {}
        rows = await self.session.execute(
            text(
                "SELECT security_id, trade_date, close FROM prices_and_securities "
                "WHERE security_id IN :sids AND trade_date BETWEEN :start AND :end "
                "ORDER BY security_id, trade_date ASC"
            ).bindparams(bindparam("sids", expanding=True)),
            {"sids": security_ids, "start": start, "end": end},
        )
        out: dict[int, list[tuple[date, float | None]]] = {}
        for r in rows:
            out.setdefault(r.security_id, []).append(
                (r.trade_date, float(r.close) if r.close is not None else None)
            )
        return out
