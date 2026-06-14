"""Read-only repository over the investment DB (Stock Dashboard).

Reads ``master_securities`` (security master) and ``prices_and_securities``
(daily bars). The full security list is cached in-process — it's ~8,230 rows
that change rarely, and the frontend fetches it once to power instant
client-side search.
"""

from __future__ import annotations

import calendar
import json
import time
from datetime import date
from pathlib import Path

from sqlalchemy import bindparam, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.investment import IndexConstituent, IndicesList, MasterSecurity, PriceRow
from src.schemas.stock import (
    BalanceSheetResponse,
    FinancialBasis,
    FinancialNode,
    IncomeRow,
    IncomeStatementResponse,
    IndexLatest,
    MoverKind,
    MoverRow,
    SecurityRead,
    StockRange,
)

# Months to subtract for each calendar-windowed range. ``5D`` (last-N rows) and
# ``MAX`` (all rows) are handled separately.
_RANGE_MONTHS: dict[str, int] = {"1M": 1, "6M": 6, "1Y": 12, "3Y": 36, "5Y": 60}

# In-process cache of the security search index (rarely changes).
_SECURITIES_CACHE: list[SecurityRead] | None = None
_SECURITIES_CACHE_AT: float = 0.0
_SECURITIES_TTL_SECONDS = 6 * 3600

# Top-movers universe (Nifty 200): its name fragment in ``indices_list`` and an
# in-process cache of the computed per-constituent day-move. Prices are EOD, so
# a 30-min TTL is plenty (and one DB pass serves all three tabs).
_MOVERS_UNIVERSE_NAME = "NIFTY 200"
_MOVERS_INDEX_NAME_LIKE = "%200%"      # matched against indices_list.index_name
_MOVERS_TTL_SECONDS = 30 * 60
_MOVERS_MIN_PRICE = 5.0                # drop sub-₹5 names so penny ticks don't dominate
_MOVERS_CACHE: dict | None = None      # {"at": float, "trade_date": date|None, "rows": list[MoverRow]}

# Balance-sheet line-item hierarchy (baked from asset_and_liabilities_parent.csv).
_BS_HIERARCHY_PATH = Path(__file__).resolve().parents[2] / "config" / "balance_sheet_hierarchy.json"
_BS_FINANCIAL_TYPES = ["asset", "capital and liabilities"]
_FIN_YEARS = 10
# Cached nested template: list of root nodes {key, label, level, children:[...]}.
_BS_TREE_TEMPLATE: list[dict] | None = None

# Income-statement sequential structure (editable config — see the file).
_IS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "income_statement_structure.json"
_IS_FINANCIAL_TYPES = ["profit_and_loss"]
_IS_CONFIG: list[dict] | None = None


def _income_statement_config() -> list[dict]:
    """Load + cache the ordered income-statement row config."""
    global _IS_CONFIG
    if _IS_CONFIG is None:
        _IS_CONFIG = json.loads(_IS_CONFIG_PATH.read_text(encoding="utf-8"))
    return _IS_CONFIG


def _balance_sheet_template() -> list[dict]:
    """Load + cache the nested balance-sheet hierarchy (no values attached)."""
    global _BS_TREE_TEMPLATE
    if _BS_TREE_TEMPLATE is not None:
        return _BS_TREE_TEMPLATE
    rows = json.loads(_BS_HIERARCHY_PATH.read_text(encoding="utf-8"))
    children_by_parent: dict[str, list[dict]] = {}
    for r in rows:
        children_by_parent.setdefault(r["parent"], []).append(r)

    def build(row: dict, level: int) -> dict:
        return {
            "key": row["key"],
            "label": row["label"],
            "level": level,
            "children": [build(c, level + 1) for c in children_by_parent.get(row["label"], [])],
        }

    # Roots are the rows whose parent is the financial_type itself.
    roots = [r for r in rows if r["parent"] == r["root"]]
    _BS_TREE_TEMPLATE = [build(r, 0) for r in roots]
    return _BS_TREE_TEMPLATE


def _minus_months(d: date, months: int) -> date:
    """Subtract ``months`` calendar months from ``d``, clamping the day."""
    total = d.year * 12 + (d.month - 1) - months
    year, month = divmod(total, 12)
    month += 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


class StockRepository:
    """Read-only queries for the Stock Dashboard."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_securities(self, *, use_cache: bool = True) -> list[SecurityRead]:
        """All securities as a lightweight search index (cached in-process)."""
        global _SECURITIES_CACHE, _SECURITIES_CACHE_AT
        now = time.monotonic()
        if (
            use_cache
            and _SECURITIES_CACHE is not None
            and now - _SECURITIES_CACHE_AT < _SECURITIES_TTL_SECONDS
        ):
            return _SECURITIES_CACHE

        stmt = select(
            MasterSecurity.security_id,
            MasterSecurity.security_name,
            MasterSecurity.symbol,
            MasterSecurity.isin,
            MasterSecurity.exchange,
            MasterSecurity.sector,
        ).order_by(MasterSecurity.security_name.asc())
        rows = (await self.session.execute(stmt)).all()
        out = [
            SecurityRead(
                security_id=r.security_id,
                security_name=r.security_name,
                symbol=r.symbol,
                isin=r.isin,
                exchange=r.exchange,
                sector=r.sector,
            )
            for r in rows
        ]
        _SECURITIES_CACHE = out
        _SECURITIES_CACHE_AT = now
        return out

    async def get_security(self, security_id: int) -> MasterSecurity | None:
        """One master row (the dashboard header), or ``None`` if not found."""
        return await self.session.get(MasterSecurity, security_id)

    async def get_price_series(
        self, security_id: int, range_: StockRange
    ) -> list[PriceRow]:
        """Daily bars for a security over ``range_``, ascending by trade date.

        ``5D`` returns the last 5 rows; ``MAX`` returns all history; the rest are
        a calendar window anchored at the security's latest trade date (so a
        suspended/delisted name still shows its final window). The PK
        ``(security_id, trade_date)`` makes every variant an index range scan.
        """
        if range_ == "5D":
            stmt = (
                select(PriceRow)
                .where(PriceRow.security_id == security_id)
                .order_by(PriceRow.trade_date.desc())
                .limit(5)
            )
            rows = list((await self.session.scalars(stmt)).all())
            rows.reverse()
            return rows

        base = select(PriceRow).where(PriceRow.security_id == security_id)
        if range_ != "MAX":
            max_date = await self.session.scalar(
                select(func.max(PriceRow.trade_date)).where(
                    PriceRow.security_id == security_id
                )
            )
            if max_date is None:
                return []
            cutoff = _minus_months(max_date, _RANGE_MONTHS[range_])
            base = base.where(PriceRow.trade_date >= cutoff)

        stmt = base.order_by(PriceRow.trade_date.asc())
        return list((await self.session.scalars(stmt)).all())

    # ── Market overview (landing) ──────────────────────────────────────────

    async def get_indices_latest(self, spark_days: int = 30) -> list[IndexLatest]:
        """Latest level + day move + a short sparkline for each index in
        ``indices_list`` (the 5 NSE universes). One windowed query pulls the
        last ``spark_days`` closes per index; the day move is derived from the
        final two closes (robust to a null ``daily_return``)."""
        name_rows = (
            await self.session.execute(
                select(IndicesList.index_id, IndicesList.index_name).order_by(
                    IndicesList.index_id.asc()
                )
            )
        ).all()
        if not name_rows:
            return []
        names = {r.index_id: r.index_name for r in name_rows}

        rows = await self.session.execute(
            text(
                "SELECT index_id, trade_date, close FROM ("
                " SELECT index_id, trade_date, close,"
                "        ROW_NUMBER() OVER (PARTITION BY index_id ORDER BY trade_date DESC) AS rn"
                " FROM index_data"
                ") t WHERE rn <= :n ORDER BY index_id ASC, trade_date ASC"
            ),
            {"n": spark_days},
        )
        series_by_index: dict[int, list[tuple]] = {}
        for index_id, trade_date, close in rows:
            series_by_index.setdefault(index_id, []).append((trade_date, close))

        out: list[IndexLatest] = []
        for index_id, name in names.items():
            series = series_by_index.get(index_id, [])
            closes = [float(c) for (_d, c) in series if c is not None]
            last_date = series[-1][0] if series else None
            level = closes[-1] if closes else None
            change_pct = None
            if len(closes) >= 2 and closes[-2]:
                change_pct = (closes[-1] / closes[-2] - 1.0) * 100.0
            out.append(
                IndexLatest(
                    index_id=index_id,
                    index_name=name,
                    trade_date=last_date,
                    level=level,
                    change_pct=change_pct,
                    spark=closes,
                )
            )
        return out

    async def _movers_universe(self) -> tuple[date | None, list[MoverRow]]:
        """Compute (and cache) the latest-day move for every Nifty 200
        constituent — one DB pass that serves all three movers tabs. Cached for
        ``_MOVERS_TTL_SECONDS`` (prices are end-of-day). Restricting to the
        ~200 index members keeps the scan small (≈200 PK range scans, not a
        21.5M-row table sweep) and the result institutionally meaningful."""
        global _MOVERS_CACHE
        now = time.monotonic()
        if _MOVERS_CACHE is not None and now - _MOVERS_CACHE["at"] < _MOVERS_TTL_SECONDS:
            return _MOVERS_CACHE["trade_date"], _MOVERS_CACHE["rows"]

        def _cache(trade_date: date | None, rows: list[MoverRow]) -> tuple[date | None, list[MoverRow]]:
            global _MOVERS_CACHE
            _MOVERS_CACHE = {"at": now, "trade_date": trade_date, "rows": rows}
            return trade_date, rows

        idx = await self.session.scalar(
            select(IndicesList.index_id)
            .where(IndicesList.index_name.ilike(_MOVERS_INDEX_NAME_LIKE))
            .order_by(IndicesList.index_id.asc())
            .limit(1)
        )
        if idx is None:
            return _cache(None, [])

        latest_snap = await self.session.scalar(
            select(func.max(IndexConstituent.date)).where(IndexConstituent.index_id == idx)
        )
        if latest_snap is None:
            return _cache(None, [])
        member_ids = list(
            (
                await self.session.scalars(
                    select(IndexConstituent.security_id).where(
                        IndexConstituent.index_id == idx,
                        IndexConstituent.date == latest_snap,
                    )
                )
            ).all()
        )
        if not member_ids:
            return _cache(None, [])

        # Last 2 bars per member via a LATERAL — each subquery is an index-backed
        # reverse scan on the (security_id, trade_date) PK that stops after 2
        # rows, instead of a ROW_NUMBER window that would read every member's
        # full price history (the latter ran ~15s cold; this is sub-second).
        bar_rows = await self.session.execute(
            text(
                "SELECT s.security_id, p.rn, p.trade_date, p.close, p.trade_value, p.market_cap "
                "FROM unnest(CAST(:ids AS int[])) AS s(security_id) "
                "CROSS JOIN LATERAL ("
                "  SELECT trade_date, close, trade_value, market_cap,"
                "         ROW_NUMBER() OVER (ORDER BY trade_date DESC) AS rn"
                "  FROM prices_and_securities p2"
                "  WHERE p2.security_id = s.security_id"
                "  ORDER BY trade_date DESC LIMIT 2"
                ") p"
            ),
            {"ids": member_ids},
        )
        latest_by_sec: dict[int, dict] = {}
        prev_close: dict[int, float | None] = {}
        global_latest: date | None = None
        for security_id, rn, trade_date, close, trade_value, market_cap in bar_rows:
            if rn == 1:
                latest_by_sec[security_id] = {
                    "close": float(close) if close is not None else None,
                    "trade_value": float(trade_value) if trade_value is not None else None,
                    "market_cap": float(market_cap) if market_cap is not None else None,
                }
                if trade_date and (global_latest is None or trade_date > global_latest):
                    global_latest = trade_date
            elif rn == 2:
                prev_close[security_id] = float(close) if close is not None else None

        master_rows = (
            await self.session.execute(
                select(
                    MasterSecurity.security_id,
                    MasterSecurity.security_name,
                    MasterSecurity.symbol,
                    MasterSecurity.exchange,
                    MasterSecurity.sector,
                ).where(MasterSecurity.security_id.in_(member_ids))
            )
        ).all()
        master = {r.security_id: r for r in master_rows}

        rows: list[MoverRow] = []
        for sid, latest in latest_by_sec.items():
            close = latest["close"]
            if close is None or close < _MOVERS_MIN_PRICE:
                continue
            pc = prev_close.get(sid)
            change_pct = ((close / pc - 1.0) * 100.0) if pc else None
            m = master.get(sid)
            rows.append(
                MoverRow(
                    security_id=sid,
                    security_name=m.security_name if m else None,
                    symbol=m.symbol if m else None,
                    exchange=m.exchange if m else None,
                    sector=m.sector if m else None,
                    close=close,
                    prev_close=pc,
                    change_pct=change_pct,
                    trade_value=latest["trade_value"],
                    market_cap=latest["market_cap"],
                )
            )
        return _cache(global_latest, rows)

    async def get_movers(
        self, kind: MoverKind, limit: int = 10
    ) -> tuple[str, date | None, list[MoverRow]]:
        """Top gainers / losers / most-active over the cached Nifty 200 universe.
        Returns ``(universe_label, trade_date, rows)``."""
        trade_date, universe = await self._movers_universe()
        if kind == "most_active":
            ranked = sorted(
                (r for r in universe if r.trade_value is not None),
                key=lambda r: r.trade_value or 0.0,
                reverse=True,
            )
        else:
            with_change = [r for r in universe if r.change_pct is not None]
            ranked = sorted(
                with_change,
                key=lambda r: r.change_pct or 0.0,
                reverse=(kind == "gainers"),
            )
        return _MOVERS_UNIVERSE_NAME, trade_date, ranked[:limit]

    async def get_top_companies(self, limit: int = 12) -> list[MoverRow]:
        """Largest companies by latest market cap over the Nifty 200 universe —
        the "suggested to build" set for the BMC dashboard. Reuses the cached
        movers universe (no extra DB scan); big, liquid, institutionally
        recognisable names, derived from data (not a hardcoded list)."""
        _, universe = await self._movers_universe()
        ranked = sorted(
            (r for r in universe if r.market_cap is not None),
            key=lambda r: r.market_cap or 0.0,
            reverse=True,
        )
        return ranked[:limit]

    async def _financials_window(
        self, security_id: int, basis: FinancialBasis, financial_types: list[str]
    ) -> tuple[list[FinancialBasis], FinancialBasis, list[str], dict[str, dict[str, float | None]]]:
        """Shared loader for annual statements: resolve the basis (with fallback),
        the 10 most recent fiscal years, and ``{variable: {year: value}}``.

        Returns ``(available_bases, resolved_basis, years, values_by_var)``;
        ``years``/``values`` are empty when the security has no data."""
        avail_rows = await self.session.execute(
            text(
                "SELECT DISTINCT data_type FROM annual_data "
                "WHERE security_id = :sid AND financial_type IN :fts"
            ).bindparams(bindparam("fts", expanding=True)),
            {"sid": security_id, "fts": financial_types},
        )
        available: list[FinancialBasis] = sorted(r[0] for r in avail_rows)
        if not available:
            return [], basis, [], {}

        resolved: FinancialBasis = basis if basis in available else available[0]

        # 10 most recent fiscal years (date is 'YYYY-MM'), oldest → newest.
        year_rows = await self.session.execute(
            text(
                "SELECT DISTINCT date FROM annual_data "
                "WHERE security_id = :sid AND data_type = :basis AND financial_type IN :fts "
                "ORDER BY date DESC LIMIT :lim"
            ).bindparams(bindparam("fts", expanding=True)),
            {"sid": security_id, "basis": resolved, "fts": financial_types, "lim": _FIN_YEARS},
        )
        years = sorted(r[0] for r in year_rows)
        if not years:
            return available, resolved, [], {}

        # Pull the values for those years in one shot.
        val_rows = await self.session.execute(
            text(
                "SELECT variable, date, value FROM annual_data "
                "WHERE security_id = :sid AND data_type = :basis "
                "AND financial_type IN :fts AND date IN :years"
            ).bindparams(bindparam("fts", expanding=True), bindparam("years", expanding=True)),
            {"sid": security_id, "basis": resolved, "fts": financial_types, "years": years},
        )
        values_by_var: dict[str, dict[str, float | None]] = {}
        for variable, d, value in val_rows:
            values_by_var.setdefault(variable, {})[d] = value
        return available, resolved, years, values_by_var

    async def get_balance_sheet(
        self, security_id: int, basis: FinancialBasis = "consolidated"
    ) -> BalanceSheetResponse:
        """Build the balance-sheet tree (last ~10 fiscal years) for a security.

        Resolves the standalone/consolidated basis (falling back to whichever is
        available), takes the 10 most recent fiscal years, attaches values onto
        the cached hierarchy, and prunes branches that are entirely empty for
        this security (e.g. bank-only lines for a non-bank). Values are ₹ crore.
        """
        available, resolved, years, values_by_var = await self._financials_window(
            security_id, basis, _BS_FINANCIAL_TYPES
        )
        if not available:
            return BalanceSheetResponse(
                security_id=security_id, basis=basis,
                available_bases=[], years=[], sections=[],
            )
        if not years:
            return BalanceSheetResponse(
                security_id=security_id, basis=resolved,
                available_bases=available, years=[], sections=[],
            )

        sections = [
            node
            for tmpl in _balance_sheet_template()
            if (node := _materialize(tmpl, values_by_var, years)) is not None
        ]
        return BalanceSheetResponse(
            security_id=security_id, basis=resolved,
            available_bases=available, years=years, sections=sections,
        )

    async def get_income_statement(
        self, security_id: int, basis: FinancialBasis = "consolidated"
    ) -> IncomeStatementResponse:
        """Build the sequential income statement (Revenue → … → PAT) for a
        security over the last ~10 fiscal years. Input rows read a single
        ``annual_data`` variable; computed rows (Operating Profit / PBT / PAT)
        apply their formula over earlier rows. Values are ₹ crore.
        """
        available, resolved, years, values_by_var = await self._financials_window(
            security_id, basis, _IS_FINANCIAL_TYPES
        )
        if not available:
            return IncomeStatementResponse(
                security_id=security_id, basis=basis,
                available_bases=[], years=[], rows=[],
            )
        if not years:
            return IncomeStatementResponse(
                security_id=security_id, basis=resolved,
                available_bases=available, years=[], rows=[],
            )

        computed: dict[str, dict[str, float | None]] = {}
        rows: list[IncomeRow] = []
        for spec in _income_statement_config():
            if spec["type"] == "input":
                var_vals = values_by_var.get(spec["variable"], {})
                vals: dict[str, float | None] = {y: var_vals.get(y) for y in years}
            else:  # computed — sum signed operands over already-built rows
                vals = {}
                for y in years:
                    total = 0.0
                    seen = False
                    for op, ref in spec["formula"]:
                        v = computed.get(ref, {}).get(y)
                        if v is not None:
                            seen = True
                            total += v if op == "+" else -v
                    vals[y] = total if seen else None
            computed[spec["key"]] = vals
            rows.append(
                IncomeRow(
                    key=spec["key"], label=spec["label"],
                    emphasis=spec.get("emphasis", False),
                    sign=spec.get("sign"), info=spec.get("info"),
                    values=vals,
                )
            )
        return IncomeStatementResponse(
            security_id=security_id, basis=resolved,
            available_bases=available, years=years, rows=rows,
        )


def _materialize(
    tmpl: dict, values_by_var: dict[str, dict[str, float | None]], years: list[str]
) -> FinancialNode | None:
    """Attach values onto a hierarchy node; drop wholly-empty branches."""
    children = [
        child
        for c in tmpl["children"]
        if (child := _materialize(c, values_by_var, years)) is not None
    ]
    row = values_by_var.get(tmpl["label"], {})
    values = {y: row.get(y) for y in years}
    if not children and all(v is None for v in values.values()):
        return None
    return FinancialNode(
        key=tmpl["key"], label=tmpl["label"], level=tmpl["level"],
        values=values, children=children,
    )
