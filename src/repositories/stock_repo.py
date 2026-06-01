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

from src.models.investment import MasterSecurity, PriceRow
from src.schemas.stock import (
    BalanceSheetResponse,
    FinancialBasis,
    FinancialNode,
    IncomeRow,
    IncomeStatementResponse,
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
