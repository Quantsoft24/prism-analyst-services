"""Shared constants for the Systematic Portfolio Builder.

Centralises the point-in-time lag, the exact ``annual_data`` variable names we
depend on (verified against the live RDS), statement/basis labels, and the
sector handling for bank/financial names. Keeping the raw variable strings here
(not scattered in queries) means a schema rename is a one-line fix.
"""

from __future__ import annotations

from typing import Final, Literal

# ── Point-in-time reporting lag ──────────────────────────────────────────────
# Indian companies can file annual financials up to ~6 months after fiscal
# year-end. Any ``annual_data`` value is treated as KNOWN only this many months
# after its fiscal period-end (look-ahead-bias guard). Named + configurable so
# it can be tuned in one place; applied everywhere annual fundamentals are used.
ANNUAL_DATA_LAG_MONTHS: Final[int] = 6

# ── Basis (data_type) ────────────────────────────────────────────────────────
Basis = Literal["consolidated", "standalone"]
DEFAULT_BASIS: Final[Basis] = "consolidated"   # fallback to standalone per-name
BASES: Final[tuple[Basis, ...]] = ("consolidated", "standalone")

# ── financial_type labels (annual_data) ──────────────────────────────────────
FT_PL: Final[str] = "profit_and_loss"
FT_ASSET: Final[str] = "asset"
FT_CAPLIAB: Final[str] = "capital and liabilities"

# ── Exact annual_data ``variable`` names we read (verified live) ─────────────
# P&L
V_REVENUE: Final[str] = "Revenue"
V_TOTAL_INCOME: Final[str] = "Total income"
V_TOTAL_EXPENSES: Final[str] = "Total expenses"
V_PBIT: Final[str] = "PBIT"                                   # ≈ EBIT
V_PBT: Final[str] = "PBT"
V_DIRECT_TAX: Final[str] = "Provision for direct tax"
V_INTEREST: Final[str] = "Interest"
V_DEPRECIATION: Final[str] = (
    "Depreciation / Amortisation (net of transfer from revaluation reserves)"
)
# Capital & liabilities
V_EQUITY: Final[str] = "Capital & Reserves"                   # net worth (decided)
V_LT_BORROWINGS: Final[str] = "Long term borrowings excl current portion"
V_ST_BORROWINGS: Final[str] = "Short-term borrowings"
# Asset
V_TOTAL_ASSETS: Final[str] = "Total assets"

# Net profit (PAT) has no direct line — it is PBT − direct tax.
# EBITDA ≈ PBIT + Depreciation.

# ── Sector handling ──────────────────────────────────────────────────────────
# master_securities.sector value for banks/NBFCs/insurers. Leverage, interest
# coverage, and ROCE are excluded for these names (deposits aren't comparable
# debt) — see the factor registry's ``exclude_sectors``.
SECTOR_FINANCIALS: Final[str] = "Financial Services"

# ── Units ────────────────────────────────────────────────────────────────────
# annual_data.value, prices.market_cap, prices.trade_value are all ₹ crore.
UNIT_CRORE: Final[str] = "₹ crore"
