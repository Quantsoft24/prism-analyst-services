# DECISIONS — `systematic_portfolio_builder`

Records where the implementation **deviates from the mock**, the **data-driven
calls** we made, and the **institutional-correctness** choices. The mock's data
and factor list were illustrative; everything here is backed by the six
whitelisted tables on the investment RDS (verified live, read-only).

## Data / schema
- **`index_list` → `indices_list`.** The spec's whitelisted `index_list` is
  actually `indices_list` (5 NSE universes: 1 Nifty 50, 2 Next 50, 3 Nifty 100,
  4 Nifty 200, 5 Nifty 500). All six tables exist.
- **`annual_data` is EAV** (`security_id, date 'YYYY-MM', variable, value ₹cr,
  data_type standalone/consolidated, financial_type asset/'capital and
  liabilities'/profit_and_loss`). It holds only balance-sheet + P&L **line items**
  — **no ratios, no dividends, no cashflow**. Every factor is therefore computed.
- **PAT (net profit) = `PBT` − `Provision for direct tax`** — there is no direct
  net-profit line.
- **Equity / net worth = `Capital & Reserves`** (verified: = paid-up + reserves
  [+ minority interest on consolidated], positive and < total assets across
  RELIANCE/HDFCBANK/TCS/INFY). `Total capital` is a small share-capital line and
  is **not** net worth. Used for P/B and ROE.
- **Prices are corporate-action adjusted** (owner-confirmed) → close-to-close
  returns are clean; no de-distortion needed. Daily `market_cap` is on the price
  table → point-in-time market cap directly.
- **Benchmark NAV = cumulative `index_data.daily_return`.** That column is stored
  in **percent** (0.43 == +0.43%, matching the close ratio) — normalised to a
  fraction in `repository.benchmark_series`.

## Factors (mock list replaced with a schema-derived registry)
- 18-factor registry across valuation / quality / growth / size / momentum /
  liquidity / volatility, plus user **custom factors** (expressions over base
  ids). Metadata-driven (`factors/registry.py`).
- **Dropped (no data): dividend yield** (no dividend variable; `index_data.yield`
  is index-level only). Any mock factor not backed by these tables is dropped.
- **Sector-aware applicability:** **Debt/Equity, Interest coverage, and ROCE are
  excluded for `Financial Services`** (bank liabilities are deposits, not
  comparable debt) — excluded names return `None` and are counted in coverage,
  never zero-filled.
- **Basis:** consolidated by default, per-name fallback to standalone.

## Point-in-time correctness (the USP)
- **6-month annual-data reporting lag** — `ANNUAL_DATA_LAG_MONTHS = 6`
  (`constants.py`); a value is usable only `period_end + 6 months`. Applied
  centrally in `lag.py` (Python + a reusable `LAG_USABLE_SQL` fragment) across
  screening, factor preview, and backtest.
- **No survivorship / look-ahead bias:** each rebalance uses the dated
  `index_constituent` snapshot (`max(date) ≤ rebalance`) and lagged fundamentals
  as of that date.
- **Missing data is explicit:** APIs report coverage ("computable for N of M");
  null inputs exclude a name, never zero it.

## Weighting
- Offered: **Equal** (default) / **Market-cap** / **Factor-score (rank tilt)** /
  **Inverse-vol**, with optional **per-name and per-sector caps** (water-filling).
- **Free-float weighting is not offered** — `free_float_marketcap` exists only at
  the index level, not per security.

## Architecture
- **Backtest = durable async job system**, not synchronous (per owner: production-
  /future-ready for full Nifty 500 × full history). Postgres-backed queue
  (`pb_backtests`) + a separate **worker** (`python -m src.portfolio.worker`,
  `FOR UPDATE SKIP LOCKED`, restart-safe via stale-job reclaim). Vectorized NAV
  with **numpy** (re-added as a dep). Strategy-hash **result cache**.
- **Persistence** lives in PRISM's **primary Postgres** (Alembic `0010`, `0011`)
  — never the read-only RDS. Firm-scoped (`firm_id` slug) + nullable
  `created_by` for per-user filtering once auth populates it.
- **Backtest result is stored as one `result` JSONB** (NAV/benchmark/drawdown/
  metrics/rebalances incl. holdings) rather than a separate holdings table — a
  simplification; revisit if per-holding querying is needed.

## Known limitations / deferred
- **Attribution tab** ships **sector exposure** (portfolio sector weights at the
  latest rebalance). Full benchmark-relative **factor attribution + sector drift
  vs the benchmark** is deferred (needs benchmark constituent weights per date).
- **Performance:** the per-rebalance screen loop is sequential over the RDS;
  fine on the prod worker (near the DB), but **batching factor computation across
  rebalances** is the headline optimization for the largest runs.
- **Custom factor "backtest this factor"** uses the live preview + inline custom
  factors in backtests; expressions are restricted to base factor ids + arithmetic
  (safe `ast` evaluator, no `eval`).

## Secrets
- DB credentials come from env only (`.env`, gitignored; `.env.example` documents
  the keys with no real values — team policy keeps even `.env.example` off git).
  All RDS access is **read-only**; new app tables live in PRISM's own DB.
