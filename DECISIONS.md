# DECISIONS — `systematic_portfolio_builder`

> **Scope:** this file covers the **`systematic_portfolio_builder`** feature only.
> Other major architectural decisions live in `../PRISM_HANDOFF.md`'s changelog —
> the chat phases (1–6 + 9); the read-only **share** security model (a frozen
> `shared_run_ids` snapshot, public-by-token, with within-firm hardening deferred
> to the auth rollout); the **SEBI Regulatory Lens**; and the **auth/billing
> foundation**.

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

## Performance + attribution (implemented 2026-06-04)
- **Batched backtest engine** (`backtest_data.py`): the run preloads membership,
  prices (close/mcap/trade_value), and the annual panel in **a handful of bulk
  queries**, then rebuilds the point-in-time factor matrix per rebalance entirely
  **in memory** (`factor_matrix_inmem`, same math as `factors.compute`). Measured
  ~**13× faster** (a 3-rebalance Nifty-50 run went 250s → ~19s) with identical
  numbers; a 26-rebalance Nifty-200 run completes in ~53s.
- **Benchmark-relative attribution** (`_attribution`): **sector active weights**
  (portfolio − cap-weighted-universe benchmark), **factor tilts** (portfolio
  weighted-avg z-score vs the universe), and **top/bottom return contributors**
  (Σ weight × period-return). Surfaced in the Attribution tab. The benchmark for
  sector weights is the chosen index's constituents cap-weighted (per-constituent
  official index weights aren't in the schema — documented approximation).
- **Custom factor "backtest this factor"** uses inline custom factors in
  screens/backtests; expressions are base factor ids + arithmetic only (safe
  `ast` evaluator, no `eval`).

## Backtest-detail depth (implemented 2026-06-04, round 2)
- **Switchable benchmark without a re-run.** `GET /portfolio/index-series`
  returns any index's cumulative NAV (growth of ₹1) over a window; the NAV chart
  re-bases it onto the result's date axis and recomputes benchmark metrics
  client-side, so comparing the same book against Nifty 50 / 200 / 500 is instant
  (no new job). The default benchmark stays the run's own `benchmark_index_id`.
- **Always-on style exposure.** Attribution now reports `style_tilts` — the
  portfolio's weighted-avg z-score (vs the universe) on one representative factor
  per classic style: **Value** (earnings yield), **Quality** (ROE), **Growth**
  (3Y revenue CAGR), **Momentum** (12M return), **Size** (market cap), **Low
  Volatility** (−1×vol), signed so "+" always means *more of that style*. The
  style factors are added to the **preload** set (variables only) and computed
  once at the last rebalance, so the per-rebalance screen — and therefore the NAV
  — is unchanged (identical numbers; the ~13× batched speed stands). Sector
  exposure surfaces **absolute portfolio + benchmark weights** alongside active.
- **Delete a backtest.** `DELETE /portfolio/backtest/{id}` (firm-scoped) backs
  the Backtests-list trash action.

## Secrets
- DB credentials come from env only (`.env`, gitignored; `.env.example` documents
  the keys with no real values — team policy keeps even `.env.example` off git).
  All RDS access is **read-only**; new app tables live in PRISM's own DB.
