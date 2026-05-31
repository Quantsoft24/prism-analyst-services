# PRISM Analyst Services

> **AI-powered equity research backend.** FastAPI + PostgreSQL + Google ADK.
> Indian markets, agent-first, read-on-demand grounding (no in-house RAG).

> **Coding agent? Start here:** [`AGENTS.md`](AGENTS.md) and
> [`../PRISM_HANDOFF.md`](../PRISM_HANDOFF.md). The workspace supports
> multi-agent collaboration (Claude Code, Antigravity, Cursor, Aider) — those
> files are the shared single source of truth across agent sessions.

## Architecture

```
                          ┌──────────────────────────────────────┐
                          │  Frontend (Next.js)  ←→  /api/v1/*   │
                          └────────────────┬─────────────────────┘
                                           │
                          ┌────────────────▼─────────────────────┐
                          │  PRISM Backend (FastAPI, :8000)      │
                          │  ─ chat agent (Google ADK)           │
                          │  ─ company catalog endpoints          │
                          │  ─ stock dashboard endpoints          │
                          │  ─ BMC proxy                          │
                          │  ─ integration registry               │
                          └──┬───────────┬───────────┬─────┬──────┘
                             │           │           │     │
            ┌────────────────┘     ┌─────┘           │     └──────────────┐
            ▼                      ▼                 ▼                    ▼
 ┌───────────────────┐ ┌─────────────────────┐ ┌──────────────┐ ┌──────────────────────┐
 │ Neon Postgres     │ │  stock_chat Postgres │ │ investment   │ │ External services    │
 │ (PRISM-owned)     │ │  (READ-ONLY catalog) │ │ RDS Postgres │ │  bmc            :8012│
 │ agent_runs,       │ │  company_industry    │ │ (READ-ONLY)  │ │  stock-chat     :8011│
 │ firms, users,     │ │  company_aliases     │ │ master_      │ │  prism-financials:8000│
 │ firm_integrations │ │  filings_index       │ │  securities  │ │  prism-news     :8001│
 │                   │ │  document_texts      │ │ prices_…     │ │ (teammate-owned;     │
 │                   │ │                      │ │ annual_data  │ │  GCP VMs)            │
 └───────────────────┘ └─────────────────────┘ └──────────────┘ └──────────────────────┘
```

**Where PRISM owns data:** `agent_runs` (audit), `firm_integrations` (per-firm
tool toggles), `firms` / `users` / `firm_memberships` (auth/tenancy).
**Where PRISM reads-only (two secondary engines):**
- *Catalog DB* (stock_chat Postgres) — `company_industry` (4,773 companies) +
  `company_aliases` (~10k algorithmic abbreviations / short-forms / typo
  variants).
- *Investment DB* (AWS RDS) — backs the **Stock Dashboard**:
  `master_securities` (8,230 NSE/BSE securities), `prices_and_securities`
  (21.5M daily OHLC/volume/value/market-cap bars), and `annual_data` (annual
  financials — balance sheet today). Values in ₹ crore. SSL required.
**External services (HTTP):** `bmc` (9-block canvas), `stock-chat` (filings
narrative Q&A, catalog lookup, technicals), `prism-financials` (text-to-SQL
over CMIE Prowess for exact figures / ratios / rankings), and `prism-news`
(financial news + sentiment). PRISM's `/api/v1/bmc/*` and `/api/v1/news/*`
thin-proxy to those services; the chat agent reaches them via the integration
registry. The Stock Dashboard endpoints (`/api/v1/stocks/*`) are **direct DB
reads** of the investment DB (not a proxy).

## Tech stack

| Layer | Choice |
|---|---|
| Web framework | FastAPI + Pydantic v2 |
| ORM / migrations | SQLAlchemy 2.x (async) + Alembic |
| Primary DB | PostgreSQL (Neon dev/staging; AWS RDS / shared Postgres in prod) |
| Catalog DB (read-only) | PostgreSQL — shared with stock-chat service (`company_industry`, `filings_index`, `document_texts`) |
| Investment DB (read-only) | PostgreSQL (AWS RDS) — Stock Dashboard data (`master_securities`, `prices_and_securities`, `annual_data`) |
| Agent runtime | Google ADK 1.33+ (LlmAgent, FunctionTool, AgentTool, OpenAPIToolset, MCPToolset) |
| LLM routing | LiteLLM Router — multi-key + multi-model fallback (free + paid tiers) |
| Tests | pytest + httpx async + real Postgres in CI |
| CI/CD | GitHub Actions → SSH deploy to EC2 + auto `alembic upgrade head` |
| Language | Python 3.12+ |

## Project structure

```
src/
├── main.py                FastAPI app + lifespan (DB engines, ModelRouter,
│                          integration registry)
├── config.py              Pydantic Settings (env-driven; back-compat for
│                          POSTGRES_URL → CATALOG_DATABASE_URL)
├── core/
│   ├── database.py        Primary engine (PRISM-owned data)
│   ├── catalog_database.py Secondary read-only engine (catalog DB)
│   ├── investment_database.py Secondary read-only engine (investment RDS —
│   │                      Stock Dashboard); own InvestmentBase, graceful if unset
│   └── auth.py            Dev-mode firm dependency (Clerk in Phase 1 W3)
├── models/                ORM — primary DB
│   ├── base.py, firm.py, user.py, agent_run.py, integration.py
│   ├── catalog/           Read-only models on the catalog engine
│   │   ├── company_industry.py
│   │   └── company_alias.py    Algorithmic alias → ticker mappings
│   └── investment/        Read-only models on the investment engine
│       ├── master_security.py  master_securities (security master)
│       └── price_row.py        prices_and_securities (daily bars)
│                          (annual_data is queried via raw SQL in stock_repo)
├── repositories/          Data access
│   ├── company_repo.py    Queries company_industry + company_aliases on
│   │                      catalog engine (3-tier alias resolution: TTL
│   │                      cache → exact alias_norm → pg_trgm similarity)
│   ├── stock_repo.py      Securities search index (cached) + price series
│   │                      (range→window) + balance-sheet tree (from the
│   │                      committed hierarchy config; prunes empty branches)
│   └── integration_repo.py
├── schemas/               Pydantic request/response shapes (incl. stock.py)
├── routers/
│   ├── companies.py       /api/v1/companies — catalog-backed (4,773 rows)
│   ├── stocks.py          /api/v1/stocks/* — investment-DB reads (securities,
│   │                      prices, balance-sheet)
│   ├── bmc.py             /api/v1/bmc/* — THIN PROXY to BMC_URL
│   ├── news.py            /api/v1/news/* — THIN PROXY to PRISM_NEWS_URL
│   ├── chat.py            /api/v1/chat/run — agent SSE stream
│   ├── integrations.py    /api/v1/integrations — list + per-firm toggle
│   └── router_health.py   /api/v1/router/health — ModelRouter debug
├── agents/
│   ├── base.py            PrismAgent (model_tier, integrations seam)
│   ├── company_intel.py   Main chat agent
│   └── web_search.py      Google Search subagent (AgentTool pattern)
├── tools/                 Built-in agent tools
│   ├── company_tools.py   lookup_company / search_companies / list_sectors
│   └── nre_tools.py       Deterministic numerical reasoning (compute_*) —
│                          on disk only; NOT attached to the agent today
│                          (prism-financials covers the ratio cases via SQL).
├── integrations/          Universal integration framework
│   ├── registry.py        Loads config/integrations.yml + builds adapters
│   ├── adapters.py        python / openapi / mcp / agent source types
│   ├── firm_state.py      Per-firm enable/disable resolver
│   └── tools/             Typed wrappers for external services
│       ├── stock_chat.py  3 tools — read (v3: question/company/synthesise
│       │                  only; planner derives every other filter) /
│       │                  lookup-filings / technicals
│       ├── bmc.py         6 tools (get / generate / library / version /
│       │                  block_chat / diff)
│       └── prism_financials.py   1 tool — financials_query (exact
│                          numbers / ratios / rankings via text-to-SQL)
├── services/
│   ├── agent_runner.py    ADK Runner + agent_runs audit row
│   ├── model_router.py    LiteLLM Router singleton (tier → deployment)
│   ├── model_router_config.py  TIER_CONFIGS — single source of model truth
│   └── nre/               Deterministic finance math
config/
├── integrations.yml       Declarative integration registry
├── balance_sheet_hierarchy.json  Balance-sheet line-item tree (variable→parent;
│                          baked from a source CSV; source of truth for the
│                          Annual Financials tree)
└── ingestion_sources.yml  (RAG retired; file may be unused)
alembic/versions/          Migrations — see "Database" below
docs/INTEGRATION_INTAKE.md Per-tool intake template (one form per integration)
scripts/
└── setup_company_aliases.py  Idempotent: creates company_aliases table on the
                          catalog DB and seeds ~10k algorithmic aliases.
                          Re-run after each catalog refresh.
.github/workflows/
├── ci.yml                 ruff + alembic + pytest (against pgvector/pg17)
└── deploy.yml             SSH-to-EC2 → docker compose + alembic upgrade head
Dockerfile, pyproject.toml, .env.example
```

## Local development

### Prerequisites

- Python 3.12+
- PostgreSQL 16+ for PRISM's primary DB. Neon (free) is easiest.
- (Optional) Network access to the catalog DB for `/api/v1/companies` to work.

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -e ".[dev]"
cp .env.example .env
# Edit .env — see "Environment variables" below.
```

### Run

```bash
alembic upgrade head                           # primary DB schema
uvicorn src.main:app --reload --port 8000
```

Open <http://localhost:8000/docs> (DEBUG=true) for Swagger.

### Tests

Backed by a real Postgres + pgvector image (matches CI exactly):

```bash
docker run --rm -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=prism_test -p 5432:5432 -d pgvector/pgvector:pg17

DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/prism_test \
  TEST_DATABASE_URL=$DATABASE_URL DB_SSL_MODE=disable \
  alembic upgrade head && pytest -v
```

CI does this automatically. `pgvector` is kept as a runtime dep because the
*historical* migrations (0001, 0004) import it; the live PRISM code doesn't.

## Environment variables

Minimum required:

| Var | Purpose |
|---|---|
| `DATABASE_URL` | PRISM's primary DB (Neon / RDS). Format: `postgresql+asyncpg://user:pass@host/db` |
| `GEMINI_API_KEY` | LLM access. Add `GEMINI_API_KEY_1..4` for multi-key resilience |

For company catalog + external integrations:

| Var | Purpose |
|---|---|
| `CATALOG_DATABASE_URL` (or `POSTGRES_URL`) | Read-only secondary engine → catalog Postgres (`company_industry`, `company_aliases`). If unset, `/api/v1/companies` returns 503 cleanly. |
| `INVESTMENT_DB_*` | Read-only secondary engine → investment RDS for the Stock Dashboard. Provide the `INVESTMENT_DB_HOST/PORT/NAME/USER/PASSWORD` parts (not a URL — the password has URL-unsafe chars) + `INVESTMENT_DB_SSL_MODE=require`. If unset, `/api/v1/stocks/*` returns 503 cleanly. The RDS security group must allow the backend host's IP. |
| `BMC_URL` | External BMC service base URL (e.g. `http://35.234.221.166:8012`). Proxied by `/api/v1/bmc/*`. |
| `PRISM_NEWS_URL` | External news+sentiment service base URL (prod `http://35.234.221.166:8001`). Proxied by `/api/v1/news/*`. |
| `STOCK_CHAT_URL` | External filings service base URL. Used by the integration registry. |
| `PRISM_FINANCIALS_URL` | External text-to-SQL financials service base URL (prod `http://35.234.221.166:8000`). **MUST be set explicitly** — the default `http://localhost:8013` is a deliberate placeholder; without this var, `financials_query` cannot reach the upstream. The teammate service runs on the same port number (8000) that PRISM itself binds to, so an unset env var would otherwise silently loop into PRISM and 404. |
| `PRISM_FINANCIALS_API_KEY` | Optional `X-API-Key` header for the financials service. Empty today (open endpoint); set when the upstream adds auth. |

Optional / firm scope:

| Var | Purpose |
|---|---|
| `DEV_FIRM_ID` | Dev-mode firm (default `QUANTSOFT`). Replaced when real auth lands. |
| `MODEL_ROUTER_*` | LiteLLM Router tuning — cooldown, strategy. Defaults work. |

See `.env.example` for the full annotated list.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Load-balancer probe |
| `GET` | `/api/v1/companies` | Paginated catalog list (4,773 companies) |
| `GET` | `/api/v1/companies/{id_or_ticker}` | Detail (ticker / NSE scrip code / ISIN) |
| `GET` | `/api/v1/stocks/securities` | Full NSE/BSE security search index (8,230; cached) |
| `GET` | `/api/v1/stocks/{security_id}` | Security master detail (dashboard header) |
| `GET` | `/api/v1/stocks/{security_id}/prices?range=` | Daily OHLC/volume/value/mcap series (5D…MAX) |
| `GET` | `/api/v1/stocks/{security_id}/balance-sheet?basis=` | 10-year balance-sheet tree (standalone/consolidated) |
| `GET` | `/api/v1/news/*` | News feed / sentiment / trending / compare (proxied to `PRISM_NEWS_URL`) |
| `POST` | `/api/v1/chat/run` | Run the company-intel agent — SSE stream |
| `GET` | `/api/v1/bmc/{ticker}` | Latest BMC (proxied to `BMC_URL`) |
| `POST` | `/api/v1/bmc/{ticker}/run` | Generate new BMC version (proxied) |
| `GET` | `/api/v1/bmc/{ticker}/library` | All saved versions (proxied) |
| `GET` | `/api/v1/bmc/{ticker}/{version}` | Specific version (proxied) |
| `POST` | `/api/v1/bmc/{ticker}/blocks/{block_id}/chat` | Block drill-down chat (proxied) |
| `GET` | `/api/v1/bmc/{ticker}/{version}/export?format=pdf\|json` | Export (proxied) |
| `POST` | `/api/v1/bmc/{ticker}/diff` | Temporal diff (proxied) |
| `GET` | `/api/v1/integrations` | List integrations + per-firm enable state |
| `PUT` | `/api/v1/integrations/{name}` | Toggle one integration ON/OFF for the firm |

Auth is dev-mode in the current phase — send `X-Dev-Firm: QUANTSOFT` (or rely
on the default).

## Integrations framework

Adding a new agent-callable resource (an HTTP API, MCP server, in-process
Python tool, or sub-agent) is a single PR:

1. Fill `docs/INTEGRATION_INTAKE.md` (one form per tool).
2. Add ~6 lines to `config/integrations.yml`.
3. (For Python source) drop the typed wrapper module under
   `src/integrations/tools/`.
4. Restart — the registry builds adapters at startup; agents with
   `integrations="*"` pick them up automatically.

The framework supports four source types (`python` · `openapi` · `mcp` ·
`agent`), uses env-var references for auth (never inline secrets), and
isolates failures per integration — a broken entry shows up as `status=error`
on `GET /api/v1/integrations`, not a backend crash.

## Database

Migrations are numbered chronologically (`YYYYMMDD_000N_*`). The current
chain ends at `0009_drop_companies_and_filings` (PRISM's RAG + companies
tables are retired; data is in the catalog DB now).

```bash
alembic upgrade head                            # apply all
alembic current                                 # show current revision
alembic revision --autogenerate -m "add foo"    # create a new migration
```

**On deploy:** `alembic upgrade head` runs automatically (see `deploy.yml`).
Don't run it by hand unless you're recovering from a failed deploy.

## Production deployment

### Containers

The 4-container stack (landing · frontend · backend · nginx) is orchestrated
by `docker-compose.prod.yml` in the
[frontend repo](https://github.com/Quantsoft24/prism-analyst-platform). The
backend service builds from this repo's `Dockerfile`.

### CI / CD

- **CI** (`.github/workflows/ci.yml`) — runs on PRs to `main` / `production`
  and pushes to either: ruff lint → `alembic upgrade head` against a real
  Postgres service container → pytest with coverage.
- **Deploy** (`.github/workflows/deploy.yml`) — runs on `push: [production]`:
  SSH to EC2 → `git pull` → docker build → restart → **`alembic upgrade head`
  on the live container** → health check → cleanup.

### Branch model

`main` is trunk. `production` is the release pointer (deploys fire only on
`push: [production]`). Standard release flow:

```bash
# After PR is approved + merged to main, CI green:
git fetch origin
git push origin main:production    # fast-forward production
```

## License

Proprietary — © 2026 TheQuantSoft. All rights reserved.
