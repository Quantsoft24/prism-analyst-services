# PRISM Analyst Services

> **AI-powered equity research backend.** FastAPI + PostgreSQL + Google ADK.
> Indian markets, agent-first, read-on-demand grounding (no in-house RAG).

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
                          │  ─ BMC proxy                          │
                          │  ─ integration registry               │
                          └────┬─────────────────┬───────┬────────┘
                               │                 │       │
              ┌────────────────┘                 │       └──────────────┐
              ▼                                  ▼                      ▼
   ┌───────────────────┐          ┌─────────────────────┐    ┌──────────────────┐
   │ Neon Postgres     │          │  stock_chat Postgres │    │ External services │
   │ (PRISM-owned)     │          │  (READ-ONLY catalog) │    │  bmc       :8012  │
   │ agent_runs,       │          │  company_industry    │    │  stock-chat:8011  │
   │ firms, users,     │          │  filings_index       │    │ (teammate-owned;  │
   │ firm_integrations │          │  document_texts      │    │  same GCP VM)     │
   └───────────────────┘          └─────────────────────┘    └──────────────────┘
```

**Where PRISM owns data:** `agent_runs` (audit), `firm_integrations` (per-firm
tool toggles), `firms` / `users` / `firm_memberships` (auth/tenancy).
**Where PRISM reads-only:** `company_industry` (4,773 companies), via a
secondary read-only engine.
**External services (HTTP):** the `bmc` service (9-block canvas) and
`stock-chat` (filings narrative Q&A, catalog lookup, technicals). PRISM's
`/api/v1/bmc/*` thin-proxies to `bmc`; the chat agent reaches both via the
integration registry.

## Tech stack

| Layer | Choice |
|---|---|
| Web framework | FastAPI + Pydantic v2 |
| ORM / migrations | SQLAlchemy 2.x (async) + Alembic |
| Primary DB | PostgreSQL (Neon dev/staging; AWS RDS / shared Postgres in prod) |
| Catalog DB (read-only) | PostgreSQL — shared with stock-chat service (`company_industry`, `filings_index`, `document_texts`) |
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
│   └── auth.py            Dev-mode firm dependency (Clerk in Phase 1 W3)
├── models/                ORM — primary DB
│   ├── base.py, firm.py, user.py, agent_run.py, integration.py
│   └── catalog/           Read-only models on the catalog engine
│       └── company_industry.py
├── repositories/          Data access
│   ├── company_repo.py    Queries company_industry on catalog engine
│   └── integration_repo.py
├── schemas/               Pydantic request/response shapes
├── routers/
│   ├── companies.py       /api/v1/companies — catalog-backed (4,773 rows)
│   ├── bmc.py             /api/v1/bmc/* — THIN PROXY to BMC_URL
│   ├── chat.py            /api/v1/chat/run — agent SSE stream
│   ├── integrations.py    /api/v1/integrations — list + per-firm toggle
│   └── router_health.py   /api/v1/router/health — ModelRouter debug
├── agents/
│   ├── base.py            PrismAgent (model_tier, integrations seam)
│   ├── company_intel.py   Main chat agent
│   └── web_search.py      Google Search subagent (AgentTool pattern)
├── tools/                 Built-in agent tools
│   ├── company_tools.py   lookup_company / search_companies / list_sectors
│   └── nre_tools.py       Deterministic numerical reasoning (compute_*)
├── integrations/          Universal integration framework
│   ├── registry.py        Loads config/integrations.yml + builds adapters
│   ├── adapters.py        python / openapi / mcp / agent source types
│   ├── firm_state.py      Per-firm enable/disable resolver
│   └── tools/             Typed wrappers for external services
│       ├── stock_chat.py  3 tools (read / lookup-filings / technicals)
│       └── bmc.py         6 tools (get / generate / library / version /
│                          block_chat / diff)
├── services/
│   ├── agent_runner.py    ADK Runner + agent_runs audit row
│   ├── model_router.py    LiteLLM Router singleton (tier → deployment)
│   ├── model_router_config.py  TIER_CONFIGS — single source of model truth
│   └── nre/               Deterministic finance math
config/
├── integrations.yml       Declarative integration registry
└── ingestion_sources.yml  (RAG retired; file may be unused)
alembic/versions/          Migrations — see "Database" below
docs/INTEGRATION_INTAKE.md Per-tool intake template (one form per integration)
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
| `CATALOG_DATABASE_URL` (or `POSTGRES_URL`) | Read-only secondary engine → catalog Postgres (`company_industry`). If unset, `/api/v1/companies` returns 503 cleanly. |
| `BMC_URL` | External BMC service base URL (e.g. `http://35.234.221.166:8012`). Proxied by `/api/v1/bmc/*`. |
| `STOCK_CHAT_URL` | External filings service base URL. Used by the integration registry. |

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
