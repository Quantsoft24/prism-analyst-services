# PRISM Analyst Services

> **AI-powered equity research backend.** FastAPI + PostgreSQL + Google ADK.
> Indian markets (NSE/BSE), agent-first, read-on-demand grounding (no in-house RAG).

> **Coding agent? Start here:** [`AGENTS.md`](AGENTS.md) and
> [`../PRISM_HANDOFF.md`](../PRISM_HANDOFF.md). The workspace supports
> multi-agent collaboration (Claude Code, Antigravity, Cursor, Aider) — those
> files are the shared single source of truth across agent sessions.

## What's in the box

| Surface | Endpoints | Backing |
|---|---|---|
| **Research Chat** | `/api/v1/chat/*` | Google ADK agent (`company_intel`) + the integration registry; conversation history, search, pin/archive, export, 👍/👎 feedback, read-only share links |
| **Stock Dashboard** | `/api/v1/stocks/*` | Direct reads of the investment RDS (security master, prices, financials, announcements, indices, movers) |
| **Regulatory Lens** | `/api/v1/regulatory/*` | Read-only SEBI Postgres (feed, content, deadlines/calendar, topics, weekly summary, bookmarks/alerts) |
| **Portfolio Builder** | `/api/v1/portfolio/*` | Factor screening + durable backtests (async worker), custom factors, saved strategies — on the investment RDS |
| **Business Model Canvas** | `/api/v1/bmc/*` | Thin proxy to the external BMC service |
| **News & Sentiment** | `/api/v1/news/*` | Thin proxy to the external `prism-news` service |
| **Account / Integrations** | `/api/v1/me/*`, `/api/v1/integrations/*` | User profile + usage; per-firm tool toggles |

## Architecture

```
                          ┌──────────────────────────────────────┐
                          │  Frontend (Next.js)  ←→  /api/v1/*   │
                          └────────────────┬─────────────────────┘
                                           │
                          ┌────────────────▼─────────────────────┐
                          │  PRISM Backend (FastAPI, :8000)      │
                          │  ─ chat agent (Google ADK + LiteLLM) │
                          │  ─ stock-dashboard endpoints         │
                          │  ─ regulatory-lens endpoints         │
                          │  ─ portfolio-builder endpoints       │
                          │  ─ BMC / news proxies                │
                          │  ─ integration registry              │
                          └──┬───────────┬───────────┬─────┬──────┘
                             │           │           │     │
            ┌────────────────┘     ┌─────┘           │     └──────────────┐
            ▼                      ▼                 ▼                    ▼
 ┌───────────────────┐ ┌─────────────────────┐ ┌──────────────┐ ┌──────────────────────┐
 │ Primary Postgres  │ │ Investment RDS       │ │ SEBI Postgres│ │ External services    │
 │ (PRISM-owned)     │ │ (AWS, READ-ONLY)     │ │ (READ-ONLY)  │ │  bmc            :8012│
 │ agent_runs,       │ │ master_securities,   │ │ regulatory   │ │  stock-chat     :8011│
 │ chat_conversations│ │ prices_and_securities│ │ documents,   │ │  prism-financials:8000│
 │ message_feedback, │ │ annual_data,         │ │ deadlines,   │ │  prism-news     :8001│
 │ firms/users/      │ │ indices_*            │ │ topics       │ │ (teammate-owned VMs) │
 │ portfolio_*,      │ │ (Stock Dashboard,    │ │ (Regulatory  │ │                      │
 │ firm_integrations │ │  resolver, backtests)│ │  Lens)       │ │                      │
 └───────────────────┘ └─────────────────────┘ └──────────────┘ └──────────────────────┘
```

**Three database engines** (`src/core/`):
- **Primary** (`database.py`) — everything PRISM owns: `agent_runs` (audit + replay),
  `chat_conversations` (title / pin / archive / **share token**), `message_feedback`,
  `firms` / `users` / `firm_memberships`, `firm_integrations`, billing, user preferences,
  and the portfolio-builder tables (`pb_strategies`, `pb_backtests`, custom factors).
- **Investment RDS** (`investment_database.py`, read-only) — `master_securities`
  (8,230 NSE/BSE securities; the **company resolver** lands every query on a `security_id`),
  `prices_and_securities` (daily OHLC/volume/value/market-cap), `annual_data`
  (balance sheet + income statement), and the index tables (for portfolio benchmarks).
  Values in ₹ crore. SSL required. Graceful 503 if unset.
- **SEBI Postgres** (`sebi_database.py`, read-only) — the regulatory corpus behind the
  Regulatory Lens. Graceful empty/disabled if unset.

> **Note:** the old "catalog DB" (`company_industry` / `company_aliases`) is **retired**.
> Company lookup is now the `resolve_company` agent tool over `master_securities`
> (returns a `security_id`, with an agentic clarification MCQ when a name is ambiguous).

**External services (HTTP):** `bmc` (9-block canvas), `stock-chat` (filings narrative Q&A
keyed on `security_id`, technicals), `prism-financials` (text-to-SQL over CMIE Prowess for
exact figures / ratios / rankings), `prism-news` (financial news + sentiment). PRISM's
`/api/v1/bmc/*` and `/api/v1/news/*` thin-proxy to them; the chat agent reaches them via the
integration registry.

## Tech stack

| Layer | Choice |
|---|---|
| Web framework | FastAPI + Pydantic v2 |
| ORM / migrations | SQLAlchemy 2.x (async) + Alembic |
| Primary DB | PostgreSQL (Neon dev/staging; AWS RDS / shared Postgres in prod) |
| Investment DB (read-only) | PostgreSQL (AWS RDS) — Stock Dashboard, company resolver, portfolio data |
| SEBI DB (read-only) | PostgreSQL — Regulatory Lens corpus |
| Agent runtime | Google ADK 1.33+ (LlmAgent, FunctionTool, AgentTool, OpenAPIToolset, MCPToolset) |
| LLM routing | LiteLLM Router — multi-key + multi-model fallback (free + paid tiers) |
| Auth | Provider-agnostic `Principal` (default Supabase JWT; swappable) + `config/access_policy.yml` |
| Background work | `src.portfolio.worker` — durable backtest queue (`pb_backtests`) |
| Tests | pytest + httpx async + real Postgres in CI |
| CI/CD | GitHub Actions → SSH deploy to EC2 + auto `alembic upgrade head` |
| Language | Python 3.12+ |

## Project structure

```
src/
├── main.py                FastAPI app + lifespan (DB engines, ModelRouter,
│                          integration registry, router registration)
├── config.py              Pydantic Settings (env-driven)
├── auth/                  Principal + policy (provider-agnostic; default Supabase JWT)
├── core/
│   ├── database.py        Primary engine (PRISM-owned data)
│   ├── investment_database.py  Read-only investment RDS (Stock Dashboard / resolver / portfolio)
│   ├── sebi_database.py   Read-only SEBI engine (Regulatory Lens); is_sebi_configured()
│   ├── agent_context.py   Per-request agent context
│   └── auth.py            Principal dependency wiring
├── models/                ORM — primary DB
│   ├── base.py, firm.py, user.py, user_preferences.py, agent_run.py,
│   │   chat_conversation.py (title/pin/archive/share), message_feedback.py,
│   │   integration.py, billing.py, portfolio.py
│   └── investment/        Read-only models on the investment engine
│       ├── master_security.py   master_securities (security master + resolver)
│       ├── price_row.py         prices_and_securities (daily bars)
│       ├── annual_data.py       annual financials (balance sheet + income statement)
│       └── index_tables.py      index membership / series (portfolio benchmarks)
├── repositories/          Data access
│   ├── conversation_repo.py  Chat history: list/search/get, pin/archive, rename,
│   │                      delete, per-answer feedback, share (create/revoke/snapshot)
│   ├── stock_repo.py      Securities search + price series + financial trees
│   ├── sebi_repo.py       Regulatory Lens reads (SEBI engine)
│   ├── preferences_repo.py / usage_repo.py / integration_repo.py
├── schemas/               Pydantic request/response shapes
├── routers/               (all registered under settings.API_PREFIX = /api/v1)
│   ├── chat.py            /chat/* — agent SSE + conversation CRUD + feedback + share
│   ├── stocks.py          /stocks/* — investment-DB reads
│   ├── regulatory.py      /regulatory/* — SEBI read-only
│   ├── portfolio.py       /portfolio/* — screening + backtests + strategies
│   ├── bmc.py             /bmc/* — THIN PROXY to BMC_URL
│   ├── news.py            /news/* — THIN PROXY to PRISM_NEWS_URL
│   ├── integrations.py    /integrations — list + per-firm toggle
│   ├── me.py              /me, /me/preferences, /me/usage
│   └── router_health.py   /router/health — ModelRouter debug
├── agents/
│   ├── base.py            PrismAgent (model_tier, integrations seam)
│   ├── company_intel.py   Main chat agent
│   └── web_search.py      Google Search subagent (AgentTool pattern)
├── tools/                 Built-in agent tools
│   └── company_tools.py   resolve_company (→ security_id, w/ clarification) / list_sectors
├── integrations/          Universal integration framework
│   ├── registry.py        Loads config/integrations.yml + builds adapters
│   ├── adapters.py        python / openapi / mcp / agent source types
│   ├── firm_state.py      Per-firm enable/disable resolver
│   └── tools/             Typed wrappers: stock_chat, bmc, prism_financials,
│                          prism_news, sebi_regulatory
├── portfolio/             Factor screening + backtest engine + worker
│   ├── worker.py          Durable backtest queue consumer (run as its own process)
│   └── factors/           Factor library
└── services/
    ├── agent_runner.py    ADK Runner + agent_runs audit row + session rehydration
    ├── model_router.py    LiteLLM Router singleton (tier → deployment)
    └── model_router_config.py  TIER_CONFIGS — single source of model truth
config/
├── integrations.yml              Declarative integration registry
├── access_policy.yml             Anonymous/feature gating policy (the `require(...)` matrix)
├── rate_limits.yml               Daily message caps per tier (drives the chat quota / 429s)
├── balance_sheet_hierarchy.json  Balance-sheet line-item tree
└── income_statement_structure.json  Income-statement line-item structure
alembic/versions/          Migrations — head is 0019_chat_share (see "Database")
docs/                      INTEGRATION_INTAKE.md, SECRETS_MIGRATION.md
evals/                     Behavioural eval harness over the real /chat/run pipeline
.github/workflows/         ci.yml (ruff + alembic + pytest) · deploy.yml (SSH + upgrade head)
Dockerfile, pyproject.toml, .env.example
```

## Local development

### Prerequisites

- Python 3.12+
- PostgreSQL 16+ for PRISM's primary DB (Neon free tier is easiest).
- (Optional) Investment RDS access for `/api/v1/stocks/*` + company resolution; SEBI DB
  access for `/api/v1/regulatory/*`. Both degrade gracefully (503 / empty) when unset.

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
alembic upgrade head                           # primary DB schema (currently 0019_chat_share)
uvicorn src.main:app --reload --port 8000
python -m src.portfolio.worker                 # (optional) portfolio backtest worker
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

CI does this automatically. `pgvector` is kept as a runtime dep because the *historical*
migrations (0001, 0004) import it; the live PRISM code doesn't.

## Environment variables

Minimum required:

| Var | Purpose |
|---|---|
| `DATABASE_URL` | PRISM's primary DB. Format: `postgresql+asyncpg://user:pass@host/db` |
| `GEMINI_API_KEY` | LLM access. Add `GEMINI_API_KEY_1..4` for multi-key resilience |

Data sources + external integrations:

| Var | Purpose |
|---|---|
| `INVESTMENT_DB_*` | Read-only investment RDS (Stock Dashboard, company resolver, portfolio). Provide `INVESTMENT_DB_HOST/PORT/NAME/USER/PASSWORD` parts (not a URL — the password has URL-unsafe chars) + `INVESTMENT_DB_SSL_MODE=require`. Unset → `/api/v1/stocks/*` returns 503 cleanly. The RDS security group must allow the backend host's IP. |
| `SEBI_DB_*` | Read-only SEBI Postgres (Regulatory Lens): `SEBI_DB_HOST/PORT/NAME/USER/PASSWORD`. Unset → `/api/v1/regulatory/*` degrades to empty/disabled. |
| `BMC_URL` | External BMC service base URL. Proxied by `/api/v1/bmc/*`. |
| `PRISM_NEWS_URL` | External news+sentiment service base URL. Proxied by `/api/v1/news/*`. |
| `STOCK_CHAT_URL` | External filings service base URL. Used by the integration registry. |
| `PRISM_FINANCIALS_URL` | External text-to-SQL financials service. **MUST be set explicitly** — the upstream runs on the same port (8000) PRISM binds to, so an unset var would silently loop into PRISM and 404. |
| `PRISM_FINANCIALS_API_KEY` | Optional `X-API-Key` for the financials service (empty today). |

Auth + scope:

| Var | Purpose |
|---|---|
| `AUTH_ENABLED` | When `false` (dev default) requests resolve to the dev firm; when `true` the `Principal` is derived from a Supabase JWT. |
| `SUPABASE_JWT_SECRET` / `SUPABASE_*` | JWT verification when auth is on. |
| `DEV_FIRM_ID` | Dev-mode firm (default `QUANTSOFT`) when auth is off. |
| `MODEL_ROUTER_*` | LiteLLM Router tuning — cooldown, strategy. Defaults work. |

> Anonymous callers are isolated per-browser by the `X-Guest-Id` header; daily message
> caps live in `config/rate_limits.yml` (over-cap → 429). See [`.env.example`](.env.example)
> for the full annotated list and [`docs/SECRETS_MIGRATION.md`](docs/SECRETS_MIGRATION.md)
> for the production secrets runbook (AWS SSM).

## API endpoints

All under the `/api/v1` prefix. Highlights (see `/docs` for the full schema):

**Chat** (`/chat`)
| Method | Path | Description |
|---|---|---|
| `POST` | `/chat/run` | Run the `company_intel` agent — SSE stream (quota-gated; 429 over cap) |
| `GET` | `/chat/conversations?q=&offset=&archived=` | List history — search over question/answer/title, pagination, archived view |
| `GET` | `/chat/conversations/{session_id}` | Replay a conversation's ordered turns (incl. saved feedback) |
| `PATCH` | `/chat/conversations/{session_id}` | Rename / pin / archive (`{title?, pinned?, archived?}`) |
| `DELETE` | `/chat/conversations/{session_id}` | Soft-delete (hide from history) |
| `POST` | `/chat/runs/{agent_run_id}/feedback` | Rate one answer 👍/👎 + reasons + comment |
| `POST` | `/chat/conversations/{session_id}/share` | Create (or get) a read-only public share link |
| `DELETE` | `/chat/conversations/{session_id}/share` | Revoke the share link |
| `GET` | `/chat/shared/{token}` | **Public, no auth** — a frozen read-only conversation snapshot |
| `GET` | `/chat/quota` | Today's message quota for the caller |

**Stocks** (`/stocks`) — `securities`, `{security_id}`, `{security_id}/prices`,
`{security_id}/balance-sheet`, `{security_id}/income-statement`, `reports`, `reports/pdf`,
`announcements`, `indices/latest`, `movers`, `top-companies`.

**Regulatory** (`/regulatory`) — `feed`, `content/{doc_id}`, `recent`, `deadlines`,
`calendar`, `weekly-summary`, `topics`, `types`, `stats`, `bookmarks`, `alerts`,
`me` (GET/PUT preferences).

**Portfolio** (`/portfolio`) — `universes`, `factors`, `factors/preview`, `screen`,
`backtest` (POST), `backtest/{job_id}` (GET/DELETE), `backtests`, `index-series`,
`custom-factors` (GET/POST/validate, DELETE `{cf_id}`), `strategies` (GET/POST,
GET/DELETE `{strategy_id}`).

**BMC** (`/bmc`) — `{ticker}` + `{ticker}/run` / `library` / `{version}` / `diff` /
`export` / `blocks/{block_id}/chat` (all proxied to `BMC_URL`).

**News** (`/news`) — `feed`, `summary`, `trending`, `compare`, `companies`, `sectors`,
`sources`, `stats` (proxied to `PRISM_NEWS_URL`).

**Account / Integrations** — `GET /me`, `PATCH /me/preferences`, `GET /me/usage`;
`GET /api/v1/integrations`, `PUT /api/v1/integrations/{name}` (per-firm toggle).
`GET /health` is the load-balancer probe.

When auth is off, requests resolve to `DEV_FIRM_ID`; when on, send a Supabase bearer token.

## Integrations framework

Adding a new agent-callable resource (an HTTP API, MCP server, in-process Python tool, or
sub-agent) is a single PR:

1. Fill [`docs/INTEGRATION_INTAKE.md`](docs/INTEGRATION_INTAKE.md) (one form per tool).
2. Add ~6 lines to `config/integrations.yml`.
3. (For Python source) drop the typed wrapper module under `src/integrations/tools/`.
4. Restart — the registry builds adapters at startup; agents with `integrations="*"` pick
   them up automatically.

Four source types (`python` · `openapi` · `mcp` · `agent`). **Teammate REST services come
in as `python`-typed wrappers** (there is no `rest` source type). Auth uses env-var
references (never inline secrets), and failures are isolated per integration — a broken
entry shows up as `status=error` on `GET /api/v1/integrations`, not a backend crash.

## Database

Migrations are numbered chronologically (`YYYYMMDD_000N_*`). The current **head is
`0019_chat_share`** (read-only public conversation share columns on `chat_conversations`).
Recent additions: `0017_chat_pin_archive`, `0018_message_feedback`, `0019_chat_share`.

```bash
alembic upgrade head                            # apply all
alembic current                                 # show current revision
alembic revision --autogenerate -m "add foo"    # create a new migration
```

> **Alembic revision ids must be ≤ 32 chars** (`alembic_version.version_num` is `varchar(32)`).

**On deploy:** `alembic upgrade head` runs automatically (see `deploy.yml`). Don't run it by
hand unless you're recovering from a failed deploy.

## Production deployment

### Containers

The 5-container stack is orchestrated by `docker-compose.prod.yml` in the
[frontend repo](https://github.com/Quantsoft24/prism-analyst-platform):

- **landing** — marketing site · **frontend** — Next.js app · **backend** — this service ·
  **worker** — `python -m src.portfolio.worker` (reuses the backend image; runs durable
  portfolio backtests — if it's down, submitted backtests stay `queued`) · **nginx** —
  reverse proxy / TLS.

### CI / CD

- **CI** (`.github/workflows/ci.yml`) — runs on PRs to `main` / `production` and pushes to
  either: ruff lint → `alembic upgrade head` against a real Postgres → pytest with coverage.
- **Deploy** (`.github/workflows/deploy.yml`) — runs on `push: [production]`: SSH to EC2 →
  `git pull` → docker build → restart → **`alembic upgrade head` on the live container** →
  health check → cleanup.

### Branch model

`main` is trunk. `production` is the release pointer (deploys fire only on
`push: [production]`):

```bash
# After PR is approved + merged to main, CI green:
git fetch origin
git push origin main:production    # fast-forward production
```

## License

Proprietary — © 2026 TheQuantSoft. All rights reserved.
