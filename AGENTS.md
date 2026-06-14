# AGENTS.md — prism-analyst-services (backend)

> If you haven't read the workspace-level [`../AGENTS.md`](../AGENTS.md) and
> [`../PRISM_HANDOFF.md`](../PRISM_HANDOFF.md) yet, do that first. This file is
> the backend-specific addendum.

## Stack at a glance

Python 3.12 · FastAPI + Pydantic v2 · SQLAlchemy 2 async + asyncpg · Alembic ·
Google ADK 1.33 · LiteLLM Router · httpx · pytest · ruff. Deployed via Docker
Compose on EC2; CI/CD via GitHub Actions on `push: [production]`.

## Knowledge graph first (CRITICAL for token economy)

```bash
graphify query "<your question>"           # default — scoped subgraph
graphify explain "<concept name>"           # focused explanation
graphify path "<A>" "<B>"                   # relationships between two nodes
graphify update .                           # refresh after edits (AST only)
```

The graph lives at `graphify-out/`. `Read` and `Grep` are fallbacks when the
graph excerpt is genuinely insufficient — not first-line tools.

## The agent tools (23 when all integrations are enabled)

| Source | Tool | What it does |
|---|---|---|
| `company_tools.py` | `resolve_company` | Name / ticker / ISIN → one `security_id`, OR a structured clarification (MCQ) when ambiguous. **Call FIRST** for any company-scoped question. Replaces the retired catalog-backed `lookup_company`. |
| `company_tools.py` | `resolve_companies` | Batch resolve for comparisons — asks all disambiguations in one form. |
| `company_tools.py` | `search_companies` | Browse view: name/ticker fragment + sector filter (distinct companies). |
| `company_tools.py` | `list_sectors` | The `master_securities` sector taxonomy. |
| `web_search.py` | `web_search` | Gemini AgentTool with `google_search` grounding. |
| `stock_chat.py` | `stock_filings_read` (**v3**) | Narrative filings Q&A. Sends only `question`+`company`+`synthesise`; service's own planner derives every other filter. |
| `stock_chat.py` | `stock_filings_lookup` | Filings catalog metadata only. |
| `stock_chat.py` | `stock_technicals` | Live price + RSI/MACD/MA. |
| `bmc.py` × 6 | `bmc_get` / `_generate` / `_library` / `_get_version` / `_block_chat` / `_diff` | 9-block Business Model Canvas. |
| `prism_financials.py` | `financials_query` | Exact numbers / ratios / rankings / time-series via text-to-SQL over CMIE Prowess. |
| `prism_news.py` × 4 | `news_sentiment` / `news_trending` / `news_search` / `news_compare` | Live Indian-market news + per-company sentiment verdict. |
| `sebi_regulatory.py` × 4 | `sebi_search` / `sebi_recent` / `sebi_deadlines` / `sebi_document` | In-process read-only SEBI corpus (Regulatory Lens) — search circulars/orders, recent filings, compliance deadlines, one doc's AI summary + impact tags. |

Count: company_tools(4) + web_search(1) + stock_chat(3) + bmc(6) +
prism_financials(1) + prism_news(4) + sebi_regulatory(4) = **23**. (The
docstring header in `company_intel.py` still says "14 total / 3 ours" — it
predates `resolve_companies`, `prism_news`, and `sebi_regulatory`; trust this
table + `config/integrations.yml`.)

`company_tools` is the only built-in/in-process tool set on the agent; the rest
arrive through the integration registry. `sebi_regulatory` is registered as a
`python` integration (`config/integrations.yml`) and reads via `sebi_repo.py`
over the SEBI engine — it is NOT UI-only.

(The **Stock Dashboard** investment-DB data is a UI feature with NO agent tool —
direct DB reads via `stock_repo.py` + `/api/v1/stocks/*`.)

The 5 NRE math tools (`compute_*`) stay on disk at `src/services/nre/` but are
NOT attached to the agent — `financials_query` covers the ratio/CAGR/margin
cases deterministically. Re-wire only when a feeder tool needs PRISM-side
computation.

## The standard tool error contract

Every tool either returns a success dict (tool-specific keys) or:

```python
{"ok": False, "error": "<msg>", "error_code": "<snake_case>",
 "next_action": "ask_user_to_retry_later" | "try_alternate_tool"
                | "ask_user_to_clarify" | "give_up_gracefully",
 "retriable": bool, "detail": "<optional, ≤500 chars>"}
```

Build it with `make_error()` from `src/integrations/tools/_errors.py`. The
runner emits `ToolResultEvent(ok=False, error=…)` when `ok` is `False` OR
`error` is truthy. **`{"error": None}` is NOT an error** (the
`prism-financials` success envelope always carries this key).

## HTTP-wrapper pattern (the four teammate services)

`stock_chat._post`, `bmc._request`, `prism_financials._post`, `prism_news._get`
all follow:

- One-shot 250 ms retry on `httpx.TimeoutException` / `httpx.RequestError`.
- 4xx / 5xx are NEVER retried.
- Retry success tags `retry_count: 1` so the runner emits `ToolRetryEvent`.
- HTTP-status routing: 422/400 → `ask_user_to_clarify`; 5xx → retriable; other
  non-200 → `try_alternate_tool` (or a service-specific branch).
- Wrapper trims operational fields before returning (timings, token usage,
  debug, echoed question).

## ADK template hazard

The agent prompt contains zero bare `{identifier}` literals — ADK's
`inject_session_state` raises `KeyError` on them. Double-braces don't escape.
Use **backticks** for code/identifiers and **angle brackets** for placeholders.
The one allowed exception is `<answer_meta>{{…}}` (the prompt is an f-string).
Regression guard: `tests/test_agent_prompts.py::_adk_template_hits`.

## Three database engines — when to use which

There are **three** engines (`src/core/`); the old "catalog DB"
(`company_industry` / `company_aliases`) is **retired** — `catalog_database.py`
and `company_repo.py` are gone. Company lookup is now the `resolve_company`
agent tool over `master_securities` → returns a `security_id` (with an agentic
clarification MCQ when ambiguous).

- **Primary (`DATABASE_URL`, Neon dev / RDS prod)**: the only writable engine —
  `agent_runs`, `chat_conversations` (title/pin/archive/**share**),
  `message_feedback`, `firms`, `users`, `firm_memberships`, `firm_integrations`,
  billing, user preferences, and the portfolio-builder tables. Alembic-controlled.
- **Investment RDS (AWS, `INVESTMENT_DB_*`, READ-ONLY)**: `master_securities`
  (the company resolver lands every query on a `security_id`),
  `prices_and_securities`, `annual_data`, and the index tables. Backs the
  **Stock Dashboard** (`/api/v1/stocks/*`) — a UI feature with **no agent tool**
  (direct reads via `stock_repo.py`) — plus the resolver and portfolio backtests.
  Own `InvestmentBase`; use `investment_session_scope()` / `get_investment_session`
  from `src.core.investment_database`. Inited gracefully in `main.py` lifespan
  (skipped if unconfigured → routes 503). NB: a dual-listed company has two
  `security_id`s (one per exchange); values in `prices_/annual_` are ₹ crore.
- **SEBI Postgres (`SEBI_DB_*`, READ-ONLY)**: the regulatory corpus behind the
  Regulatory Lens (`/api/v1/regulatory/*`) and the `sebi_regulatory` agent tools.
  Reached via `is_sebi_configured()` + `sebi_session_scope()` from
  `src.core.sebi_database`; read through `src/repositories/sebi_repo.py`. Degrades
  to empty/disabled when unset.

Each engine is **separate** — never cross-join across them.

## Auth, chat history & where things live

- **Auth foundation exists** — don't treat the firm as a throwaway dev stub.
  `src/auth/` resolves a provider-agnostic `Principal`; with `AUTH_ENABLED=true`
  it's derived from a Supabase JWT, and `config/access_policy.yml` is the
  anonymous/feature `require(...)` matrix. Billing models (`models/billing.py`)
  are in place. When auth is off, requests resolve to `DEV_FIRM_ID`.
- **Chat history / feedback / share** live in `conversation_repo.py` behind the
  `chat.py` router: `list_conversations` (search / pagination / archived),
  `get_conversation`, `set_title` / `set_pinned` / `set_archived`,
  `hide_conversation`, `upsert_feedback` / `get_feedback_for_runs`, and the
  read-only share surface `create_or_get_share` / `revoke_share` /
  `get_shared_snapshot`.
- **SEBI engine** is reached via `is_sebi_configured()` + `sebi_session_scope()`
  (`src/core/sebi_database.py`) and read through `sebi_repo.py`.

## Adding a tool — the integration framework

1. User fills `docs/INTEGRATION_INTAKE.md` (or hands you the answers).
2. Add an env var to `src/config.py` for the base URL (+ optional `_API_KEY`).
3. New wrapper in `src/integrations/tools/<slug>.py` mirroring `stock_chat.py`
   or `prism_financials.py`.
4. Entry in `config/integrations.yml`. Source types: `python` / `openapi` /
   `mcp` / `agent` only — **there is no `rest` source type**. Teammate REST
   services come in as `python` typed wrappers.
5. Prompt edits in `src/agents/company_intel.py` (TOOL CATALOGUE row +
   failure-handling row + a Rule if response shapes are unusual).
6. Tests in `tests/test_<slug>.py` with the fake-`AsyncClient` factory.
7. `ruff check src/ tests/` and `pytest tests/test_<slug>.py -v`.
8. `graphify update .`.

## Common traps

- `.env` is gitignored — production env-var changes are SSH-applied, not
  deployed via PR. Use the appended-via-ssh pattern in `PRISM_HANDOFF.md` §0.
- Company resolution is the `resolve_company` tool over `master_securities`
  (`src/services/company_resolver.py`) — there are **no** alias tables or
  `company_repo.py` anymore. Don't reintroduce hardcoded aliases; the resolver
  derives acronyms + does typo-tolerant matching from the securities master.
- `google.adk` isn't installed in the Windows venv. Tests that need it skip-
  fail locally and pass in CI. Don't add skip markers.
- `stock_filings_read` (v3) accepts ONLY `question` / `company` / `synthesise`
  — do NOT pre-fill `category` / `period` / `date_from` / `date_to` /
  `max_filings`. The upstream planner derives them.

## Test + lint cheat sheet

```bash
ruff check src/ tests/                          # CI lint
pytest tests/test_prism_financials.py -v        # one file, no DB
pytest tests/test_company_resolver.py -v        # needs CI Postgres
graphify update .                               # after any edit
```

## Files always worth reading first

- `src/agents/company_intel.py` — the prompt is the routing source of truth.
- `config/integrations.yml` — which tools are even registered.
- `src/integrations/tools/_errors.py` — the error contract.
- `tests/conftest.py` — `NullPool` test engine (don't change without reason).
