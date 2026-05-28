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

## The 14 agent tools

| Source | Tool | What it does |
|---|---|---|
| `company_tools.py` | `lookup_company` | Ticker / ISIN / **alias** / fuzzy name. |
| `company_tools.py` | `search_companies` | Sector filter + fuzzy resolution. |
| `company_tools.py` | `list_covered_sectors` | The canonical sector list. |
| `web_search.py` | `web_search` | Gemini AgentTool with `google_search` grounding. |
| `stock_chat.py` | `stock_filings_read` (**v3**) | Narrative filings Q&A. Sends only `question`+`company`+`synthesise`; service's own planner derives every other filter. |
| `stock_chat.py` | `stock_filings_lookup` | Filings catalog metadata only. |
| `stock_chat.py` | `stock_technicals` | Live price + RSI/MACD/MA. |
| `bmc.py` × 6 | `bmc_get` / `_generate` / `_library` / `_get_version` / `_block_chat` / `_diff` | 9-block Business Model Canvas. |
| `prism_financials.py` | `financials_query` | Exact numbers / ratios / rankings / time-series via text-to-SQL over CMIE Prowess. |

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

## HTTP-wrapper pattern (the three teammate services)

`stock_chat._post`, `bmc._request`, `prism_financials._post` all follow:

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

## Two databases — when to use which

- **Primary (Neon, `DATABASE_URL`)**: writes — `agent_runs`, `firms`, `users`,
  `firm_memberships`, `firm_integrations`. Alembic-controlled.
- **Catalog (teammate VM `35.234.221.166:5434/stock_chat`, `POSTGRES_URL` /
  `CATALOG_DATABASE_URL`)**: reads only — `company_industry`, **`company_aliases`**,
  `filings_index`, `document_texts`, `bmc_*`, `chunks`. Use
  `catalog_session_scope()` from `src.core.catalog_database`.

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
- `_normalize_query` in `company_repo.py` ↔ `_normalize` in
  `scripts/setup_company_aliases.py` MUST stay identical. Break one and alias
  lookups silently miss.
- `google.adk` isn't installed in the Windows venv. Tests that need it skip-
  fail locally and pass in CI. Don't add skip markers.
- `stock_filings_read` (v3) accepts ONLY `question` / `company` / `synthesise`
  — do NOT pre-fill `category` / `period` / `date_from` / `date_to` /
  `max_filings`. The upstream planner derives them.

## Test + lint cheat sheet

```bash
ruff check src/ tests/                          # CI lint
pytest tests/test_prism_financials.py -v        # one file, no DB
pytest tests/test_company_repo.py -v            # needs CI Postgres
graphify update .                               # after any edit
```

## Files always worth reading first

- `src/agents/company_intel.py` — the prompt is the routing source of truth.
- `config/integrations.yml` — which tools are even registered.
- `src/integrations/tools/_errors.py` — the error contract.
- `tests/conftest.py` — `NullPool` test engine (don't change without reason).
