# Integration Intake

How PRISM onboards a new agent tool — an external REST API, an MCP server, a
Python tool, or a sub-agent. Goal: a teammate fills **one form per tool**, and
the tool becomes agent-callable via the Integration Registry with **no agent
code changes** (just a `config/integrations.yml` entry).

Supported integration types (all native to Google ADK 1.33):

| Type | ADK primitive | Use when |
|------|---------------|----------|
| `openapi` | `OpenAPIToolset` | The API has an OpenAPI/Swagger spec — every endpoint becomes a tool automatically. **Preferred.** |
| `rest` | `FunctionTool` (thin wrapper) | A REST endpoint with no machine-readable spec. |
| `mcp` | `MCPToolset` | The tool is exposed over the Model Context Protocol (stdio or HTTP/SSE). |
| `python` | `FunctionTool` | An in-process Python function (like the NRE/BMC tools today). |
| `agent` | `AgentTool` | A specialist sub-agent callable by other agents. |

---

## Part A — Platform decisions (answered ONCE, by the product owner)

- [ ] **Governance:** who can add/toggle integrations — any user or admin-only?
- [ ] **Scope:** integrations are firm-wide or per-user?
- [ ] **Per-agent control:** can specific tools be assigned to specific agents?
- [ ] **Default state** for a newly added tool: on, or off-until-reviewed?
- [ ] **Compliance:** are external-tool calls logged in the audit trail / subject
      to MNPI–PII redaction / citation policy?
- [ ] **Usage & cost** tracking per integration — required?
- [ ] **Failure behaviour** when a tool is down: skip silently / tell the user / retry.

---

## Part B — Per-tool intake  (COPY THIS BLOCK ONCE PER TOOL)

### Identity & purpose
- **Name:**
- **One-line purpose** (what it does + when the agent should call it):
- **Type:** `openapi` | `rest` | `mcp` | `python` | `agent`
- **Owner / contact + repo:**

### Interface  (fill the sub-section matching the type)

**If `openapi`:**
- Spec URL or file (OpenAPI 2 or 3?):
- Base URL(s): dev / staging / prod:

**If `rest` (no spec) — repeat per endpoint:**
- Method + path:
- Request schema (field · type · required?):
- Response schema:
- Example request + response:
- Error shape + status codes:

**If `mcp`:**
- Transport: stdio command  |  HTTP/SSE URL:
- How to launch / connect:
- Tools it exposes (names + purpose):

**If `python` / `agent`:**
- Import path (module:function or agent factory):

### Auth & secrets
- **Method:** none | API key (header name: ____) | Bearer | OAuth2 | mTLS
- **Where to obtain the credential:**
- **Shared service credential or per-user?**
- **Env var name to hold it** (never inline a secret — e.g. `CREDIT_API_KEY`):

### Operational & safety
- **Read-only, or does it mutate/charge anything?**
- **Typical latency:**
- **Rate limits / quotas / timeout:**
- **Data sensitivity** (PII / MNPI / client data received or returned?):
- **Which environment to wire first** (dev/staging/prod):
- **Golden example call** (inputs → expected output) for the integration test:

### Agent guidance (so the LLM calls it correctly)
- **Example user questions that should trigger this tool:**
- **Inputs the agent must supply** vs optional/defaulted:

---

## What happens after intake

1. Add a ~6-line entry to `config/integrations.yml` referencing the answers above.
2. The `IntegrationRegistry` builds the right ADK adapter at startup and validates it.
3. The tool is assigned to the declared agent(s) via `PrismAgent(integrations=[...])`.
4. It appears in `GET /api/v1/integrations` (with live health) and, later, in
   Settings → Tools & Capabilities (toggle on/off, per agent).
5. The golden example call becomes a registry integration test.
