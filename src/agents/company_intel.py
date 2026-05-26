"""Company Intelligence Agent — PRISM's first real agent.

Job:
  Given a free-form analyst question about an Indian listed company, identify
  the company, gather verified metadata from PRISM's coverage universe, and
  produce a concise, cited answer. For current events / news, delegate to
  the ``web_search`` subagent (Google Search grounding).

Scope (intentionally narrow for the first slice):
  * Single-turn or short multi-turn — no long-running planning loops yet.
  * Tools: company lookup/search/sectors + ``web_search`` subagent
    (wrapped as an AgentTool — see why below).
  * NO filing retrieval yet (Slice 5+) — agent gracefully degrades.
  * NO numerical reasoning yet (Slice 4 NRE) — agent refuses arithmetic.

Why ``web_search`` is a wrapped sub-agent and not a tool directly:
  ADK's built-in ``google_search`` cannot coexist with ``FunctionTool``s on
  the same agent — Gemini's API rejects mixed tool kinds with "Tool use with
  function calling is unsupported" (see google/adk-python issue #53). The
  documented workaround at https://adk.dev/grounding/google_search_grounding/
  is the ``AgentTool`` pattern: a dedicated single-tool search agent exposed
  as a tool to the main orchestrator. See ``src/agents/web_search.py``.

This agent is the template every future agent (BMC, modelling, writer) copies.
Keep its prompt + tool list tight; resist scope creep here.
"""

from __future__ import annotations

from src.agents.base import FINANCE_DOMAIN_RULES, PrismAgent
from src.agents.web_search import build_web_search_agent
from src.config import settings
from src.tools.company_tools import COMPANY_TOOLS
from src.tools.nre_tools import NRE_TOOLS

COMPANY_INTEL_INSTRUCTION = f"""\
{FINANCE_DOMAIN_RULES}

YOUR ROLE: Company Intelligence Analyst for Indian listed companies.

# CORE CONTRACT — verify before you answer

You are NOT a general-knowledge LLM here. You are a research analyst whose
only valid sources are the tools listed below. Three hard rules:

1. **Every factual claim MUST come from a tool result you observed THIS
   turn.** If you cannot produce a tool call that surfaced the fact, you
   do not have the fact — say "I don't have that information" or ask the
   user to clarify. Do NOT fall back on your training data.

2. **When a lookup misses, surface the alternatives instead of guessing.**
   `lookup_company` and `search_companies` return a `suggestions` array
   when the query was likely a typo or partial name. When that array is
   non-empty, ask the user "Did you mean <X> or <Y>?" rather than picking
   one yourself or proceeding with the typo'd term.

3. **Read every tool response's `ok` / `error` / `next_action` fields.**
   A tool that returns `ok=False` did NOT succeed; ignoring this is the
   single biggest source of hallucinated answers. Follow `next_action`:
     - `ask_user_to_retry_later` → tell the user the source is briefly
       unavailable. STOP. Do not invent results.
     - `try_alternate_tool`      → reach for a different tool that could
       answer (e.g. if `stock_filings_read` 5xx'd, try
       `stock_filings_lookup` for the metadata at least). If no alternate
       fits, apologize and STOP.
     - `ask_user_to_clarify`     → the user's input was ambiguous; ask
       a tight follow-up question and STOP.
     - `give_up_gracefully`      → a clean dead-end; tell the user and STOP.

# TOOL CATALOGUE — pick the RIGHT one

Use this decision table before calling a tool. Re-read it on every turn.

  Question shape                                          | Tool first to try
  ------------------------------------------------------- | --------------------------
  Ticker known (3-6 letters, all caps)                    | `lookup_company`
  Company name only, possibly partial / misspelled         | `search_companies`
  "What sectors do you cover?"                            | `list_covered_sectors`
  "Filter banks / IT companies / pharma"                   | `search_companies(sector=…)`
  "What did X SAY / DISCLOSE / ANNOUNCE in their filings"  | `stock_filings_read`
  "Which filings did X submit / how many"                  | `stock_filings_lookup`
  "Current price / RSI / 52-week / MA"                    | `stock_technicals`
  "Show / explain / refresh the business model canvas"     | `bmc_get`, then `bmc_generate`
  "Drill into the [block] of the canvas"                   | `bmc_block_chat`
  "How has X's BMC changed FY24 → FY26"                    | `bmc_diff`
  Current events / news NOT in filings                     | `web_search`
  Any % / ratio / growth / CAGR / margin                   | `compute_*` (NEVER do it yourself)

When in doubt between `stock_filings_read` and `stock_filings_lookup`:
LOOKUP returns metadata only (which filings exist) — fast, free of LLM
calls. READ actually opens PDFs and synthesizes — slow, expensive, but
returns the actual answer.

# DATA-FRESHNESS RULE

Every quotable fact must carry a date. The tools tell you when their data
is from:
  - filings tools return `selected_filings[].announcement_dt` and a
    top-level `data_freshness` — quote the date in your answer.
  - `stock_technicals` is "live" — say so when quoting prices.
  - `web_search` results are dated — preserve the year/month in citations.

If you cannot date a fact, do not present it as current.

# REFUSALS

- Buy / sell / hold / accumulate / target-price recommendations. ("PRISM
  produces research, not investment advice. Your firm's analysts publish
  the call; my job is to ground their work.")
- Mental arithmetic — every percentage, ratio, growth, CAGR, margin call
  goes through a `compute_*` tool. No exceptions.
- Predictions / forecasts beyond what a cited filing or analyst note
  explicitly states.

# OUTPUT FORMAT

- Lead with a 1-2 sentence answer.
- Follow with 3-5 short bullets of supporting facts, each ending with a
  citation in the format ``[<company-or-source> | <date or page>]``.
- Preserve `[Company | p.N]` citations from filings READ results verbatim.
- End with a "Sources" line listing the tools you called + any URLs.
- No Markdown headers (`#`) — analysts paste into reports.

# STRUCTURED METADATA TAIL (optional, but PREFERRED)

If you have a clear sense of confidence + data freshness for the answer,
END your response with EXACTLY this fenced block (the runner parses it
to power UI citation chips):

  <answer_meta>{{
    "confidence": "high" | "medium" | "low",
    "data_freshness": "<ISO date or fiscal label, e.g. '2026-03-31' or 'FY24'>",
    "citations": [
      {{"label": "Reliance Q4 FY24, p.12", "source_kind": "filing", "as_of": "2024-04-30"}}
    ]
  }}</answer_meta>

The block is optional — if you omit it the answer still renders. Include
it whenever you have non-trivial confidence to communicate. Pure prose
goes BEFORE the tag; nothing after the closing tag.
"""


def build_company_intel_agent(integrations: str | list[str] | None = "*") -> PrismAgent:
    """Construct the Company Intelligence agent declaration.

    Args:
        integrations: which registered integrations to attach — ``"*"`` (all,
            the default) or a list of integration names. Callers with a firm
            context (the chat router) pass the firm's *enabled* names so a
            firm-disabled tool isn't offered to the LLM. See ``firm_state``.

    Lazily-bound: actual ADK Agent object is built when ``.build()`` is
    called, so this function is safe to call in any context (tests, CLI,
    HTTP handler).

    Tool composition:
      * Company FunctionTools (lookup / search / sectors) — go through the
        router via ``model_tier="fast"`` on this orchestrator.
      * ``web_search`` AgentTool — wraps a dedicated single-tool subagent
        that uses ``google_search`` directly. The subagent runs on a
        literal Gemini model (bypassing the router) because Google's
        search-grounding feature requires it.
    """
    # Lazy import — only needed when we actually build (keeps module
    # importable without ADK installed for tests / lint).
    from google.adk.tools.agent_tool import AgentTool

    # Build the web_search subagent and wrap it as a callable tool.
    web_search_agent_decl = build_web_search_agent()
    web_search_adk_agent = web_search_agent_decl.build()
    web_search_tool = AgentTool(agent=web_search_adk_agent)

    # Built-in tools = catalog-backed company lookups + deterministic NRE math
    # + web search. RAG/filings and BMC are now external services wired
    # through the integration registry (config/integrations.yml) — picked up
    # via `integrations="*"` below so the agent has stock_filings_read,
    # stock_filings_lookup, stock_technicals, and the 6 bmc_* tools.
    tools = (
        COMPANY_TOOLS.to_list()
        + NRE_TOOLS.to_list()
        + [web_search_tool]
    )

    return PrismAgent(
        name="company_intel",
        description=(
            "Answers questions about Indian listed companies using verified "
            "metadata + live Google web search via a subagent."
        ),
        # Tier-based routing: ``ModelRouter`` resolves to one of the free-tier
        # Gemini Flash / Gemma 4 deployments at call time with multi-key fallback.
        # See ``services/model_router_config.py`` for the chain.
        model_tier="fast",
        instruction=COMPANY_INTEL_INSTRUCTION,
        tools=tools,
        integrations=integrations,  # "*" = all; chat passes the firm's enabled names
        max_iterations=settings.AGENT_MAX_ITERATIONS,
    )
