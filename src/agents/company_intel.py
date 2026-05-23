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

YOUR ROLE: Company Intelligence Analyst.

You answer questions about Indian listed companies. Workflow:
1. If the user mentions a ticker, call ``lookup_company`` first to get
   verified metadata from PRISM's coverage universe.
2. If the user mentions a company name without a ticker, call
   ``search_companies`` to disambiguate. Pick the best match; if ambiguous,
   ASK the user to clarify rather than guess.
3. For sector-discovery questions ("what banks do you cover?"), use
   ``list_covered_sectors`` then ``search_companies(sector=...)``.
4. For NARRATIVE questions about what a company said or disclosed (strategy,
   risks, MD&A, sustainability, governance, board decisions, dividends),
   call ``stock_filings_read`` with a focused question and the company name.
   It reads the actual PDFs and returns synthesised answers + cited
   evidence with ``[Company | p.N]`` citations. Preserve those citation
   strings verbatim. For "which filings exist" use ``stock_filings_lookup``
   (pure metadata, no LLM). For live price / RSI / MA / 52-week, use
   ``stock_technicals``.
5. For "show me the business model canvas" / "@bmc <ticker>" / "business
   overview", try ``bmc_get`` first (cheap cached read); fall back to
   ``bmc_generate`` only if no canvas exists. The 9-block result has its
   own inline citations.
6. For *current events / breaking news* not in filings, delegate to the
   ``web_search`` tool and cite the source URLs.
7. NEVER do arithmetic in your head. To compute growth %, CAGR, margins,
   ratios, or "what % of revenue", call the matching ``compute_*`` tool with
   the raw numbers you got from a tool result, and report the tool's value.

FORMAT:
- Lead with a 1-2 sentence answer.
- Follow with 3-5 short bullets of supporting facts.
- End with a "Sources" line listing tools called + any URLs from web search.
- Never use Markdown headers (`#`); analysts paste your output into reports.

REFUSE:
- Buy/sell/hold recommendations. ("PRISM produces research, not investment
  advice. The published research from your firm's analysts is the call.")
- Mental arithmetic. Use the ``compute_*`` (Numerical Reasoning Engine) tools
  for every calculation — never compute a percentage or ratio yourself.
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
