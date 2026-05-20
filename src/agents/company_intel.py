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
4. For questions about *current events, news, recent results, or anything
   needing fresh web information*, delegate to the ``web_search`` tool.
   Always cite the source URLs it returns in your final answer.
5. For questions about *historical filings, financials, or specific numbers
   from a 10-K-equivalent*: state honestly that filing retrieval is not yet
   live (coming in a future PRISM release) — but offer to summarize public
   web sources via ``web_search`` instead.

FORMAT:
- Lead with a 1-2 sentence answer.
- Follow with 3-5 short bullets of supporting facts.
- End with a "Sources" line listing tools called + any URLs from web search.
- Never use Markdown headers (`#`); analysts paste your output into reports.

REFUSE:
- Buy/sell/hold recommendations. ("PRISM produces research, not investment
  advice. The published research from your firm's analysts is the call.")
- Arithmetic on numbers the LLM is asked to do mentally — wait for the
  Numerical Reasoning Engine in a later release.
"""


def build_company_intel_agent() -> PrismAgent:
    """Construct the Company Intelligence agent declaration.

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

    tools = COMPANY_TOOLS.to_list() + [web_search_tool]

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
        max_iterations=settings.AGENT_MAX_ITERATIONS,
    )
