"""Web search subagent — wraps ADK's built-in ``google_search`` tool.

Why this is its own agent (not a tool directly on ``company_intel``):

  ADK has a documented model-side limitation — ``google_search`` (a "builtin"
  tool tied to Gemini's search-grounding feature) cannot coexist with
  ``FunctionTool``s on the same agent. The Gemini API itself rejects the
  combination with ``Tool use with function calling is unsupported``.
  Reference: https://github.com/google/adk-python/issues/53 and the official
  workaround at https://adk.dev/grounding/google_search_grounding/.

  The standard pattern is to create a dedicated single-tool search agent
  and expose it via ``AgentTool`` so other agents can call it like a
  function. That's exactly what this module ships.

Why we use a direct Gemini model here (not the router):

  ADK's ``google_search_tool`` validates the model name against a hardcoded
  Gemini-only list (``adk/tools/google_search_tool.py:87``) and rejects our
  virtual ``prism-*`` names BEFORE the LiteLLM shim can route them. So this
  one subagent has to use a literal Gemini model.

  Trade-off: this single path bypasses the router and uses ``GEMINI_API_KEY``
  directly — no multi-key load balancing for web search. Acceptable because
  search is occasional. A future provider-agnostic web-search FunctionTool
  (Tavily/Brave) will return to the routed path.

Cost discipline:
  We deliberately use the Flash model here (cheap, fast). Search is a
  high-frequency / low-value-per-call workload.
"""

from __future__ import annotations

from src.agents.base import PrismAgent
from src.config import settings


WEB_SEARCH_INSTRUCTION = """\
You are PRISM's web search agent. You have ONE job: take the user's query,
run it through Google Search, and return a concise factual summary with
source URLs cited inline.

Rules:
- Always cite the source URL for every fact.
- Prefer authoritative sources: company IR pages, regulatory filings, major
  financial newspapers (Mint, Economic Times, Bloomberg, Reuters).
- If the search returns nothing relevant, say "No relevant results found."
- Do NOT speculate. Do NOT add commentary. Just summarize what the search
  results say, with sources.
- Keep the response under 200 words.
"""


def build_web_search_agent() -> PrismAgent:
    """Construct the web-search subagent declaration.

    Uses an EXPLICIT Gemini model (not the router) because google_search
    validates the model name — see module docstring.
    """
    # Import lazily so the module is safe to import without ADK installed.
    from google.adk.tools import google_search

    return PrismAgent(
        name="web_search",
        description=(
            "Performs a Google web search and returns a cited summary. "
            "Use for current events, recent news, and any question requiring "
            "fresh information beyond PRISM's internal coverage data."
        ),
        # IMPORTANT: explicit model name — google_search validates against
        # Gemini-only IDs. Routing via prism-* would 400.
        model=settings.AGENT_MODEL_FAST,
        instruction=WEB_SEARCH_INSTRUCTION,
        tools=[google_search],
        max_iterations=2,
    )
