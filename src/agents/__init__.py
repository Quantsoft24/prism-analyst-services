"""PRISM Agents — declarative agent definitions on top of Google ADK.

Application code:

    from src.agents import build_company_intel_agent

    agent = build_company_intel_agent()
    # ... hand to services.agent_runner.AgentRunner

Adding a new agent:
  1. Create ``src/agents/<name>.py`` with a ``build_*_agent() -> PrismAgent``.
  2. Re-export here.
  3. Register the agent's invocation route in ``src/routers/chat.py``
     (or directly call from another agent as a sub-agent tool).
"""

from src.agents.base import FINANCE_DOMAIN_RULES, PrismAgent
from src.agents.company_intel import build_company_intel_agent
from src.agents.web_search import build_web_search_agent

__all__ = [
    "PrismAgent",
    "FINANCE_DOMAIN_RULES",
    "build_company_intel_agent",
    "build_web_search_agent",
]
