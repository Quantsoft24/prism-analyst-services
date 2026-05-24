"""Thin abstraction over Google ADK ``Agent``.

Why an abstraction layer?
  ADK is the youngest of the major agent frameworks (per industry comparison,
  LangGraph has more battle-testing in finance — BlackRock, JPMorgan, Klarna).
  Our `final_docs/02_ARCHITECTURE_AND_STACK.md` commits us to ADK for cost
  + multi-model routing, but with the explicit caveat that we'd swap if
  production pain emerges. This module is the swap point.

  Application code imports ``PrismAgent`` and calls ``.run()``; it never
  touches ``google.adk.*`` directly. To swap frameworks later we rewrite
  THIS file and ``services/agent_runner.py`` — nothing else.

Design rules:
  * Stateless agent instances — sessions live in the Runner / SessionService.
  * Tools are passed in explicitly, not magically discovered.
  * Configuration (model, instruction, max-iterations) declared at construction
    time so a given agent's behavior is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.config import settings

if TYPE_CHECKING:
    # Avoid importing ADK at module load time so the rest of the app can
    # boot even if google-adk isn't installed yet (tests, CI lint, etc.).
    from google.adk.agents import Agent as AdkAgent


@dataclass(slots=True)
class PrismAgent:
    """Declarative wrapper for one agent.

    Attributes
    ----------
    name:
        Short identifier; used in the ``agent_runs.agent_name`` column.
    model_tier:
        Preferred model tier — ``"fast"`` | ``"quality"`` | ``"classify"``.
        Resolved at ``.build()`` time through ``ModelRouter`` so the agent
        gets a load-balanced + fallback-aware model. **Use this** for almost
        every agent. Slice 4 plan addendum has the rationale.
    model:
        Explicit Gemini model ID — only set this to override the tier router
        for a one-off (e.g., debugging, narrow regression test). When unset,
        ``model_tier`` wins. When both are set, ``model`` takes precedence.
    instruction:
        System prompt. **Keep finance-specific rules here** — e.g. "never do
        arithmetic, call the NRE tool", "always cite filings if available",
        "Indian fiscal year ends 31 March".
    tools:
        ADK-compatible tool objects (``FunctionTool`` instances, ``google_search``,
        etc.). Empty list = a pure-LLM agent (no tools).
    integrations:
        Which registered integrations (from ``config/integrations.yml``) this
        agent gets, resolved through the registry at ``.build()`` time:
          * ``None``  → no integration tools (default — keeps tightly-scoped
            sub-agents like the BMC block agents clean).
          * ``"*"``   → ALL enabled integration tools (firm-wide; the main
            user-facing agent uses this — Part-A: no per-agent restriction).
          * ``[names]`` → only those named integrations (future per-agent control).
    description:
        Short human-readable summary surfaced in the OpenAPI spec.
    max_iterations:
        Hard cap on tool-calling loop depth — fail-safe against runaway agents.
    """

    name: str
    instruction: str
    model_tier: str = "fast"
    model: str | None = None  # explicit override; None = use model_tier via router
    tools: list[Any] = field(default_factory=list)
    integrations: str | list[str] | None = None
    description: str = ""
    max_iterations: int = 10

    def build(self) -> "AdkAgent":
        """Construct the underlying ADK Agent instance.

        Resolution order for the LLM:
          1. ``self.model`` (explicit string override)            — debugging path
          2. ``ModelRouter.acquire(self.model_tier)``              — production path
          3. ``settings.AGENT_MODEL_FAST`` as last-resort fallback — only when the
             router is disabled (``MODEL_ROUTER_ENABLED=False``)

        We import inside the function so the module doesn't require ADK at
        import time — important for tests that don't exercise this path.
        """
        from google.adk.agents import Agent as AdkAgent

        model_arg = self._resolve_model()
        tools = list(self.tools) + self._integration_tools()

        return AdkAgent(
            name=self.name,
            model=model_arg,
            instruction=self.instruction,
            description=self.description or self.name,
            tools=tools,
        )

    def _integration_tools(self) -> list[Any]:
        """Resolve ``self.integrations`` against the registry (see field docs).
        Registry-agnostic at import time; returns [] if the registry isn't built
        (tests, registry disabled) so agents always build."""
        if self.integrations is None:
            return []
        from src.integrations import get_registry

        registry = get_registry()
        if registry is None:
            return []
        if self.integrations == "*":
            return registry.tools()
        if isinstance(self.integrations, list):
            return registry.tools_for(self.integrations)
        return []

    def _resolve_model(self) -> Any:
        """Pick the model object/string for this agent — see ``build()`` docstring."""
        if self.model:
            # Explicit override wins — used for narrow regression tests.
            return self.model
        if settings.MODEL_ROUTER_ENABLED:
            # Lazy import to keep this module router-agnostic at import time.
            from src.services.model_router import get_router

            return get_router().acquire(self.model_tier)  # type: ignore[arg-type]
        # Router disabled — fall back to the single configured model.
        return (
            settings.AGENT_MODEL_QUALITY
            if self.model_tier == "quality"
            else settings.AGENT_MODEL_FAST
        )


# ── Standard instruction fragments ──────────────────────────────────────────
# Composed into every PRISM agent's system prompt so behavior is consistent.
# Edit cautiously — these change the behavior of every agent at once.

FINANCE_DOMAIN_RULES = """\
You are an AI research assistant for Indian equity analysts. Rules:
- Fiscal year in India ends 31 March. "FY24" = year ending March 2024.
- All amounts in Indian Rupees (₹) unless explicitly stated otherwise. Use
  "lakh" (₹100,000) and "crore" (₹10,000,000) where idiomatic.
- Indian markets: NSE and BSE. Tickers are uppercase (TCS, RELIANCE).
- Cite a source for every fact. If you don't have a source, say "I don't
  know" — never invent numbers, dates, or names.
- Never do arithmetic yourself. If math is required, call a tool that does
  deterministic computation, or state "computation tool not yet available"
  and stop.
- Be concise. Analysts read fast.
"""
