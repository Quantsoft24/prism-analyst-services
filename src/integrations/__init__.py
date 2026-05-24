"""PRISM integration framework — plug any agent resource in via config.

A teammate adds a tool by appending an entry to ``config/integrations.yml``
(see ``docs/INTEGRATION_INTAKE.md``); the registry builds the right ADK adapter
at startup and exposes it to agents. No agent code changes.

Supported sources (ADK-native): ``python`` (FunctionTool), ``openapi``
(OpenAPIToolset), ``mcp`` (MCPToolset), ``agent`` (AgentTool).
"""

from src.integrations.registry import (
    IntegrationRegistry,
    dispose_registry,
    get_registry,
    init_registry,
    load_specs,
)
from src.integrations.schema import IntegrationSpec

__all__ = [
    "IntegrationRegistry",
    "IntegrationSpec",
    "init_registry",
    "get_registry",
    "dispose_registry",
    "load_specs",
]
