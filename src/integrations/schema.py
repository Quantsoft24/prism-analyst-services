"""Declarative integration spec — one entry in ``config/integrations.yml``.

An *integration* is any agent-callable resource a teammate builds: an external
REST API (with or without an OpenAPI spec), an MCP server, an in-process Python
tool, or a specialist sub-agent. Each becomes one or more ADK tools at startup
via ``src/integrations/adapters.py``.

Design mirrors ``src/services/ingestion/registry.py`` (the other declarative,
versioned, validated-at-load registry). Secrets are referenced by **env var
name**, never inlined — nothing new ever leaks to git.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# The ADK-native adapter the registry uses to turn this entry into tool(s).
#   python  → FunctionTool(s) from an in-process callable / tool list
#   openapi → OpenAPIToolset from an OpenAPI v3 spec
#   mcp     → MCPToolset connected to an MCP server (streamable_http | sse | stdio)
#   agent   → AgentTool wrapping a specialist sub-agent
IntegrationSource = Literal["python", "openapi", "mcp", "agent"]


class AuthSpec(BaseModel):
    """How PRISM authenticates to the integration. Secret is read from ``env``
    at call time — the YAML only ever names the variable, never the value."""

    type: Literal["none", "api_key", "bearer"] = "none"
    env: str | None = None          # env var holding the secret (api_key / bearer)
    header: str | None = None       # header name for api_key (default X-API-Key)

    @model_validator(mode="after")
    def _need_env_for_secret(self) -> "AuthSpec":
        if self.type in ("api_key", "bearer") and not self.env:
            raise ValueError(f"auth.type={self.type} requires auth.env (the env var name)")
        return self


class IntegrationSpec(BaseModel):
    """One integration. Maps 1:1 to a registry entry and (later) a DB row."""

    name: str
    source: IntegrationSource
    description: str = ""
    enabled: bool = True
    # Free-form grouping for the UI / future per-agent assignment. Not enforced
    # yet (Part-A: no per-agent restriction now), but carried so it's ready.
    tags: list[str] = Field(default_factory=list)
    auth: AuthSpec = Field(default_factory=AuthSpec)
    # Source-specific settings (entrypoint / spec_url / transport+url / ...).
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_config(self) -> "IntegrationSpec":
        c = self.config
        if self.source in ("python", "agent") and not c.get("entrypoint"):
            raise ValueError(
                f"integration {self.name!r} (source={self.source}) requires "
                f"config.entrypoint in 'module.path:attribute' form"
            )
        if self.source == "openapi" and not (c.get("spec_url") or c.get("spec_str")):
            raise ValueError(
                f"integration {self.name!r} (openapi) requires config.spec_url or config.spec_str"
            )
        if self.source == "mcp":
            transport = c.get("transport")
            if transport not in ("streamable_http", "sse", "stdio"):
                raise ValueError(
                    f"integration {self.name!r} (mcp) requires config.transport "
                    f"in (streamable_http|sse|stdio), got {transport!r}"
                )
            if transport in ("streamable_http", "sse") and not c.get("url"):
                raise ValueError(
                    f"integration {self.name!r} (mcp/{transport}) requires config.url"
                )
            if transport == "stdio" and not c.get("command"):
                raise ValueError(
                    f"integration {self.name!r} (mcp/stdio) requires config.command"
                )
        return self
