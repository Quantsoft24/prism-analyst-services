"""Source adapters — turn an ``IntegrationSpec`` into ADK tool object(s).

One builder per source type. Each returns a list of objects valid in an ADK
``Agent.tools`` list — either ``BaseTool`` instances (FunctionTool / AgentTool)
or ``BaseToolset`` instances (OpenAPIToolset / MCPToolset), which the runner
expands at call time.

All ADK constructor signatures here were verified against google-adk 1.33.0.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
from typing import Any, Callable

import httpx

from src.integrations.schema import AuthSpec, IntegrationSpec

logger = logging.getLogger(__name__)


def _resolve_entrypoint(path: str) -> Any:
    """Import ``module.path:attribute`` and return the attribute."""
    module_name, _, attr = path.partition(":")
    if not module_name or not attr:
        raise ValueError(f"entrypoint must be 'module.path:attribute', got {path!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"{module_name!r} has no attribute {attr!r}") from exc


# ── Auth helpers (secret read from env at build/call time) ──────────────────


def _auth_headers(auth: AuthSpec) -> dict[str, str]:
    if auth.type == "none" or not auth.env:
        return {}
    secret = os.environ.get(auth.env, "")
    if not secret:
        logger.warning("auth env var %r is empty — sending no auth header", auth.env)
        return {}
    if auth.type == "bearer":
        return {"Authorization": f"Bearer {secret}"}
    if auth.type == "api_key":
        return {(auth.header or "X-API-Key"): secret}
    return {}


def _header_provider(auth: AuthSpec) -> Callable[[Any], dict[str, str]] | None:
    """OpenAPIToolset/MCP header provider — re-reads env each call so a rotated
    secret takes effect without a restart."""
    if auth.type == "none":
        return None

    def provider(_ctx: Any) -> dict[str, str]:
        return _auth_headers(auth)

    return provider


# ── Builders ────────────────────────────────────────────────────────────────


def build_python(spec: IntegrationSpec) -> list:
    """`python` → FunctionTool(s) from an in-process callable, a list of plain
    functions, or an already-built tool list (e.g. NRE_TOOLS)."""
    from google.adk.tools import FunctionTool

    obj = _resolve_entrypoint(spec.config["entrypoint"])
    if hasattr(obj, "to_list"):          # a lazy tool-list like NRE_TOOLS
        items = list(obj.to_list())
    elif isinstance(obj, (list, tuple)):
        items = list(obj)
    else:
        items = [obj]

    tools = []
    for item in items:
        # Plain (async) functions get wrapped; anything else is assumed to be an
        # already-built ADK tool/toolset and passed through.
        if inspect.isfunction(item) or inspect.iscoroutinefunction(item) or inspect.ismethod(item):
            tools.append(FunctionTool(func=item))
        elif callable(item) or hasattr(item, "name"):
            tools.append(item)
        else:
            raise ValueError(f"{spec.name}: entrypoint item {item!r} is not a function or tool")
    return tools


def build_agent(spec: IntegrationSpec) -> list:
    """`agent` → AgentTool wrapping a specialist sub-agent.

    Entrypoint is a zero-arg factory returning a ``PrismAgent`` (or ADK Agent).
    """
    from google.adk.tools.agent_tool import AgentTool

    factory = _resolve_entrypoint(spec.config["entrypoint"])
    decl = factory()
    adk_agent = decl.build() if hasattr(decl, "build") else decl
    return [AgentTool(agent=adk_agent)]


async def build_openapi(spec: IntegrationSpec) -> list:
    """`openapi` → OpenAPIToolset (one tool per operation; `tool_filter` restricts)."""
    from google.adk.tools.openapi_tool import OpenAPIToolset

    c = spec.config
    spec_str: str | None = c.get("spec_str")
    spec_type: str = c.get("spec_type", "json")
    if not spec_str:
        url = c["spec_url"]
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            spec_str = r.text
        if url.endswith((".yaml", ".yml")):
            spec_type = "yaml"

    return [
        OpenAPIToolset(
            spec_str=spec_str,
            spec_str_type=spec_type,  # type: ignore[arg-type]
            tool_filter=c.get("tool_filter"),
            header_provider=_header_provider(spec.auth),
        )
    ]


def build_mcp(spec: IntegrationSpec) -> list:
    """`mcp` → MCPToolset over Streamable HTTP / SSE / stdio.

    The connection is opened lazily by the runner (not at startup), so a down
    MCP server doesn't block boot.
    """
    from google.adk.tools.mcp_tool import (
        SseConnectionParams,
        StdioConnectionParams,
        StreamableHTTPConnectionParams,
    )
    from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset

    c = spec.config
    transport = c["transport"]
    headers = _auth_headers(spec.auth) or None

    if transport == "streamable_http":
        conn = StreamableHTTPConnectionParams(url=c["url"], headers=headers)
    elif transport == "sse":
        conn = SseConnectionParams(url=c["url"], headers=headers)
    elif transport == "stdio":
        from mcp import StdioServerParameters

        conn = StdioConnectionParams(
            server_params=StdioServerParameters(
                command=c["command"],
                args=c.get("args", []),
                env=c.get("env"),
            )
        )
    else:  # pragma: no cover — schema already validates
        raise ValueError(f"unknown mcp transport {transport!r}")

    return [MCPToolset(connection_params=conn, tool_filter=c.get("tool_filter"))]


async def build_tools_for_spec(spec: IntegrationSpec) -> list:
    """Dispatch to the right builder. Raises on bad config (caller records health)."""
    if spec.source == "python":
        return build_python(spec)
    if spec.source == "agent":
        return build_agent(spec)
    if spec.source == "openapi":
        return await build_openapi(spec)
    if spec.source == "mcp":
        return build_mcp(spec)
    raise ValueError(f"unknown integration source {spec.source!r}")
