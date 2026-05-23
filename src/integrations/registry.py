"""Integration registry — loads ``config/integrations.yml``, builds the ADK
tools for each enabled entry at startup, and exposes them to agents.

Mirrors two existing patterns:
  * ``src/services/ingestion/registry.py`` — declarative YAML, validated on load,
    with a documented DB-migration seam (``load_registry`` is the only thing that
    changes when Phase 2 moves to a ``firm_integrations`` table).
  * ``src/services/model_router.py`` — a process-singleton built once in the
    FastAPI lifespan and disposed on shutdown.

Failures are isolated per integration: a bad entry is recorded as unhealthy and
surfaced via ``GET /api/v1/integrations`` — it never crashes startup or silently
disappears (Part-A: transparent failures).
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from src.integrations.adapters import build_tools_for_spec
from src.integrations.schema import IntegrationSpec

logger = logging.getLogger(__name__)


class IntegrationHealth(dict):
    """Plain dict subclass for clarity at call sites; serialises as-is."""


class IntegrationRegistry:
    """In-memory registry: specs → built ADK tools + per-integration health."""

    def __init__(self, specs: list[IntegrationSpec]) -> None:
        self._specs = specs
        self._tools: list = []
        self._tools_by_name: dict[str, list] = {}
        self._health: list[dict] = []
        self._built = False

    async def build(self) -> None:
        """Build tools for every enabled spec. Idempotent. Per-entry try/except so
        one broken integration can't take down the rest (or the app)."""
        if self._built:
            return
        for spec in self._specs:
            if not spec.enabled:
                self._health.append(
                    {"name": spec.name, "source": spec.source, "enabled": False,
                     "status": "disabled", "tool_count": 0, "error": None,
                     "description": spec.description, "tags": spec.tags}
                )
                continue
            try:
                tools = await build_tools_for_spec(spec)
                self._tools.extend(tools)
                self._tools_by_name[spec.name] = tools
                self._health.append(
                    {"name": spec.name, "source": spec.source, "enabled": True,
                     "status": "ok", "tool_count": len(tools), "error": None,
                     "description": spec.description, "tags": spec.tags}
                )
                logger.info("Integration %r (%s): %d tool(s) loaded.", spec.name, spec.source, len(tools))
            except Exception as exc:  # noqa: BLE001 — isolate per-integration failure
                self._health.append(
                    {"name": spec.name, "source": spec.source, "enabled": True,
                     "status": "error", "tool_count": 0, "error": str(exc),
                     "description": spec.description, "tags": spec.tags}
                )
                logger.warning("Integration %r (%s) failed to load: %s", spec.name, spec.source, exc)
        self._built = True

    def tools(self) -> list:
        """All successfully-built tools, for merging into an agent's tool list."""
        return list(self._tools)

    def tools_for(self, names: list[str]) -> list:
        """Tools for the named integrations only (skips unknown names)."""
        out: list = []
        for n in names:
            out.extend(self._tools_by_name.get(n, []))
        return out

    def names(self) -> list[str]:
        """Names of integrations that successfully built tools (the toggle-able set)."""
        return list(self._tools_by_name.keys())

    def health(self) -> list[dict]:
        """Per-integration status — drives GET /integrations and the Settings UI."""
        return list(self._health)


# ── Loading ──────────────────────────────────────────────────────────────────


def load_specs(path: str | Path) -> list[IntegrationSpec]:
    """Parse the YAML registry into validated ``IntegrationSpec``s.

    Missing file → empty list (integrations are optional). Malformed entries
    raise ``ValueError`` via pydantic validation.
    """
    p = Path(path)
    if not p.exists():
        logger.info("No integrations registry at %s — starting with none.", p)
        return []
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"Registry {p} must be a YAML list of entries, got {type(raw).__name__}")
    return [IntegrationSpec(**entry) for entry in raw]


# ── Process singleton (built in the FastAPI lifespan) ────────────────────────

_registry: IntegrationRegistry | None = None


async def init_registry(path: str | Path) -> IntegrationRegistry:
    """Load + build the registry. Called once at startup."""
    global _registry
    specs = load_specs(path)
    _registry = IntegrationRegistry(specs)
    await _registry.build()
    return _registry


def get_registry() -> IntegrationRegistry | None:
    """The built registry, or None if not initialised (e.g. tests)."""
    return _registry


def dispose_registry() -> None:
    global _registry
    _registry = None
