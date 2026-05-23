"""Tests for the universal integration framework (schema, loader, registry,
adapters). No live LLM/HTTP calls — the python adapter wraps functions offline."""

from __future__ import annotations

import asyncio

import pytest

from src.integrations.registry import IntegrationRegistry, load_specs
from src.integrations.schema import IntegrationSpec


def _write(tmp_path, content: str):
    p = tmp_path / "integrations.yml"
    p.write_text(content, encoding="utf-8")
    return p


# ── Schema validation ────────────────────────────────────────────────────────


def test_python_requires_entrypoint():
    with pytest.raises(ValueError, match="entrypoint"):
        IntegrationSpec(name="x", source="python", config={})


def test_openapi_requires_spec():
    with pytest.raises(ValueError, match="spec_url or config.spec_str"):
        IntegrationSpec(name="x", source="openapi", config={})


def test_mcp_requires_valid_transport():
    with pytest.raises(ValueError, match="transport"):
        IntegrationSpec(name="x", source="mcp", config={"transport": "carrier-pigeon"})


def test_mcp_streamable_http_requires_url():
    with pytest.raises(ValueError, match="requires config.url"):
        IntegrationSpec(name="x", source="mcp", config={"transport": "streamable_http"})


def test_auth_secret_requires_env():
    with pytest.raises(ValueError, match="requires auth.env"):
        IntegrationSpec(name="x", source="python", config={"entrypoint": "m:f"},
                        auth={"type": "bearer"})


def test_valid_spec_ok():
    spec = IntegrationSpec(
        name="stock-chat", source="python",
        config={"entrypoint": "src.integrations.tools.stock_chat:STOCK_CHAT_TOOLS"},
    )
    assert spec.enabled is True  # Part-A default ON
    assert spec.source == "python"


# ── Loader ───────────────────────────────────────────────────────────────────


def test_load_specs_missing_file_is_empty(tmp_path):
    assert load_specs(tmp_path / "nope.yml") == []


def test_load_specs_non_list_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be a YAML list"):
        load_specs(_write(tmp_path, "name: x\nsource: python"))


def test_shipped_registry_is_valid():
    """config/integrations.yml must always parse — guards a malformed PR edit."""
    specs = load_specs("config/integrations.yml")
    assert any(s.name == "stock-chat" for s in specs)
    for s in specs:
        assert isinstance(s, IntegrationSpec)


# ── Registry build (offline) ─────────────────────────────────────────────────


def test_registry_builds_stock_chat_tools():
    """The python adapter wraps the 3 stock-chat functions into FunctionTools
    without any network call."""
    spec = IntegrationSpec(
        name="stock-chat", source="python",
        config={"entrypoint": "src.integrations.tools.stock_chat:STOCK_CHAT_TOOLS"},
    )
    reg = IntegrationRegistry([spec])
    asyncio.run(reg.build())

    health = reg.health()
    assert len(health) == 1
    assert health[0]["status"] == "ok"
    assert health[0]["tool_count"] == 3
    assert len(reg.tools()) == 3
    assert len(reg.tools_for(["stock-chat"])) == 3
    assert reg.tools_for(["does-not-exist"]) == []


def test_registry_disabled_spec_skipped():
    spec = IntegrationSpec(
        name="off", source="python", enabled=False,
        config={"entrypoint": "src.integrations.tools.stock_chat:STOCK_CHAT_TOOLS"},
    )
    reg = IntegrationRegistry([spec])
    asyncio.run(reg.build())
    assert reg.health()[0]["status"] == "disabled"
    assert reg.tools() == []


def test_registry_isolates_failures():
    """A broken integration is recorded as 'error' and does NOT crash the build
    or take down healthy ones (Part-A: transparent failures, graceful degradation)."""
    good = IntegrationSpec(
        name="good", source="python",
        config={"entrypoint": "src.integrations.tools.stock_chat:STOCK_CHAT_TOOLS"},
    )
    bad = IntegrationSpec(
        name="bad", source="python",
        config={"entrypoint": "src.integrations.tools.does_not_exist:nope"},
    )
    reg = IntegrationRegistry([good, bad])
    asyncio.run(reg.build())

    statuses = {h["name"]: h["status"] for h in reg.health()}
    assert statuses == {"good": "ok", "bad": "error"}
    bad_entry = next(h for h in reg.health() if h["name"] == "bad")
    assert bad_entry["error"]
    assert len(reg.tools()) == 3  # only the good one's tools
