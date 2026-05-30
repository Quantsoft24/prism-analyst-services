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
    names = {s.name for s in specs}
    assert {"stock-chat", "bmc", "prism-financials", "prism-news"}.issubset(names)
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


def test_registry_builds_bmc_tools():
    """The new BMC integration (external service) exposes 6 typed tools."""
    spec = IntegrationSpec(
        name="bmc", source="python",
        config={"entrypoint": "src.integrations.tools.bmc:BMC_TOOLS"},
    )
    reg = IntegrationRegistry([spec])
    asyncio.run(reg.build())

    health = reg.health()
    assert health[0]["status"] == "ok"
    assert health[0]["tool_count"] == 6
    assert len(reg.tools_for(["bmc"])) == 6
    # Sanity: each is a FunctionTool with one of the canonical names.
    tool_names = {getattr(t, "name", None) for t in reg.tools()}
    expected = {"bmc_get", "bmc_generate", "bmc_library", "bmc_get_version",
                "bmc_block_chat", "bmc_diff"}
    assert expected == tool_names


def test_registry_builds_prism_financials_tool():
    """The prism-financials integration exposes one typed tool (financials_query)."""
    spec = IntegrationSpec(
        name="prism-financials", source="python",
        config={"entrypoint": "src.integrations.tools.prism_financials:PRISM_FINANCIALS_TOOLS"},
    )
    reg = IntegrationRegistry([spec])
    asyncio.run(reg.build())

    health = reg.health()
    assert health[0]["status"] == "ok"
    assert health[0]["tool_count"] == 1
    tool_names = {getattr(t, "name", None) for t in reg.tools()}
    assert tool_names == {"financials_query"}


def test_registry_builds_prism_news_tools():
    """The prism-news integration exposes 4 typed tools (news_sentiment,
    news_trending, news_search, news_compare)."""
    spec = IntegrationSpec(
        name="prism-news", source="python",
        config={"entrypoint": "src.integrations.tools.prism_news:PRISM_NEWS_TOOLS"},
    )
    reg = IntegrationRegistry([spec])
    asyncio.run(reg.build())

    health = reg.health()
    assert health[0]["status"] == "ok"
    assert health[0]["tool_count"] == 4
    tool_names = {getattr(t, "name", None) for t in reg.tools()}
    expected = {"news_sentiment", "news_trending", "news_search", "news_compare"}
    assert expected == tool_names


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
