"""Unit tests for ``ModelRouter`` — no live LLM calls.

We mock ``litellm.Router`` so these tests run on any machine without API
keys or network access. The actual LiteLLM behavior (cooldowns, routing
strategies) is LiteLLM's responsibility to test; here we verify our
adapter layer:
  * tier × api_keys → model_list expansion is correct
  * acquire() returns ADK LiteLlm with the right virtual model name
  * unknown tier raises a clear error
  * dispose() actually clears the singleton
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import settings
from src.services.model_router import (
    ModelRouter,
    _provider_of,
    dispose_router,
    get_router,
    init_router,
)
from src.services.model_router_config import (
    MODEL_PRICING_USD_PER_1M,
    TIER_CONFIGS,
    tier_from_virtual,
    virtual_model_name,
)


@pytest.fixture(autouse=True)
def _reset_router_singleton():
    """Each test starts with a fresh router state."""
    dispose_router()
    yield
    dispose_router()


# ── Config sanity ─────────────────────────────────────────────────────────


def test_tier_configs_define_expected_tiers():
    """The four canonical tiers exist. If you intentionally add/remove a tier,
    update this test — but only after updating the plan addendum first."""
    assert set(TIER_CONFIGS.keys()) == {"fast", "quality", "classify", "embedding"}


def test_every_tier_has_at_least_one_model():
    for tier, config in TIER_CONFIGS.items():
        assert config["models"], f"Tier {tier!r} has empty model list"


def test_every_model_has_a_pricing_entry():
    """Cost tracking depends on every model in any tier appearing in the
    pricing table (even if (0, 0) for free tier). A model missing here means
    AgentRun.cost_usd will silently report 0 for that model."""
    referenced = {m for cfg in TIER_CONFIGS.values() for m in cfg["models"]}
    missing = referenced - MODEL_PRICING_USD_PER_1M.keys()
    assert not missing, f"Models referenced in TIER_CONFIGS but missing from pricing: {missing}"


def test_virtual_model_name_roundtrip():
    for tier in TIER_CONFIGS:
        assert tier_from_virtual(virtual_model_name(tier)) == tier
    # Provider-fallback sub-groups resolve back to their base tier.
    assert tier_from_virtual("prism-fast-fb1") == "fast"
    assert tier_from_virtual("prism-quality-fb2") == "quality"
    assert tier_from_virtual("gemini-direct") is None
    assert tier_from_virtual("prism-not-a-real-tier") is None


# ── Router construction ──────────────────────────────────────────────────


def test_router_rejects_empty_api_keys():
    with pytest.raises(ValueError, match="at least one"):
        ModelRouter(api_keys=[])


def _count_models(tier: str, provider: str) -> int:
    """Number of models of ``provider`` in a tier's config."""
    return sum(1 for m in TIER_CONFIGS[tier]["models"] if _provider_of(m) == provider)


def test_gemini_only_collapses_to_single_group_per_tier(monkeypatch):
    """With NO openai/deepseek keys, each tier has ONE group named
    ``prism-<tier>`` containing only its gemini models — identical to the
    pre-multi-provider behavior (gemini models × K keys)."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "")
    router = ModelRouter(api_keys=["key-a", "key-b", "key-c"])
    model_list, fallbacks = router._build_model_list_and_fallbacks()

    # Only gemini models survive (openai/* + deepseek/* skipped → no key).
    expected_total = sum(_count_models(t, "gemini") * 3 for t in TIER_CONFIGS)
    assert len(model_list) == expected_total
    # No provider-fallback sub-groups exist when only one provider is present.
    assert not any(e["model_name"].endswith(("-fb1", "-fb2")) for e in model_list)
    for entry in model_list:
        assert entry["litellm_params"]["model"].startswith("gemini/")
        assert entry["litellm_params"]["api_key"] in {"key-a", "key-b", "key-c"}
    # Cross-tier net still wired (fast↔quality, classify→fast).
    fb = {k: v for d in fallbacks for k, v in d.items()}
    assert "prism-fast" in fb["prism-quality"]
    assert "prism-quality" in fb["prism-fast"]


def test_provider_priority_groups_and_failover_chain(monkeypatch):
    """With OpenAI + DeepSeek keys set, each generative tier splits into
    per-provider groups (openai = head, deepseek = fb1, gemini = fb2) chained
    as strict fallbacks: prism-<tier> → -fb1 → -fb2 → cross-tier head."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "ok-key")
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "ds-key")
    router = ModelRouter(api_keys=["g1", "g2", "g3"])  # 3 gemini keys
    model_list, fallbacks = router._build_model_list_and_fallbacks()

    # Per-provider key multiplicity: openai/deepseek = 1 key, gemini = 3.
    expected_total = sum(
        _count_models(t, "openai") * 1
        + _count_models(t, "deepseek") * 1
        + _count_models(t, "gemini") * 3
        for t in TIER_CONFIGS
    )
    assert len(model_list) == expected_total

    # The fast HEAD group is OpenAI-only; its first model is the config's first.
    fast_head = [e for e in model_list if e["model_name"] == "prism-fast"]
    assert fast_head and all(
        e["litellm_params"]["model"].startswith("openai/") for e in fast_head
    )
    assert fast_head[0]["litellm_params"]["model"] == TIER_CONFIGS["fast"]["models"][0]
    # fb1 = deepseek, fb2 = gemini.
    assert all(
        e["litellm_params"]["model"].startswith("deepseek/")
        for e in model_list if e["model_name"] == "prism-fast-fb1"
    )
    assert all(
        e["litellm_params"]["model"].startswith("gemini/")
        for e in model_list if e["model_name"] == "prism-fast-fb2"
    )

    # Strict failover chain for the head, then the cross-tier net.
    fb = {k: v for d in fallbacks for k, v in d.items()}
    assert fb["prism-fast"][:2] == ["prism-fast-fb1", "prism-fast-fb2"]
    assert "prism-quality" in fb["prism-fast"]  # cross-tier last resort


def test_embedding_tier_never_falls_back(monkeypatch):
    """Embedding stays single-model, single-group, and is in NO fallback chain
    (mixing embedding models corrupts the vector space)."""
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "ok-key")
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "ds-key")
    router = ModelRouter(api_keys=["g1"])
    _model_list, fallbacks = router._build_model_list_and_fallbacks()
    fb = {k: v for d in fallbacks for k, v in d.items()}
    assert "prism-embedding" not in fb
    # And nothing falls back INTO embedding.
    assert all("prism-embedding" not in targets for targets in fb.values())


# ── Singleton lifecycle + acquire ────────────────────────────────────────


@patch("litellm.Router")
def test_init_router_is_idempotent(mock_router_cls):
    """Calling init_router twice returns the same instance — important for
    test runners that may re-run lifespan + we don't want compounded shims."""
    mock_router_cls.return_value = MagicMock()
    a = init_router(["k1"])
    b = init_router(["k2"])  # second key list is ignored — already built
    assert a is b


@patch("litellm.Router")
def test_get_router_before_init_raises(_mock_router_cls):
    with pytest.raises(RuntimeError, match="not initialized"):
        get_router()


@patch("litellm.Router")
def test_acquire_returns_litellm_with_virtual_name(mock_router_cls):
    mock_router_cls.return_value = MagicMock()
    router = init_router(["k1", "k2"])

    # ``acquire`` lazy-imports ADK's LiteLlm. Patch it at the import site.
    with patch("google.adk.models.lite_llm.LiteLlm") as mock_litellm_cls:
        sentinel = MagicMock()
        mock_litellm_cls.return_value = sentinel
        result = router.acquire("fast")

    assert result is sentinel
    mock_litellm_cls.assert_called_once_with(model="prism-fast")


@patch("litellm.Router")
def test_acquire_unknown_tier_raises(mock_router_cls):
    mock_router_cls.return_value = MagicMock()
    router = init_router(["k1"])
    with pytest.raises(KeyError, match="Unknown model tier"):
        router.acquire("nonexistent")  # type: ignore[arg-type]


@patch("litellm.Router")
def test_health_snapshot_omits_api_keys(mock_router_cls):
    """Defense in depth — even the debug endpoint must never leak keys."""
    mock_router_cls.return_value = MagicMock()
    router = init_router(["super-secret-key"])
    snapshot = router.health()
    serialized = str(snapshot)
    assert "super-secret-key" not in serialized
    assert snapshot["ready"] is True
    assert snapshot["api_keys_count"] == 1
    assert set(snapshot["tiers"]) == set(TIER_CONFIGS.keys())


@patch("litellm.Router")
def test_dispose_clears_singleton(mock_router_cls):
    mock_router_cls.return_value = MagicMock()
    init_router(["k1"])
    dispose_router()
    with pytest.raises(RuntimeError, match="not initialized"):
        get_router()
