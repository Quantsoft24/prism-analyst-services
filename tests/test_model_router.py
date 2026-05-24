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

from src.services.model_router import (
    ModelRouter,
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
    assert tier_from_virtual("gemini-direct") is None
    assert tier_from_virtual("prism-not-a-real-tier") is None


# ── Router construction ──────────────────────────────────────────────────


def test_router_rejects_empty_api_keys():
    with pytest.raises(ValueError, match="at least one"):
        ModelRouter(api_keys=[])


def test_build_model_list_expands_tier_by_keys():
    """N models × K keys = N*K deployments per tier."""
    router = ModelRouter(api_keys=["key-a", "key-b", "key-c"])
    # Call internal builder directly — no need to spin up litellm.Router.
    model_list, fallbacks = router._build_model_list_and_fallbacks()

    # Each tier should contribute (num_models × 3 keys) entries.
    expected_total = sum(len(cfg["models"]) * 3 for cfg in TIER_CONFIGS.values())
    assert len(model_list) == expected_total

    # Spot-check: every entry has the right shape.
    for entry in model_list:
        assert entry["model_name"].startswith("prism-")
        assert "litellm_params" in entry
        assert entry["litellm_params"]["api_key"] in {"key-a", "key-b", "key-c"}
        assert entry["rpm"] > 0
        assert entry["tpm"] > 0

    # Fallbacks: quality should cascade to fast.
    fallback_targets = {k: v for d in fallbacks for k, v in d.items()}
    assert "prism-fast" in fallback_targets["prism-quality"]


def test_build_model_list_preserves_tier_chain_order():
    """First entry for ``prism-fast`` must be the first model in the config —
    LiteLLM's usage-based-routing-v2 honors order on ties."""
    router = ModelRouter(api_keys=["k"])
    model_list, _ = router._build_model_list_and_fallbacks()

    fast_entries = [e for e in model_list if e["model_name"] == "prism-fast"]
    assert fast_entries, "Expected at least one prism-fast deployment"
    first_model = fast_entries[0]["litellm_params"]["model"]
    expected_first = TIER_CONFIGS["fast"]["models"][0]
    assert first_model == expected_first


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
