"""ModelRouter — tier-based multi-key fallback router for LLM calls.

**What it does.**
  Agents declare a ``model_tier`` ("fast", "quality", ...). ``ModelRouter``
  expands that tier into a list of (model × api_key) deployments using
  ``TIER_CONFIGS``, hands the list to ``litellm.Router``, and returns an
  ADK-compatible ``LiteLlm`` model whose virtual model name (e.g.
  ``"prism-fast"``) is dispatched through the router at call time.

  LiteLLM Router handles:
    * Load-balancing across deployments (``usage-based-routing-v2``)
    * 429-aware cooldown (deployment goes idle for ``cooldown_time``)
    * Fallback to the next deployment if all in the chain cool down
    * RPM / TPM client-side caps so we don't hammer the provider 429

**How ADK calls it.**
  ADK's ``LiteLlm`` wrapper calls ``litellm.acompletion(model=...)``. We
  install a thin shim at startup that intercepts calls with ``model``
  beginning with ``prism-`` and routes them through our ``Router``.
  Non-prism model names fall through to the original LiteLLM behavior.

  This avoids subclassing ADK internals (which change between versions)
  and lets ANY caller — ADK, a future custom agent, a one-off script —
  get routing behavior just by using a ``prism-<tier>`` model name.

**Lifecycle.**
  Built once at app startup via ``init_router()`` from ``main.py`` lifespan.
  Disposed via ``dispose_router()`` on shutdown (router has no resources to
  release beyond gc, but we null the singleton so tests can reinit).

**Testability.**
  The factory takes the API keys explicitly (not from settings) so tests
  can build a router with fake keys. The shim install is idempotent so
  repeated startups in test runners don't compound.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config import settings
from src.services.model_router_config import (
    TIER_CONFIGS,
    Tier,
    virtual_model_name,
)

if TYPE_CHECKING:
    from litellm import Router

logger = logging.getLogger(__name__)


# ── Singleton state ────────────────────────────────────────────────────────

_router_instance: "ModelRouter | None" = None
_litellm_patched: bool = False
# Original references we restore on uninstall. We patch TWO locations because
# ADK's ``google.adk.models.lite_llm`` lazy-imports ``acompletion`` /
# ``completion`` into its own module globals (verified against ADK source —
# uses ``globals()["acompletion"] = getattr(litellm, "acompletion")``), so a
# patch at only ``litellm.acompletion`` would not intercept ADK's calls.
_original_acompletion: Any = None
_original_completion: Any = None
_original_adk_acompletion: Any = None
_original_adk_completion: Any = None


@dataclass(slots=True)
class DeploymentSpec:
    """One (model × key) row in the LiteLLM ``model_list``."""

    virtual_name: str
    real_model: str
    api_key_index: int  # 0-based; for logging only — we never log the key itself
    rpm: int
    tpm: int


def _provider_of(model_id: str) -> str:
    """LiteLLM provider prefix of a model id (``"gemini/gemini-2.5-pro"`` →
    ``"gemini"``). Bare ids (no slash) are treated as Gemini for back-compat."""
    return model_id.split("/", 1)[0].lower() if "/" in model_id else "gemini"


class ModelRouter:
    """Owns the LiteLLM ``Router`` instance and exposes tier-based access."""

    def __init__(self, api_keys: list[str]) -> None:
        if not api_keys:
            raise ValueError(
                "ModelRouter requires at least one Gemini API key. "
                "Set GEMINI_API_KEY (and optionally GEMINI_API_KEY_1..4) in .env."
            )
        self._api_keys = api_keys
        self._deployments: list[DeploymentSpec] = []
        self._router: Router | None = None

    # ── Construction ──────────────────────────────────────────────────────

    def build(self) -> None:
        """Construct the underlying ``litellm.Router``. Idempotent."""
        if self._router is not None:
            return

        from litellm import Router  # lazy import — keeps module light at import time

        model_list, fallbacks = self._build_model_list_and_fallbacks()
        self._router = Router(
            model_list=model_list,
            fallbacks=fallbacks,
            routing_strategy=settings.MODEL_ROUTER_STRATEGY,
            cooldown_time=settings.MODEL_ROUTER_COOLDOWN_SECONDS,
            # Retries across DIFFERENT healthy deployments in the group on a
            # retryable error (429/503). Free-tier 503s are common, so 2 gives
            # the request a couple of shots at other models/keys before the
            # cross-tier fallback kicks in.
            num_retries=2,
            # Don't crash the whole router if a single model in the list is
            # unrecognized (Gemini may rename a model). Log + skip.
            set_verbose=False,
            allowed_fails=2,
        )

        _install_litellm_shim(self._router)
        logger.info(
            "ModelRouter built: %d deployments across %d keys, %d tiers, strategy=%s",
            len(self._deployments),
            len(self._api_keys),
            len(TIER_CONFIGS),
            settings.MODEL_ROUTER_STRATEGY,
        )

    def _keys_for_provider(self, provider: str) -> list[str]:
        """API key pool for a provider. Gemini uses the keys this router was
        built with; other providers read their dedicated settings key (blank →
        empty list → that model is skipped). Extend here for Anthropic etc."""
        if provider == "gemini":
            return self._api_keys
        if provider == "openai":
            key = (getattr(settings, "OPENAI_API_KEY", "") or "").strip()
            return [key] if key else []
        # Generic convention: <PROVIDER>_API_KEY in settings (e.g. ANTHROPIC_API_KEY).
        key = (getattr(settings, f"{provider.upper()}_API_KEY", "") or "").strip()
        return [key] if key else []

    # Cross-tier safety net (HEADS only), appended AFTER a tier's own
    # provider-failover chain. Free-tier Gemini still 503/429s, and a whole
    # provider chain could (rarely) cool out — so each generative tier can
    # still spill into another rather than failing the user. classify (no
    # tools) spills into fast. Embedding is deliberately absent — mixing
    # embedding models would corrupt the vector space.
    _CROSS_TIER: dict[str, list[str]] = {
        "fast": ["quality"],
        "quality": ["fast"],
        "classify": ["fast"],
    }

    def _build_model_list_and_fallbacks(self) -> tuple[list[dict], list[dict]]:
        """Expand TIER_CONFIGS × api_keys into a flat deployment list, grouped
        by PROVIDER so providers fail over in strict priority order.

        For each tier, models are bucketed by provider in first-appearance
        order (e.g. openai → deepseek → gemini). Each bucket becomes its own
        LiteLLM model group: the first available provider is the tier HEAD
        (``prism-<tier>``, what ``acquire()`` returns); the rest are
        ``prism-<tier>-fb1``, ``-fb2`` … chained as the head's ``fallbacks``.
        LiteLLM only moves to the next provider group once the current one is
        exhausted/cooled (it does NOT load-balance across providers) — within a
        single provider group, multiple models + keys DO load-balance.

        A model whose provider key isn't set is skipped, so the chain
        auto-collapses: with only Gemini keys configured, a tier has a single
        group named ``prism-<tier>`` — identical to the pre-multi-provider
        behavior.
        """
        model_list: list[dict] = []
        self._deployments = []

        # Pass 1 — bucket each tier's models by provider (first-appearance
        # order), dropping providers with no key. Record which tiers built a
        # head group (so cross-tier fallbacks only target real groups).
        tier_groups: dict[str, list[tuple[str, list[str]]]] = {}
        for tier_name, config in TIER_CONFIGS.items():
            order: list[str] = []
            bucket: dict[str, list[str]] = {}
            for real_model in config["models"]:
                provider = _provider_of(real_model)
                if not self._keys_for_provider(provider):
                    logger.info(
                        "Router: skipping %r in tier %r — no %s API key configured.",
                        real_model, tier_name, provider,
                    )
                    continue
                if provider not in bucket:
                    bucket[provider] = []
                    order.append(provider)
                bucket[provider].append(real_model)
            groups = [(p, bucket[p]) for p in order]
            if groups:
                tier_groups[tier_name] = groups

        # Pass 2 — build deployments + per-tier fallback chains.
        fallbacks: list[dict] = []
        for tier_name, groups in tier_groups.items():
            config = TIER_CONFIGS[tier_name]
            head = virtual_model_name(tier_name)
            # Group names: head, head-fb1, head-fb2, … (one per provider bucket).
            group_names = [head] + [f"{head}-fb{i}" for i in range(1, len(groups))]
            for gname, (_provider, models) in zip(group_names, groups):
                keys = self._keys_for_provider(_provider)
                for real_model in models:
                    for idx, api_key in enumerate(keys):
                        spec = DeploymentSpec(
                            virtual_name=gname,
                            real_model=real_model,
                            api_key_index=idx,
                            rpm=config["rpm"],
                            tpm=config["tpm"],
                        )
                        self._deployments.append(spec)
                        model_list.append(
                            {
                                "model_name": gname,
                                "litellm_params": {
                                    "model": real_model,
                                    "api_key": api_key,
                                },
                                "rpm": config["rpm"],
                                "tpm": config["tpm"],
                            }
                        )
            # Head's fallback chain: this tier's provider sub-groups (fb1, fb2…)
            # then the cross-tier heads that actually built.
            chain = group_names[1:]
            for xt in self._CROSS_TIER.get(tier_name, []):
                xt_head = virtual_model_name(xt)
                if xt in tier_groups and xt_head not in chain and xt_head != head:
                    chain.append(xt_head)
            if chain:
                fallbacks.append({head: chain})

        return model_list, fallbacks

    # ── Public API ────────────────────────────────────────────────────────

    def acquire(self, tier: Tier) -> Any:
        """Return an ADK-compatible model object for the requested tier.

        The returned object is an ADK ``LiteLlm`` whose ``model`` attribute
        is the virtual name (e.g. ``"prism-fast"``). When ADK calls
        ``litellm.acompletion(model="prism-fast", ...)`` our installed shim
        intercepts and dispatches through the router.
        """
        if self._router is None:
            raise RuntimeError("ModelRouter.build() must be called before acquire().")
        if tier not in TIER_CONFIGS:
            raise KeyError(
                f"Unknown model tier {tier!r}. Known tiers: {list(TIER_CONFIGS)}"
            )

        from google.adk.models.lite_llm import LiteLlm

        return LiteLlm(model=virtual_model_name(tier))

    async def acomplete(
        self,
        tier: Tier,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        response_json: bool = False,
    ) -> str:
        """Single chat completion via a tier — for non-agent structured calls.

        Used by deterministic generators (e.g. BMC Phase 2 Lite's per-block
        summarization) that want one grounded LLM call, not a full agent loop.
        Goes through ``litellm.Router.acompletion`` so it gets the same
        multi-key load balancing + 429 cooldown + fallback as everything else.

        Returns the assistant message text. ``response_json=True`` requests
        JSON output where the provider supports it (best-effort — callers
        should still parse defensively).
        """
        if self._router is None:
            raise RuntimeError("ModelRouter.build() must be called before acomplete().")

        kwargs: dict[str, Any] = {
            "model": virtual_model_name(tier),
            "messages": messages,
            "temperature": temperature,
        }
        if response_json:
            # Gemini honors this; harmless where unsupported.
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._router.acompletion(**kwargs)
        return response.choices[0].message.content or ""

    async def aembed(self, texts: list[str], *, dimensions: int | None = None) -> list[list[float]]:
        """Embed a batch of texts via the ``embedding`` tier.

        Goes through ``litellm.Router.aembedding`` directly (NOT the
        ``litellm.acompletion`` shim) so it gets the same multi-key load
        balancing + 429 cooldown + fallback as chat, but on the embedding
        deployments. Returns one vector per input, order-preserving.

        Args:
            texts: Inputs to embed. Caller is responsible for batch sizing
                within the model's token limits (the Embedder service does this).
            dimensions: Optional Matryoshka truncation. Gemini embedding models
                support requesting fewer dimensions; defaults to the model's
                native size when None.
        """
        if self._router is None:
            raise RuntimeError("ModelRouter.build() must be called before aembed().")

        kwargs: dict[str, Any] = {
            "model": virtual_model_name("embedding"),
            "input": texts,
        }
        if dimensions is not None:
            kwargs["dimensions"] = dimensions

        response = await self._router.aembedding(**kwargs)
        # LiteLLM normalizes to an OpenAI-shaped response: ``.data`` is a list
        # of ``{"embedding": [...], "index": i}`` dicts. Re-sort by index to be
        # safe, then strip to plain vectors.
        items = sorted(response.data, key=lambda d: d.get("index", 0))
        return [list(item["embedding"]) for item in items]

    def health(self) -> dict:
        """Snapshot of router state — exposed via /api/v1/router/health (DEBUG)."""
        return {
            "ready": self._router is not None,
            "api_keys_count": len(self._api_keys),
            "deployments_count": len(self._deployments),
            "tiers": list(TIER_CONFIGS.keys()),
            "strategy": settings.MODEL_ROUTER_STRATEGY,
            "cooldown_seconds": settings.MODEL_ROUTER_COOLDOWN_SECONDS,
            # We deliberately do NOT include API keys (even masked).
            "deployments_by_tier": _group_deployments_for_health(self._deployments),
        }


# ── Module-level helpers ───────────────────────────────────────────────────


def init_router(api_keys: list[str]) -> ModelRouter:
    """Build the singleton ``ModelRouter``. Called once from app lifespan.

    Idempotent: subsequent calls return the existing instance.
    """
    global _router_instance
    if _router_instance is not None:
        return _router_instance
    router = ModelRouter(api_keys=api_keys)
    router.build()
    _router_instance = router
    return router


def dispose_router() -> None:
    """Clear the singleton — for tests + shutdown. Removes the litellm shim."""
    global _router_instance
    _router_instance = None
    _uninstall_litellm_shim()


def get_router() -> ModelRouter:
    """Fetch the singleton — raises if not yet initialized."""
    if _router_instance is None:
        raise RuntimeError(
            "ModelRouter not initialized. "
            "Either call init_router() at app startup, or disable the router "
            "via settings.MODEL_ROUTER_ENABLED = False."
        )
    return _router_instance


# ── LiteLLM shim ───────────────────────────────────────────────────────────
#
# The single piece of "magic" in this module. Patches ``litellm.acompletion``
# and ``litellm.completion`` once at startup. Calls with model names beginning
# with ``prism-`` are routed through our Router; everything else passes
# through unchanged. This is the ONLY way to make ADK's ``LiteLlm`` (which
# calls module-level ``litellm.acompletion``) honor the router, without
# subclassing ADK internals that change across versions.
#
# Document this thoroughly because future maintainers will look at a stack
# trace one day and wonder why ``litellm.acompletion`` is wrapped.


def _install_litellm_shim(router: "Router") -> None:
    """Wrap ``litellm.{a,}completion`` AND ADK's local module references.

    Why two locations: ADK's ``google.adk.models.lite_llm`` does
    ``globals()["acompletion"] = getattr(litellm, "acompletion")`` inside a
    lazy-loader (``_ensure_litellm_imported``). Once that loader runs, ADK
    holds its OWN reference to the original function; patching only
    ``litellm.acompletion`` would not intercept ADK's calls — they'd hit the
    unrouted original with ``model="prism-fast"`` and 400 with "model not
    found". We therefore patch both module-level references.
    """
    global _litellm_patched, _original_acompletion, _original_completion
    global _original_adk_acompletion, _original_adk_completion
    if _litellm_patched:
        return

    # Force ADK's lazy loader to populate its module globals so we can patch
    # them. The import is cheap (LiteLlm class only) but triggers the
    # ``_ensure_litellm_imported`` path on some ADK versions.
    import google.adk.models.lite_llm as adk_lite_llm  # noqa: F401
    import litellm

    if hasattr(adk_lite_llm, "_ensure_litellm_imported"):
        try:
            adk_lite_llm._ensure_litellm_imported()
        except Exception as exc:  # pragma: no cover — ADK internals best-effort
            logger.warning("ADK _ensure_litellm_imported failed (continuing): %s", exc)

    _original_acompletion = litellm.acompletion
    _original_completion = litellm.completion
    _original_adk_acompletion = getattr(adk_lite_llm, "acompletion", None)
    _original_adk_completion = getattr(adk_lite_llm, "completion", None)

    async def routed_acompletion(*args, **kwargs):
        model = kwargs.get("model") or (args[0] if args else None)
        if isinstance(model, str) and model.startswith("prism-"):
            kwargs_routed = {k: v for k, v in kwargs.items() if k != "model"}
            return await router.acompletion(model=model, **kwargs_routed)
        return await _original_acompletion(*args, **kwargs)

    def routed_completion(*args, **kwargs):
        model = kwargs.get("model") or (args[0] if args else None)
        if isinstance(model, str) and model.startswith("prism-"):
            kwargs_routed = {k: v for k, v in kwargs.items() if k != "model"}
            return router.completion(model=model, **kwargs_routed)
        return _original_completion(*args, **kwargs)

    # Patch both — ``litellm.*`` for any direct callers, ``adk_lite_llm.*``
    # for ADK's cached references.
    litellm.acompletion = routed_acompletion  # type: ignore[assignment]
    litellm.completion = routed_completion  # type: ignore[assignment]
    if _original_adk_acompletion is not None:
        adk_lite_llm.acompletion = routed_acompletion  # type: ignore[attr-defined]
    if _original_adk_completion is not None:
        adk_lite_llm.completion = routed_completion  # type: ignore[attr-defined]

    _litellm_patched = True
    logger.info(
        "litellm.{a,}completion shim installed (patched %s)",
        "litellm + adk.lite_llm" if _original_adk_acompletion else "litellm only",
    )


def _uninstall_litellm_shim() -> None:
    global _litellm_patched, _original_acompletion, _original_completion
    global _original_adk_acompletion, _original_adk_completion
    if not _litellm_patched:
        return
    import google.adk.models.lite_llm as adk_lite_llm
    import litellm

    if _original_acompletion is not None:
        litellm.acompletion = _original_acompletion  # type: ignore[assignment]
    if _original_completion is not None:
        litellm.completion = _original_completion  # type: ignore[assignment]
    if _original_adk_acompletion is not None:
        adk_lite_llm.acompletion = _original_adk_acompletion  # type: ignore[attr-defined]
    if _original_adk_completion is not None:
        adk_lite_llm.completion = _original_adk_completion  # type: ignore[attr-defined]

    _original_acompletion = None
    _original_completion = None
    _original_adk_acompletion = None
    _original_adk_completion = None
    _litellm_patched = False


def _group_deployments_for_health(specs: list[DeploymentSpec]) -> dict:
    """Aggregate deployments by tier for the /router/health response."""
    by_tier: dict[str, list[dict]] = {}
    for spec in specs:
        tier = spec.virtual_name.removeprefix("prism-")
        by_tier.setdefault(tier, []).append(
            {
                "model": spec.real_model,
                "api_key_index": spec.api_key_index,
                "rpm_cap": spec.rpm,
                "tpm_cap": spec.tpm,
            }
        )
    return by_tier
