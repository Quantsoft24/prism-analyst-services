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
            # ``num_retries`` is per-deployment retries before LiteLLM treats
            # the deployment as failed and tries the next. 1 is enough — the
            # router itself does the cross-deployment retry via fallbacks.
            num_retries=1,
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

    def _build_model_list_and_fallbacks(self) -> tuple[list[dict], list[dict]]:
        """Expand TIER_CONFIGS × api_keys into a flat deployment list.

        Returns the ``model_list`` and ``fallbacks`` arguments LiteLLM
        ``Router`` expects.
        """
        model_list: list[dict] = []
        self._deployments = []

        for tier_name, config in TIER_CONFIGS.items():
            virtual = virtual_model_name(tier_name)
            for real_model in config["models"]:
                for idx, api_key in enumerate(self._api_keys):
                    spec = DeploymentSpec(
                        virtual_name=virtual,
                        real_model=real_model,
                        api_key_index=idx,
                        rpm=config["rpm"],
                        tpm=config["tpm"],
                    )
                    self._deployments.append(spec)
                    model_list.append(
                        {
                            "model_name": virtual,
                            "litellm_params": {
                                "model": real_model,
                                "api_key": api_key,
                            },
                            "rpm": config["rpm"],
                            "tpm": config["tpm"],
                        }
                    )

        # Inter-tier fallback: if every deployment in ``prism-quality`` cools
        # down (e.g., all 3 Flash models 429-rate-limited at once), route the
        # request to ``prism-fast`` rather than failing the user. The reverse
        # is NOT configured — quality fallback is acceptable, but classify
        # shouldn't bleed into quality (we'd waste quota).
        fallbacks: list[dict] = [
            {"prism-quality": ["prism-fast"]},
        ]
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

    import litellm

    # Force ADK's lazy loader to populate its module globals so we can patch
    # them. The import is cheap (LiteLlm class only) but triggers the
    # ``_ensure_litellm_imported`` path on some ADK versions.
    import google.adk.models.lite_llm as adk_lite_llm  # noqa: F401

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
    import litellm
    import google.adk.models.lite_llm as adk_lite_llm

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
