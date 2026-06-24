"""Declarative tier → model-chain configuration.

**This is the single file you edit when:**
  * Google releases a new free model worth adopting
  * You move to paid Vertex / Bedrock / OpenAI — prepend the premium model
    to the relevant tier's chain; everything else (agents, router, runner)
    stays untouched
  * You discover a model is better suited to a different tier
  * Per-key free-tier rate limits change

Tier semantics (see plan addendum):

    fast       — Tool-using agent steps, intent detection, single-company Q&A.
                 Must support function calling.
    quality    — Final answer composition, BMC drafting, multi-step reasoning.
                 Higher accuracy budget, slower OK.
    classify   — Lightweight classification / tagging WITHOUT tools.
                 Cheapest models, highest free-tier RPD available.
    embedding  — Vector embedding (Slice 5+ consumer).

The router treats each (model × api_key) pair as a separate deployment for
load balancing and 429-aware fallback. A chain of N model IDs with K API
keys expands to N×K deployments in the LiteLLM ``model_list``.

Adding a new tier:
  1. Add an entry to ``TIER_CONFIGS``.
  2. Update the type alias ``Tier`` below.
  3. Add a pricing entry to ``MODEL_PRICING_USD_PER_1M`` for cost tracking.
  4. (Optional) Reference the new tier in an agent's ``PrismAgent(model_tier=...)``.
"""

from __future__ import annotations

from typing import Final, Literal, TypedDict

Tier = Literal["fast", "quality", "classify", "embedding"]


class TierConfig(TypedDict):
    description: str
    # Ordered model IDs in LiteLLM ``provider/model`` format, listed in
    # PROVIDER-PRIORITY order (e.g. all openai/* first, then deepseek/*, then
    # gemini/*). The router groups consecutive same-provider models into
    # sub-groups and chains them as strict fallbacks: the PRIMARY provider's
    # group is fully exhausted/cooled before the next provider is touched
    # (NOT load-balanced across providers — see ModelRouter
    # ._build_model_list_and_fallbacks). Within one provider's group, multiple
    # models + multiple API keys DO load-balance. A model whose provider key
    # isn't configured is skipped, so the chain auto-collapses (e.g. with only
    # Gemini keys set, a tier behaves exactly as the Gemini-only list it used
    # to be).
    models: list[str]
    # Per-deployment soft caps used by ``usage-based-routing-v2`` WITHIN a
    # provider group. Conservative defaults below the published free-tier RPM.
    rpm: int
    tpm: int


# ── Provider-priority configuration: OpenAI (primary) → DeepSeek → Gemini ──
#
# Strategy (2026-06): OpenAI is primary (free daily-token program — 2.5M/day
# across the mini pool, 250k/day across the large pool; paid overflow past
# that), DeepSeek is the 1st fallback (cheap, OpenAI-compatible), Gemini the
# 2nd fallback (free, multi-key — the final safety net). Map the chatty `fast`
# orchestrator onto OpenAI's big *mini* pool and the low-volume `quality`
# synthesis onto the scarcer *large* pool.
#
# MODEL-ID VERIFICATION (2026-06-13, against the live accounts):
#   • OpenAI: gpt-5.4 / gpt-5.4-mini / gpt-5.4-nano all exist; gpt-5.4-mini
#     verified to accept tool-calls + temperature=0.2 (no reasoning-model
#     param restriction) → safe as the agent primary.
#   • DeepSeek: the live model ids are `deepseek-v4-flash` / `deepseek-v4-pro`
#     (NOT deepseek-chat/-reasoner). The key currently has NO balance
#     ("Insufficient Balance") → DeepSeek deployments error-through to Gemini
#     until the account is funded. That's fine (graceful), just inert for now.
#   • gemini/* ids verified against AI Studio ListModels (2026-05).
# A 404 on an unknown id is non-retryable — re-verify before adding new ids.
# rpm/tpm are per-deployment caps WITHIN each provider group (per-key).
TIER_CONFIGS: Final[dict[Tier, TierConfig]] = {
    "fast": {
        "description": "Tool-using agent steps, intent detection, Q&A. Function-calling required.",
        # PRIMARY: OpenAI mini pool (2.5M/day free w/ data-sharing) — strong,
        #   reliable tool-use for the company_intel orchestrator (only `fast`
        #   consumer). FB1: DeepSeek v4-flash (cheap; inert until funded).
        #   FB2: Gemini flash (free, multi-key) — keeps strong instruction-
        #   following so tool-call qualifiers (period/topic) survive.
        "models": [
            "openai/gpt-5.4-mini",              # PRIMARY (OpenAI mini pool)
            "deepseek/deepseek-v4-flash",       # FB1 (DeepSeek)
            "gemini/gemini-2.5-flash",          # FB2 primary Gemini — best tool-use
            "gemini/gemini-2.5-flash-lite",     # FB2 burst capacity
            "gemini/gemini-3.1-flash-lite",     # FB2 extra capacity
        ],
        "rpm": 12,
        "tpm": 200_000,
    },
    "quality": {
        "description": "Final answer composition, BMC drafting, multi-step reasoning.",
        # PRIMARY: OpenAI large pool (250k/day free) — best synthesis/reasoning.
        # FB1: DeepSeek v4-pro. FB2: Gemini pro/flash (free).
        "models": [
            "openai/gpt-5.4",                   # PRIMARY (OpenAI large pool)
            "deepseek/deepseek-v4-pro",         # FB1 (DeepSeek)
            "gemini/gemini-2.5-flash",          # FB2 — proven for drafting
            "gemini/gemini-2.5-pro",            # FB2 — highest quality (low free RPD)
            "gemini/gemini-2.5-flash-lite",     # FB2 — last resort so quality never 100% fails
        ],
        "rpm": 4,
        "tpm": 200_000,
    },
    "classify": {
        "description": "Pure classification/tagging — no tools. Cheap + abundant.",
        # PRIMARY: OpenAI nano (mini pool, cheapest). FB1: DeepSeek. FB2: Gemini.
        "models": [
            "openai/gpt-5.4-nano",              # PRIMARY (OpenAI nano, mini pool)
            "deepseek/deepseek-v4-flash",       # FB1
            "gemini/gemini-2.5-flash-lite",     # FB2 — cheap + reliable
            "gemini/gemma-4-26b-a4b-it",        # FB2 — high RPD
            "gemini/gemma-4-31b-it",
        ],
        "rpm": 12,
        "tpm": 200_000,
    },
    "embedding": {
        # IMPORTANT: exactly ONE embedding model. Mixing embedding models
        # corrupts the shared vector space (cosine distances across models are
        # meaningless), which silently wrecks retrieval. Resilience comes from
        # the router replicating this ONE model across multiple API keys — NOT
        # from falling back to a different model. To change the embedding model
        # you must re-embed the whole corpus (a migration), never hot-swap.
        #
        # ``gemini-embedding-001`` is the current GA Gemini embedding model on
        # the Developer API (verified available; supports output_dimensionality
        # truncation to settings.EMBEDDING_DIMENSION via Matryoshka).
        "description": "Vector embedding for retrieval. Single model, multi-key for resilience.",
        "models": [
            "gemini/gemini-embedding-001",
        ],
        "rpm": 80,
        "tpm": 25_000,
    },
}


# Virtual model name surfaced to ADK. Agents see ``"prism-fast"`` etc.; the
# router resolves to a concrete (model × key) deployment at call time.
def virtual_model_name(tier: Tier) -> str:
    return f"prism-{tier}"


def tier_from_virtual(name: str) -> Tier | None:
    """Reverse lookup — useful when ADK hands us back a model name on an event.
    Returns None if the name isn't one of ours (e.g. a direct model override).
    Also resolves provider-fallback sub-group names (``prism-fast-fb1`` → ``fast``)
    that the router creates for strict OpenAI→DeepSeek→Gemini failover."""
    if not name.startswith("prism-"):
        return None
    suffix = name[len("prism-") :]
    # Strip a provider-fallback sub-group marker, e.g. "fast-fb1" → "fast".
    base = suffix.split("-fb", 1)[0]
    if base in TIER_CONFIGS:
        return base  # type: ignore[return-value]
    return None


# ── Pricing table (USD per 1M tokens) ────────────────────────────────────
#
# Free-tier entries have (0, 0). Paid-tier rates come in when we move to
# Vertex AI / Bedrock and the router prepends premium models to a tier.
# Used by ``AgentRunner._estimate_cost_usd``.
MODEL_PRICING_USD_PER_1M: Final[dict[str, tuple[float, float]]] = {
    # ── Free (Gemini AI Studio + Gemma) — all verified to exist (2026-05) ──
    "gemini/gemini-2.5-flash": (0.0, 0.0),
    "gemini/gemini-2.5-flash-lite": (0.0, 0.0),
    "gemini/gemini-2.5-pro": (0.0, 0.0),
    "gemini/gemini-3.1-flash-lite": (0.0, 0.0),
    "gemini/gemma-4-26b-a4b-it": (0.0, 0.0),
    "gemini/gemma-4-31b-it": (0.0, 0.0),
    "gemini/gemini-embedding-001": (0.0, 0.0),
    # ── OpenAI (primary) — APPROXIMATE rates (USD/1M in,out); refine from the
    #    billing dashboard. NOTE: with the data-sharing free program these are
    #    effectively $0 until the daily budget (2.5M mini / 250k large), then
    #    billed at ~these rates → this is a conservative cost CEILING. ──
    "openai/gpt-5.4": (1.25, 10.00),         # large pool (quality primary)
    "openai/gpt-5.4-mini": (0.25, 2.00),     # mini pool (fast primary)
    "openai/gpt-5.4-nano": (0.05, 0.40),     # mini pool (classify primary)
    # ── DeepSeek (1st fallback) — APPROXIMATE; verify when account is funded ──
    "deepseek/deepseek-v4-flash": (0.28, 0.42),
    "deepseek/deepseek-v4-pro": (0.55, 2.19),
    # ── Paid (Vertex AI, Mumbai region) — fill when migrating to paid ──
    # "vertex_ai/gemini-2.5-pro": (1.25, 5.00),
}
