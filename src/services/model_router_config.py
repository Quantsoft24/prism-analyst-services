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
    # Ordered model IDs in LiteLLM provider/model format.
    # Earlier entries are preferred; later entries are fallbacks.
    # The router replicates each entry per available API key.
    models: list[str]
    # Per-deployment soft caps used by ``usage-based-routing-v2`` to spread
    # load BEFORE we hit the actual provider's 429. Conservative defaults
    # below the published free-tier RPM, to leave headroom.
    rpm: int
    tpm: int


# ── Free-tier configuration (May 2026 Google AI Studio limits, 2 keys) ──
#
# Numbers are PER-KEY caps (not totals). Router multiplies by number of
# configured keys at startup. Conservative — sits ~20% below published
# Gemini AI Studio limits to absorb burst + clock drift.
TIER_CONFIGS: Final[dict[Tier, TierConfig]] = {
    # NOTE: every model ID below is verified against the live AI Studio
    # ListModels for this account (2026-05). Do NOT add a model name without
    # confirming it exists — a 404 is non-retryable and kills the request with
    # no fallback. `gemini-3-flash` (no suffix) does NOT exist; the real names
    # are `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-3.1-flash-lite`.
    "fast": {
        "description": "Tool-using agent steps, intent detection, Q&A. Function-calling required.",
        "models": [
            "gemini/gemini-2.5-flash-lite",     # primary free workhorse — fast, function-calling
            "gemini/gemini-2.5-flash",          # more capable flash — burst capacity
            "gemini/gemini-3.1-flash-lite",     # newer-gen flash-lite — extra capacity
        ],
        "rpm": 12,
        "tpm": 200_000,
    },
    "quality": {
        "description": "Final answer composition, BMC drafting, multi-step reasoning.",
        "models": [
            "gemini/gemini-2.5-flash",          # proven, strong for drafting
            "gemini/gemini-2.5-pro",            # highest quality (low free RPD — used as needed)
            "gemini/gemini-2.5-flash-lite",     # last resort so quality never 100% fails
        ],
        "rpm": 4,
        "tpm": 200_000,
    },
    "classify": {
        "description": "Pure classification/tagging — no tools. Cheap + abundant.",
        "models": [
            "gemini/gemini-2.5-flash-lite",     # cheap + reliable
            "gemini/gemma-4-26b-a4b-it",        # high RPD, no tools needed here
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
    Returns None if the name isn't one of ours (e.g. a direct model override)."""
    if not name.startswith("prism-"):
        return None
    suffix = name[len("prism-") :]
    if suffix in TIER_CONFIGS:
        return suffix  # type: ignore[return-value]
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
    # ── Paid (Vertex AI, Mumbai region) — fill when migrating to paid ──
    # "vertex_ai/gemini-2.5-pro": (1.25, 5.00),
    # "vertex_ai/gemini-3.1-pro": (1.25, 5.00),
    # "bedrock/anthropic.claude-sonnet-4-6-v1:0": (3.00, 15.00),
}
