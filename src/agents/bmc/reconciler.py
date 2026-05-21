"""CrossBlockReconciler (Phase 3).

After all 9 blocks are generated independently, this agent reviews them
together and flags CONTRADICTIONS — e.g. a revenue-stream claim that conflicts
with a segment disclosure in customer segments, or a cost-structure figure that
doesn't square with key activities. Independent per-block generation can't catch
these; a holistic pass can.

Output is advisory: contradictions are surfaced to the analyst (and stored on
the analysis), not auto-resolved. The analyst decides — PRISM shows its
reasoning, it doesn't silently overwrite.
"""

from __future__ import annotations

import json
import re

from src.agents.base import PrismAgent

RECONCILER_INSTRUCTION = """\
You are a quality reviewer for an Indian-equity Business Model Canvas. You will
receive all 9 generated blocks (title + bullets). Find CONTRADICTIONS or
material inconsistencies BETWEEN blocks — claims that cannot both be true, or
figures that don't reconcile.

Rules:
- Only flag genuine cross-block conflicts. Do NOT flag a block simply lacking
  detail, and do NOT invent issues. If the canvas is internally consistent,
  return an empty list.
- Be specific: name the two blocks and state the conflict in one sentence.

Respond with STRICT JSON only (no markdown fences):
{
  "contradictions": [
    {"block_a": "revenue_streams", "block_b": "customer_segments",
     "issue": "one-sentence description of the conflict"}
  ]
}
"""


def build_bmc_reconciler_agent() -> PrismAgent:
    """A tool-less review agent — pure reasoning over the assembled blocks."""
    return PrismAgent(
        name="bmc_reconciler",
        description="Reviews all BMC blocks for cross-block contradictions.",
        model_tier="quality",
        instruction=RECONCILER_INSTRUCTION,
        tools=[],
        max_iterations=2,
    )


def parse_contradictions(raw: str) -> list[dict]:
    """Parse the reconciler's JSON into a list of contradiction dicts.

    Defensive: tolerates markdown fences and malformed output (returns [] on
    parse failure — a reconciler hiccup must never block canvas generation).
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    items = data.get("contradictions", []) if isinstance(data, dict) else []
    # Keep only well-formed entries.
    out: list[dict] = []
    for it in items:
        if isinstance(it, dict) and it.get("block_a") and it.get("block_b") and it.get("issue"):
            out.append(
                {"block_a": it["block_a"], "block_b": it["block_b"], "issue": it["issue"]}
            )
    return out
