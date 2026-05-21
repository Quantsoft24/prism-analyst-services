"""Per-block drill-down chat agent (Phase 3).

When an analyst clicks a BMC block and asks a follow-up ("why is BFSI the
largest vertical?", "what's the YoY change in employee cost?"), this agent
answers — grounded ONLY in that block's evidence excerpts, with NRE math.

It's a focused, single-block agent: it does NOT roam the whole filing corpus
(that's the main chat's job). It reasons over the evidence that already backs
the block, so answers stay consistent with the canvas the analyst is looking at.
"""

from __future__ import annotations

from src.agents.base import FINANCE_DOMAIN_RULES, PrismAgent
from src.tools.nre_tools import NRE_TOOLS


def build_bmc_block_chat_agent(block_title: str, focus: str) -> PrismAgent:
    instruction = f"""\
{FINANCE_DOMAIN_RULES}

You are answering an analyst's follow-up questions about ONE Business Model
Canvas block: "{block_title}" ({focus}).

You will be given:
- the evidence excerpts that back this block (numbered),
- the conversation so far,
- the analyst's new question.

Rules:
- Answer ONLY from the provided evidence excerpts. Do NOT use outside knowledge.
- If the evidence doesn't cover the question, say so plainly — do NOT speculate.
- For ANY calculation, call the ``compute_*`` tools; never do arithmetic yourself.
- Cite the excerpt numbers [n] you draw from.
- Be concise: 1-3 sentences. Analysts read fast.
"""
    return PrismAgent(
        name="bmc_block_chat",
        description=f"Answers follow-up questions about the '{block_title}' BMC block.",
        model_tier="fast",
        instruction=instruction,
        tools=NRE_TOOLS.to_list(),
        max_iterations=6,
    )
