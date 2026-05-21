"""Business Model Canvas — the agentic, filing-grounded BMC.

PRISM's signature differentiator. Unlike the old PRISM_ANALYST BMC (one-shot
LLM call with hallucinated sources) and every entrepreneur-facing BMC tool on
the market, PRISM's BMC grounds every block in actual filing chunks via the
RAG layer, with citations.

Phase 2 Lite (this slice):
  * One grounded LLM call per block (retrieve chunks → summarize → cite).
  * Blocks run with bounded concurrency (not full ADK multi-agent yet).
  * 9 canonical Osterwalder blocks.

Phase 3 (later, behind the same interface):
  * Per-block ADK sub-agents with tool-calling.
  * CrossBlockReconciler to catch contradictions.
  * Per-block drill-down chat.

The generation is exposed BOTH as an agent tool (``bmc.generate``) and a
user-facing surface (``@bmc TICKER`` / sidebar), per the approved plan.
"""

from src.agents.bmc.block_agent import build_bmc_block_agent
from src.agents.bmc.blocks import BMC_BLOCKS, BMCBlockDef
from src.agents.bmc.reconciler import build_bmc_reconciler_agent, parse_contradictions

__all__ = [
    "BMC_BLOCKS",
    "BMCBlockDef",
    "build_bmc_block_agent",
    "build_bmc_reconciler_agent",
    "parse_contradictions",
]
