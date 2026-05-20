"""The 9 canonical Business Model Canvas blocks.

Each block carries:
  * ``block_id`` — stable key (matches DB ``bmc_blocks.block_id``)
  * ``title`` — display title (Osterwalder canonical names)
  * ``order`` — position in the 3x3 grid (for deterministic UI layout)
  * ``retrieval_query`` — the query we run against the filing RAG layer to
    pull evidence for THIS block. Tuned per block so e.g. "revenue_streams"
    pulls revenue/segment/pricing chunks, not customer chunks.
  * ``focus`` — guidance injected into the per-block LLM prompt describing
    exactly what this block should capture.

The taxonomy + titles are ported from the canonical Osterwalder model (and
the old PRISM_ANALYST BMC), but the retrieval queries + focus are written
fresh for filing-grounded generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(slots=True, frozen=True)
class BMCBlockDef:
    block_id: str
    title: str
    order: int          # 0-8, row-major in the 3x3 canvas
    retrieval_query: str
    focus: str


BMC_BLOCKS: Final[list[BMCBlockDef]] = [
    BMCBlockDef(
        block_id="key_partners",
        title="Key Partners",
        order=0,
        retrieval_query="key partners alliances joint ventures suppliers subsidiaries strategic relationships",
        focus="The network of suppliers, alliances, JVs, and strategic partners the company relies on.",
    ),
    BMCBlockDef(
        block_id="key_activities",
        title="Key Activities",
        order=1,
        retrieval_query="core business activities operations services delivery what the company does",
        focus="The most important activities the company performs to deliver its value proposition.",
    ),
    BMCBlockDef(
        block_id="value_propositions",
        title="Value Propositions",
        order=2,
        retrieval_query="value proposition products services offerings differentiation competitive advantage",
        focus="The bundle of products/services that create value and why customers choose this company.",
    ),
    BMCBlockDef(
        block_id="customer_relationships",
        title="Customer Relationships",
        order=3,
        retrieval_query="customer relationships engagement retention client management account model",
        focus="How the company acquires, retains, and grows relationships with its customers.",
    ),
    BMCBlockDef(
        block_id="customer_segments",
        title="Customer Segments",
        order=4,
        retrieval_query="customer segments target clients markets served geographies verticals industries",
        focus="The distinct groups of customers/markets the company serves.",
    ),
    BMCBlockDef(
        block_id="key_resources",
        title="Key Resources",
        order=5,
        retrieval_query="key resources assets workforce headcount technology intellectual property capabilities",
        focus="The most important assets (people, IP, physical, financial) required to operate.",
    ),
    BMCBlockDef(
        block_id="channels",
        title="Channels",
        order=6,
        retrieval_query="channels distribution go to market sales delivery how products reach customers",
        focus="How the company reaches and delivers its value proposition to customers.",
    ),
    BMCBlockDef(
        block_id="cost_structure",
        title="Cost Structure",
        order=7,
        retrieval_query="cost structure expenses cost of revenue operating costs major cost drivers margins",
        focus="The most significant costs the business incurs to operate.",
    ),
    BMCBlockDef(
        block_id="revenue_streams",
        title="Revenue Streams",
        order=8,
        retrieval_query="revenue streams revenue by segment pricing how the company makes money revenue mix",
        focus="How the company generates revenue — segments, pricing models, revenue mix.",
    ),
]

# Quick lookup by id.
BMC_BLOCKS_BY_ID: Final[dict[str, BMCBlockDef]] = {b.block_id: b for b in BMC_BLOCKS}
