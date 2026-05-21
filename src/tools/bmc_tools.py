"""Agent-callable BMC tools.

For Phase 2 Lite we expose a FAST, read-only tool: ``get_company_bmc`` returns
the latest stored canvas for a company (or an honest "not generated yet" note).
Agents (company_intel, future writer/report agents) call this to ground answers
about a company's business model.

Generation (9 grounded LLM calls) is intentionally NOT an in-chat tool — it's
too slow for a chat turn's timeout. It's triggered explicitly via
``POST /api/v1/bmc/{ticker}/run`` (UI ``@bmc`` action / CLI). This keeps the
agent tool snappy while the heavy work happens out-of-band — the "BMC is both
a tool and a surface" split from the plan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.database import session_scope
from src.repositories.bmc_repo import BMCRepository
from src.repositories.company_repo import CompanyRepository

if TYPE_CHECKING:
    from google.adk.tools import FunctionTool


async def get_company_bmc(ticker: str, firm_id: str = "QUANTSOFT") -> dict:
    """Return the latest Business Model Canvas for a company, if one exists.

    Use this when the user asks about a company's *business model* — how it
    makes money, who its customers are, its key partners/activities/resources,
    cost structure, etc. Returns the 9 canonical blocks with cited bullets.

    Args:
        ticker: NSE ticker, e.g. "TCS".
        firm_id: Tenant id (defaults to the dev firm).

    Returns:
        On success: {"found": true, "ticker", "version", "overall_confidence",
                     "blocks": [{block_id, title, bullets, confidence, status}]}.
        If no canvas exists yet: {"found": false, "note": "..."} — tell the
        user a canvas hasn't been generated and they can trigger it; do NOT
        invent a business model.
    """
    async with session_scope() as session:
        company = await CompanyRepository(session).get_by_ticker(ticker.strip().upper())
        if company is None:
            return {"found": False, "note": f"{ticker} is not in PRISM's coverage universe."}

        analysis = await BMCRepository(session).get_latest(firm_id, ticker.strip().upper())
        if analysis is None:
            return {
                "found": False,
                "note": (
                    f"No Business Model Canvas has been generated for {ticker} yet. "
                    "Suggest the user generate one (@bmc) — do not fabricate the model."
                ),
            }

        ordered = sorted(analysis.blocks, key=lambda b: b.order)
        return {
            "found": True,
            "ticker": analysis.ticker,
            "version": analysis.version,
            "overall_confidence": analysis.overall_confidence,
            "blocks": [
                {
                    "block_id": b.block_id,
                    "title": b.title,
                    "bullets": b.summary_bullets,
                    "confidence": b.confidence,
                    "status": b.status,
                }
                for b in ordered
            ],
        }


# ── ADK FunctionTool wrappers (lazy; mirrors company_tools/filing_tools) ─────


def _build_tools() -> list["FunctionTool"]:
    from google.adk.tools import FunctionTool

    return [FunctionTool(func=get_company_bmc)]


class _LazyToolList:
    def __init__(self) -> None:
        self._tools: list[FunctionTool] | None = None

    def _ensure(self) -> list:
        if self._tools is None:
            self._tools = _build_tools()
        return self._tools

    def __iter__(self):
        return iter(self._ensure())

    def __len__(self) -> int:
        return len(self._ensure())

    def to_list(self) -> list:
        return list(self._ensure())


BMC_TOOLS = _LazyToolList()
