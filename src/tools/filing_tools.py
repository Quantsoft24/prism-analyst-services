"""Agent-callable tools for filing retrieval — the RAG window for agents.

``retrieve_filings`` is THE tool that makes PRISM's answers filing-grounded.
Any agent (company_intel now; BMC block-agents + writer later) calls it to
pull primary-source evidence with citations.

Each tool opens its own short-lived session via ``session_scope`` because
tools run inside ADK's runner loop, not a FastAPI request — same pattern as
``company_tools``.

The returned payload is citation-shaped on purpose: every hit carries the
``filing_id``, ``page_number``, and ``section`` so the agent can cite
"[MOIL Q4-FY26 results, p.4, MD&A]" and the UI can deep-link to the source.
This is the "show your work" contract from the architecture doc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.database import session_scope
from src.repositories.company_repo import CompanyRepository
from src.services.retrieval import HybridRetrievalService

if TYPE_CHECKING:
    from google.adk.tools import FunctionTool


async def retrieve_filings(
    query: str,
    ticker: str | None = None,
    section: str | None = None,
) -> dict:
    """Search ingested company filings for passages relevant to ``query``.

    Use this for any question about a company's actual disclosures — reported
    numbers, management commentary, risk factors, segment results, related-party
    transactions, etc. ALWAYS prefer this over your training knowledge when the
    question is about a specific company's filings, and cite the results.

    Args:
        query: Natural-language search, e.g. "revenue growth and margin guidance".
        ticker: Optional NSE ticker to scope the search to one company (e.g. "MOIL").
            Strongly recommended when the user names a company — it sharpens results.
        section: Optional filing section filter. One of: mda, balance_sheet,
            profit_loss, cash_flow, notes, auditors_report, directors_report,
            risk_factors, related_party, segment_reporting.

    Returns:
        Dict with:
          * ``hits``: list of {filing_id, section, page, text, score} — the
            evidence passages, best first. Cite these.
          * ``count``: number of hits.
          * ``note``: present + explains when nothing was found (e.g. company
            has no ingested filings yet) so you can tell the user honestly
            instead of guessing.
    """
    async with session_scope() as session:
        company_id = None
        if ticker:
            repo = CompanyRepository(session)
            company = await repo.get_by_ticker(ticker.strip().upper())
            if company is None:
                return {
                    "hits": [],
                    "count": 0,
                    "note": f"{ticker} is not in PRISM's coverage universe.",
                }
            company_id = company.id

        retriever = HybridRetrievalService(session)
        results = await retriever.retrieve(query, company_id=company_id, section=section)

        if not results:
            scope = f" for {ticker}" if ticker else ""
            return {
                "hits": [],
                "count": 0,
                "note": (
                    f"No ingested filings matched{scope}. Filing data may not be "
                    "ingested yet for this company. Say so honestly — do not "
                    "fabricate figures."
                ),
            }

        return {
            "count": len(results),
            "hits": [
                {
                    "filing_id": str(r.filing_id),
                    "section": r.section,
                    "page": r.page_number,
                    "text": r.text,
                    "score": round(r.fused_score, 4),
                }
                for r in results
            ],
        }


# ── ADK FunctionTool wrappers (lazy, mirrors company_tools pattern) ─────────


def _build_tools() -> list["FunctionTool"]:
    from google.adk.tools import FunctionTool

    return [FunctionTool(func=retrieve_filings)]


class _LazyToolList:
    """Defers the ADK import until first access — keeps the module importable
    without google-adk (tests, lint)."""

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


FILING_TOOLS = _LazyToolList()
