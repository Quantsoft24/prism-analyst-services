"""stock-chat — the teammate-built Indian-filings service, as agent tools.

The service (FastAPI, OpenAPI 3) exposes several endpoints. We wire the three
CURRENT ones as separate, precisely-described tools so the LLM can pick the
right one for any question:

  * ``stock_filings_read``    → POST /tools/read           (narrative Q&A from PDFs)
  * ``stock_filings_lookup``  → POST /tools/lookup-filings  (catalog metadata only)
  * ``stock_technicals``      → POST /tools/technicals      (live price / indicators)

The legacy ``/tools/ask`` and ``/tools/search-narrative`` are intentionally NOT
wired — the service's own docs say "do not use for new integrations".

Base URL: ``settings.STOCK_CHAT_URL`` (env ``STOCK_CHAT_URL``). No caller auth —
the service is network-restricted to the PRISM backend.
"""

from __future__ import annotations

import logging

import httpx

from src.config import settings
from src.integrations.tools._errors import make_error

logger = logging.getLogger(__name__)

# /tools/read reads PDFs + calls an answer LLM (cold sector survey ~30s, PDF
# budget 120s/doc) — generous timeout. The metadata/technicals calls are fast.
_READ_TIMEOUT = 120.0
_FAST_TIMEOUT = 30.0


def _base_url() -> str:
    return (settings.STOCK_CHAT_URL or "http://localhost:8011").rstrip("/")


async def _post(path: str, payload: dict, timeout: float) -> dict:
    """POST helper with transparent, graceful error reporting (Part-A).

    Errors come back in the standard shape (see _errors.py) so the agent
    runner can emit ToolResultEvent(ok=False, error=...) and the LLM gets
    a structured ``next_action`` hint.
    """
    url = f"{_base_url()}{path}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
    except httpx.TimeoutException as exc:
        logger.warning("stock-chat timed out at %s: %s", url, exc)
        return make_error(
            message="The filings service timed out reading filings. Try again in a moment, or refine the question.",
            code="stock_chat_timeout",
            next_action="ask_user_to_retry_later",
            retriable=True,
            detail=str(exc),
        )
    except httpx.RequestError as exc:
        logger.warning("stock-chat unreachable at %s: %s", url, exc)
        return make_error(
            message="The filings service is unreachable. The data team has been notified.",
            code="stock_chat_unreachable",
            next_action="ask_user_to_retry_later",
            retriable=True,
            detail=f"{path}: {exc}",
        )
    if resp.status_code >= 500:
        return make_error(
            message="The filings service returned an internal error.",
            code=f"stock_chat_http_{resp.status_code}",
            next_action="ask_user_to_retry_later",
            retriable=True,
            detail=resp.text,
        )
    if resp.status_code == 400:
        return make_error(
            message="The filings service rejected the request — the question may need more detail.",
            code="stock_chat_bad_request",
            next_action="ask_user_to_clarify",
            retriable=False,
            detail=resp.text,
        )
    if resp.status_code != 200:
        return make_error(
            message=f"The filings service returned HTTP {resp.status_code}.",
            code=f"stock_chat_http_{resp.status_code}",
            next_action="try_alternate_tool",
            retriable=False,
            detail=resp.text,
        )
    return resp.json()


async def stock_filings_read(
    question: str,
    company: str | None = None,
    companies: list[str] | None = None,
    category: str | None = None,
    period: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    max_filings: int = 3,
) -> dict:
    """Answer a NARRATIVE question about what an Indian listed company (or a whole
    sector) said, disclosed, announced, decided, or commented on in its NSE/BSE
    filings — strategy, risks, MD&A, sustainability, governance, board decisions,
    dividends, regulatory disclosures. Searches a 649k-filing catalog, reads the
    actual PDF(s), and returns a synthesised answer plus verbatim evidence
    passages with ``[Company | p.N]`` citations.

    WHEN TO USE: the user asks what a company/sector *said* or *disclosed*
    qualitatively, or when the filing may not be in PRISM's locally ingested set
    (this covers the full public NSE/BSE catalog).

    WHEN NOT TO USE: exact financial figures (precise revenue, margins, ratios,
    segment numbers) — use PRISM's own filing retrieval + compute_* tools. For
    *which filings exist* use ``stock_filings_lookup``; for live price/technicals
    use ``stock_technicals``.

    Always preserve the returned ``[Company | p.N]`` citation strings verbatim.
    If ``needs_clarification`` is true, ask the user the ``clarification_question``.

    Args:
        question: The user's natural-language question (required). Phrase sector
            questions naturally ("the cement sector"); the service infers the sector.
        company: Single company hint (name or ticker) — best-match resolved.
        companies: Multiple companies — each best-match resolved.
        category: One of "Annual Report", "Result", "Board Meeting", "AGM/EGM",
            "Corp. Action", "Company Update", "Insider Trading / SAST".
        period: Fiscal to-year, e.g. "2025" for FY2024-25.
        date_from: ISO YYYY-MM-DD lower bound (board-meeting/update windows).
        date_to: ISO YYYY-MM-DD upper bound.
        max_filings: PDFs to open, 1-8. Keep >=3 for sector surveys.

    Returns:
        dict with ``answer``, ``needs_clarification`` / ``clarification_question``,
        ``resolved_companies``, ``selected_filings``, ``evidence``. ``{"error": ...}``
        on a transport/service failure.
    """
    payload: dict = {"question": question, "synthesise": True, "max_filings": max_filings}
    if company:
        payload["company"] = company
    if companies:
        payload["companies"] = companies
    if category:
        payload["category"] = category
    if period:
        payload["period"] = period
    if date_from:
        payload["date_from"] = date_from
    if date_to:
        payload["date_to"] = date_to

    data = await _post("/tools/read", payload, _READ_TIMEOUT)
    if data.get("ok") is False or "error" in data:
        return data
    # Compute a data_freshness signal from the latest selected filing — the
    # runner emits it as a DataFreshnessEvent so the UI can show "as of …".
    selected = data.get("selected_filings") or []
    latest_dt = max(
        (f.get("announcement_dt") for f in selected if f.get("announcement_dt")),
        default=None,
    )
    # Trim bulky fields (document_excerpts / token_usage / timings) to keep context lean.
    return {
        "answer": data.get("answer"),
        "needs_clarification": data.get("needs_clarification", False),
        "clarification_question": data.get("clarification_question"),
        "resolved_companies": data.get("resolved_companies"),
        "selected_filings": [
            {
                "company_name": f.get("company_name"),
                "headline": f.get("headline"),
                "announcement_dt": f.get("announcement_dt"),
                "read_ok": f.get("read_ok"),
                "page_count": f.get("page_count"),
            }
            for f in selected
        ],
        "evidence": data.get("evidence"),
        "data_freshness": latest_dt,
    }


async def stock_filings_lookup(
    company: str | None = None,
    companies: list[str] | None = None,
    ticker: str | None = None,
    category: str | None = None,
    period: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    text_match: list[str] | None = None,
    limit: int = 50,
) -> dict:
    """List WHICH Indian NSE/BSE filings exist for a company/sector — pure catalog
    metadata, no PDF read, no LLM. Fast.

    WHEN TO USE: the user wants to know what filings are available or how many
    ("list Reliance's annual reports", "how many board meetings did TCS file in
    2026", "what corporate actions happened this quarter"). To answer a question
    *from* a filing's content, use ``stock_filings_read`` instead.

    Args:
        company: Single company (name or ticker) — best-match resolved.
        companies: Multiple companies.
        ticker: Exact scrip code or NSE symbol.
        category: "Annual Report" | "Result" | "Board Meeting" | "AGM/EGM" |
            "Corp. Action" | "Company Update" | "Insider Trading / SAST".
        period: Fiscal to-year, e.g. "2025".
        date_from: ISO YYYY-MM-DD lower bound.
        date_to: ISO YYYY-MM-DD upper bound.
        text_match: Topical keywords for soft ranking (e.g. ["dividend"]).
        limit: Max rows to return (default 50).

    Returns:
        dict with ``total`` (full match count), ``resolved_companies``, and
        ``filings`` (the current page: company, category, period, date, exchange,
        headline, …). ``{"error": ...}`` on failure.
    """
    payload: dict = {"limit": limit, "order_by_date_desc": True}
    if company:
        payload["company"] = company
    if companies:
        payload["companies"] = companies
    if ticker:
        payload["ticker"] = ticker
    if category:
        payload["category"] = category
    if period:
        payload["period"] = period
    if date_from:
        payload["date_from"] = date_from
    if date_to:
        payload["date_to"] = date_to
    if text_match:
        payload["text_match"] = text_match

    data = await _post("/tools/lookup-filings", payload, _FAST_TIMEOUT)
    if data.get("ok") is False or "error" in data:
        return data
    filings = data.get("filings") or []
    latest_dt = max(
        (f.get("announcement_dt") for f in filings if f.get("announcement_dt")),
        default=None,
    )
    return {
        "total": data.get("total"),
        "resolved_companies": data.get("resolved_companies"),
        "filings": [
            {
                "company_name": f.get("company_name"),
                "category": f.get("category"),
                "subcategory": f.get("subcategory"),
                "period": f.get("period"),
                "announcement_dt": f.get("announcement_dt"),
                "exchange": f.get("exchange"),
                "headline": f.get("headline"),
                "pdf_status": f.get("pdf_status"),
            }
            for f in filings
        ],
        "data_freshness": latest_dt,
    }


async def stock_technicals(
    company: str | None = None,
    ticker: str | None = None,
    exchange: str = "NSE",
    period: str = "1y",
) -> dict:
    """Live price + technical indicators for an Indian listed company — current
    price, 52-week range, moving averages (20/50/200), RSI(14), MACD. Backed by
    yfinance.

    WHEN TO USE: market-data questions ("what's TCS trading at", "is Infosys above
    its 200-day MA", "RSI for HDFC Bank"). NOT for anything from filings.

    Provide EITHER ``company`` OR ``ticker`` (at least one is required).

    Args:
        company: Company name — resolved to a symbol.
        ticker: Exact symbol (e.g. "TCS").
        exchange: "NSE" | "BSE" — only used with ``ticker``.
        period: yfinance history window ("1y", "6mo", "5d", …), default "1y".

    Returns:
        dict with ``status`` ("ok" or a failure reason), ``current_price``,
        ``fifty_two_week_high/low``, ``ma_20/50/200``, ``rsi_14``, ``macd`` …
        On non-ok status the indicator fields are null and ``error`` has detail.
        ``{"error": ...}`` on a transport failure.
    """
    if not company and not ticker:
        return make_error(
            message="Need either a company name or a ticker for technicals.",
            code="missing_company_hint",
            next_action="ask_user_to_clarify",
        )
    payload: dict = {"exchange": exchange, "period": period}
    if company:
        payload["company"] = company
    if ticker:
        payload["ticker"] = ticker
    data = await _post("/tools/technicals", payload, _FAST_TIMEOUT)
    # Technicals are intrinsically "as of now" — mark for UI freshness chip.
    if data.get("ok") is not False and "error" not in data:
        data.setdefault("data_freshness", "live")
    return data


# The registry's `python` adapter wraps each plain function here in a FunctionTool.
STOCK_CHAT_TOOLS = [stock_filings_read, stock_filings_lookup, stock_technicals]
