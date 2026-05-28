"""stock-chat — the teammate-built Indian-filings service, as agent tools.

The service (FastAPI, OpenAPI 3) exposes several endpoints. We wire the three
CURRENT ones as separate, precisely-described tools so the LLM can pick the
right one for any question:

  * ``stock_filings_read``    → POST /tools/read           (narrative Q&A from PDFs)
  * ``stock_filings_lookup``  → POST /tools/lookup-filings  (catalog metadata only)
  * ``stock_technicals``      → POST /tools/technicals      (live price / indicators)

The legacy ``/tools/ask`` and ``/tools/search-narrative`` are intentionally NOT
wired — the service's own docs say "do not use for new integrations".

**v3 contract (2026-05):** ``stock_filings_read`` sends only 3 fields —
``question``, ``company`` (optional hint), and ``synthesise``. Every catalog
filter (category, period, dates, industry, text_match, max_filings) is derived
by the service's own LLM planner, which has domain context the upstream agent
lacks (the catalog's exact category enum, the screener industry taxonomy,
date-phrase semantics). The upstream agent MUST NOT pre-fill those fields.

Base URL: ``settings.STOCK_CHAT_URL`` (env ``STOCK_CHAT_URL``). No caller auth —
the service is network-restricted to the PRISM backend.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from src.config import settings
from src.integrations.tools._errors import make_error

logger = logging.getLogger(__name__)

# One-shot retry on transient transport failures. 250ms is below the
# user's perception threshold for "feeling slow" but long enough to clear
# most TCP / DNS / brief-load-spike blips. Anything that survives the
# retry is a real failure the agent + UI should surface.
_RETRY_DELAY_S = 0.25

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

    Transient transport failures (timeout / network error) get **one
    silent retry** after a 250 ms pause — that absorbs brief blips
    without surfacing them to the LLM. When the retry runs, the response
    dict carries ``retry_count: 1`` so the runner can emit a
    ``ToolRetryEvent`` (the UI shows the ↻ chip on the tool card). 4xx
    responses are never retried — those reflect bad input, not bad luck.
    """
    url = f"{_base_url()}{path}"
    last_exc: Exception | None = None
    retry_count = 0
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
            if attempt == 2:
                retry_count = 1  # first attempt failed transiently — track for runner
            break  # exit the retry loop — we have a response
        except httpx.TimeoutException as exc:
            last_exc = exc
            if attempt == 1:
                logger.warning(
                    "stock-chat timed out at %s (attempt %d) — retrying in %sms",
                    url, attempt, int(_RETRY_DELAY_S * 1000),
                )
                await asyncio.sleep(_RETRY_DELAY_S)
                continue
            logger.warning("stock-chat timed out at %s after retry: %s", url, exc)
            return make_error(
                message="The filings service timed out reading filings. Try again in a moment, or refine the question.",
                code="stock_chat_timeout",
                next_action="ask_user_to_retry_later",
                retriable=True,
                detail=str(exc),
            )
        except httpx.RequestError as exc:
            last_exc = exc
            if attempt == 1:
                logger.warning(
                    "stock-chat unreachable at %s (attempt %d) — retrying in %sms",
                    url, attempt, int(_RETRY_DELAY_S * 1000),
                )
                await asyncio.sleep(_RETRY_DELAY_S)
                continue
            logger.warning("stock-chat unreachable at %s after retry: %s", url, exc)
            return make_error(
                message="The filings service is unreachable. The data team has been notified.",
                code="stock_chat_unreachable",
                next_action="ask_user_to_retry_later",
                retriable=True,
                detail=f"{path}: {exc}",
            )
    else:
        # Safety net — the for/else only runs when the loop completed
        # without break, i.e. both attempts raised. Branches above already
        # return on the second failure, so this is defensive.
        return make_error(
            message="The filings service is unreachable.",
            code="stock_chat_unreachable",
            next_action="ask_user_to_retry_later",
            retriable=True,
            detail=str(last_exc),
        )

    # ``resp`` is guaranteed to be set once we broke out of the loop.
    if resp.status_code >= 500:
        return make_error(
            message="The filings service returned an internal error.",
            code=f"stock_chat_http_{resp.status_code}",
            next_action="ask_user_to_retry_later",
            retriable=True,
            detail=resp.text,
        )
    if resp.status_code in (400, 422):
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
    data = resp.json()
    # Tag a transient-retry success so the runner can emit a
    # ToolRetryEvent (renders ↻ on the tool card in the UI).
    if retry_count and isinstance(data, dict):
        data["retry_count"] = retry_count
    return data


async def stock_filings_read(
    question: str,
    company: str | list[str] | None = None,
    synthesise: bool = True,
) -> dict:
    """Answer a NARRATIVE question about what an Indian listed company (or a whole
    sector) said, disclosed, announced, decided, or commented on in its NSE/BSE
    filings — strategy, risks, MD&A, sustainability, governance, board decisions,
    dividends, regulatory disclosures, board members / directors. Searches a
    650k-filing catalog, reads the actual PDF(s), and returns a synthesised
    answer plus verbatim evidence passages with ``[Company | p.N]`` citations.

    **SIMPLIFIED CONTRACT (v3):** This tool's internal LLM planner handles ALL
    catalog filtering — category, period, dates, sector industry matching,
    text keywords, and how many PDFs to open. Do NOT pre-fill those fields;
    the planner has domain-specific context (the catalog's exact category enum,
    screener industry taxonomy, and date-phrase semantics) that the upstream
    agent lacks. Just pass the question and an optional company hint.

    WHEN TO USE:
      • The user asks what a company/sector *said* or *disclosed* qualitatively.
      • The user asks about strategy, risks, MD&A, governance, sustainability.
      • The user asks for board members, directors, board decisions.
      • The user asks about dividends, corporate actions FROM filings.
      • The user asks for sector-wide narrative ("What did the cement sector's
        boards decide recently?") — the planner infers the sector from 16
        supported sector keys (Auto, Bank, Pharma, IT, Steel, Cement, Chemical,
        FMCG, Telecom, Insurance, RealEstate, Power, Textile, Hospital, Airline,
        Media) and picks filings from distinct companies.
      • Comparison questions ("Compare ICICI and HDFC Bank board outcomes").

    WHEN NOT TO USE:
      • Exact financial figures (precise revenue, margins, ratios, segment
        numbers) → use ``financials_query`` instead.
      • "Which filings exist?" (catalog metadata only) → ``stock_filings_lookup``.
      • Live price/technicals → ``stock_technicals``.

    Always preserve the returned ``[Company | p.N]`` citation strings verbatim.
    If ``needs_clarification`` is true, show the ``clarification_question`` to
    the user. If ``selected_filings[].is_scanned`` is true, note the gap.

    Args:
        question: The user's natural-language question (required). Pass it
            verbatim — the service's 6-tier company resolver handles short
            forms (TCS, RIL, L&T, M&M, HUL, SBI), 1-2 char typos
            (Relianse, Bharat Petrolium), &/and variants, punctuation
            (Dr. Reddy's), and BSE numeric scrip codes.
        company: Optional company hint — a single name/ticker (``"TCS"``),
            a list for comparisons (``["ICICI Bank", "HDFC Bank"]``), or
            omitted (the planner extracts company names from the question).
            Pass the user's input as-is; do NOT pre-resolve via
            ``lookup_company`` — the service has its own resolver.
        synthesise: ``True`` (default) → prose answer with inline
            ``[Company | p.N]`` citations for direct user display.
            ``False`` → evidence passages + bulk document excerpts only,
            for when the orchestrator blends output from multiple tools.

    Returns:
        dict with ``answer``, ``plan``, ``needs_clarification`` /
        ``clarification_question``, ``resolved_companies``,
        ``selected_filings`` (enriched), ``evidence``,
        ``candidates_considered``, ``data_freshness``.
        ``{"error": ...}`` on a transport/service failure.
    """
    payload: dict = {"question": question, "synthesise": synthesise}
    if company:
        payload["company"] = company

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
    # Trim bulky fields (document_excerpts / token_usage / timings) to keep
    # context lean, but preserve enriched metadata the agent needs.
    return {
        "answer": data.get("answer"),
        "plan": data.get("plan"),
        "needs_clarification": data.get("needs_clarification", False),
        "clarification_question": data.get("clarification_question"),
        "resolved_companies": data.get("resolved_companies"),
        "candidates_considered": data.get("candidates_considered"),
        "selected_filings": [
            {
                "company_name": f.get("company_name"),
                "headline": f.get("headline"),
                "category": f.get("category"),
                "announcement_dt": f.get("announcement_dt"),
                "read_ok": f.get("read_ok"),
                "page_count": f.get("page_count"),
                "is_scanned": f.get("is_scanned"),
                "why_selected": f.get("why_selected"),
                "sections_read": f.get("sections_read"),
            }
            for f in selected
        ],
        "evidence": data.get("evidence"),
        "data_freshness": latest_dt,
        **({"retry_count": data["retry_count"]} if "retry_count" in data else {}),
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
