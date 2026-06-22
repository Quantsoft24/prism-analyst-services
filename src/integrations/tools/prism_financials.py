"""prism-financials — the teammate-built numeric-Q&A service, as an agent tool.

The service (FastAPI) answers a natural-language finance question about Indian
listed companies and returns a structured, **operation-typed** result plus a
human-readable ``answer``. This is PRISM's exact-numbers path — single metrics,
trends, growth/CAGR, derived ratios, whole statements (balance sheet / P&L),
head-to-head comparisons, and market-wide screening / ranking. Narrative
("what did X *say*") still goes to ``stock_filings_read``.

**Contract (security_id migration, 2026-06):** the service no longer resolves a
company from the question text — the agent resolves the company FIRST via
``resolve_company`` (clarifying if ambiguous) and passes the resulting
``security_id`` (or ``security_ids`` for a comparison). For a market-wide
screen / ranking ("top 10 NBFCs by ROE", "midcap high-ROCE low-debt") the agent
passes NEITHER id — the service detects the universe from the question.

``POST /ask`` always returns HTTP 200 with a ``status`` field:
  * ``ok``                  — answered; carries operation-specific fields.
  * ``no_data``             — no value for that company/period (not an error).
  * ``needs_clarification`` — the METRIC isn't derivable from the catalog;
                              carries ``suggestions`` (closest fields). NOT a
                              company-disambiguation (that happens upstream now).
HTTP 404 means the supplied ``security_id`` doesn't exist (resolver mismatch);
422 means ``question`` was missing.

Base URL: ``settings.PRISM_FINANCIALS_URL`` (env ``PRISM_FINANCIALS_URL``; prod
``http://35.234.221.166:8090``). No caller auth today — the endpoint is open.
When the service adds ``X-API-Key`` auth, set ``PRISM_FINANCIALS_API_KEY`` and
the wrapper attaches the header automatically.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from src.config import settings
from src.integrations.tools._errors import make_error

logger = logging.getLogger(__name__)

# One-shot retry on transient transport failures — matches stock_chat._post /
# bmc._request. 250ms clears most TCP/DNS/load-spike blips without feeling slow.
_RETRY_DELAY_S = 0.25

# Latency budget (intake): single-company lookups/trends ~1-4s, comparisons a
# few s, rankings ~5-15s, screens ~5-40s (first market-wide screen after a
# restart ~30s). A 75s ceiling covers the slowest screen with headroom.
_TIMEOUT = 75.0


def _base_url() -> str:
    return (settings.PRISM_FINANCIALS_URL or "http://localhost:8090").rstrip("/")


def _auth_headers() -> dict[str, str]:
    """Attach X-API-Key only when the env var is set. No-op today (open
    endpoint); zero rework when the service turns auth on."""
    key = settings.PRISM_FINANCIALS_API_KEY
    return {"X-API-Key": key} if key else {}


async def _post(path: str, payload: dict, timeout: float) -> dict:
    """POST helper with transparent, graceful error reporting (Part-A).

    Transient transport failures (timeout / network) get one silent retry after
    250 ms; on retry success the response carries ``retry_count: 1`` so the
    runner can emit a ToolRetryEvent (↻ chip). 4xx/5xx are never retried.
    """
    url = f"{_base_url()}{path}"
    headers = _auth_headers()
    last_exc: Exception | None = None
    retry_count = 0
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
            if attempt == 2:
                retry_count = 1
            break
        except httpx.TimeoutException as exc:
            last_exc = exc
            if attempt == 1:
                logger.warning(
                    "prism-financials timed out at %s (attempt %d) — retrying in %sms",
                    url, attempt, int(_RETRY_DELAY_S * 1000),
                )
                await asyncio.sleep(_RETRY_DELAY_S)
                continue
            logger.warning("prism-financials timed out at %s after retry: %s", url, exc)
            return make_error(
                message="The financials service timed out. Try again in a moment, or simplify the question.",
                code="prism_financials_timeout",
                next_action="ask_user_to_retry_later",
                retriable=True,
                detail=str(exc),
            )
        except httpx.RequestError as exc:
            last_exc = exc
            if attempt == 1:
                logger.warning(
                    "prism-financials unreachable at %s (attempt %d) — retrying in %sms",
                    url, attempt, int(_RETRY_DELAY_S * 1000),
                )
                await asyncio.sleep(_RETRY_DELAY_S)
                continue
            logger.warning("prism-financials unreachable at %s after retry: %s", url, exc)
            return make_error(
                message="The financials service is unreachable. The data team has been notified.",
                code="prism_financials_unreachable",
                next_action="ask_user_to_retry_later",
                retriable=True,
                detail=f"{path}: {exc}",
            )
    else:
        return make_error(
            message="The financials service is unreachable.",
            code="prism_financials_unreachable",
            next_action="ask_user_to_retry_later",
            retriable=True,
            detail=str(last_exc),
        )

    if resp.status_code >= 500:
        return make_error(
            message="The financials service returned an internal error.",
            code=f"prism_financials_http_{resp.status_code}",
            next_action="ask_user_to_retry_later",
            retriable=True,
            detail=resp.text,
        )
    if resp.status_code == 404:
        # The supplied security_id doesn't exist in the financials DB — almost
        # always a resolver mismatch. Ask the user to re-pick rather than retry.
        return make_error(
            message="That company id wasn't found in the financials database — please re-pick the company.",
            code="prism_financials_security_not_found",
            next_action="ask_user_to_clarify",
            retriable=False,
            detail=resp.text,
        )
    if resp.status_code == 422:
        return make_error(
            message="The financials service rejected the request — the question may need more detail.",
            code="prism_financials_bad_request",
            next_action="ask_user_to_clarify",
            retriable=False,
            detail=resp.text,
        )
    if resp.status_code != 200:
        return make_error(
            message=f"The financials service returned HTTP {resp.status_code}.",
            code=f"prism_financials_http_{resp.status_code}",
            next_action="try_alternate_tool",
            retriable=False,
            detail=resp.text,
        )
    data = resp.json()
    if retry_count and isinstance(data, dict):
        data["retry_count"] = retry_count
    return data


def _data_freshness(data: dict) -> str | None:
    """Latest period covered by the answer — drives the DataFreshnessEvent
    ("as of FY2024"). Single-period ops carry ``period``; trends carry a
    ``series`` of ``{period, value}``; comparisons carry per-company periods."""
    periods: list[str] = []
    if isinstance(data.get("period"), str):
        periods.append(data["period"])
    for row in data.get("series") or []:
        if isinstance(row, dict) and isinstance(row.get("period"), str):
            periods.append(row["period"])
    for row in data.get("comparison") or []:
        if isinstance(row, dict) and isinstance(row.get("period"), str):
            periods.append(row["period"])
    # FY/quarter labels sort lexate-correctly enough for "latest" (FY2025 > FY2024).
    return max(periods, default=None)


# Structured fields worth keeping for the deterministic frontend render + the
# composer. Operation-specific arrays are passed through untouched (the service
# is canonical — do NOT re-rank or mutate). ``provenance`` (SQL) is intentionally
# dropped: there's no SQL viewer, and it bloats the agent's context.
_KEEP_FIELDS = (
    "operation", "answer", "value", "period", "field", "company",
    "series", "comparison", "ranking", "matches", "line_items",
    "attributes", "count", "names", "results", "note",
)


async def financials_query(
    question: str,
    security_id: int | None = None,
    security_ids: list[int] | None = None,
) -> dict:
    """Answer a NUMERICAL question about Indian listed companies (fundamentals,
    ratios, valuation, statements, screening, rankings) — exact figures with
    structured, operation-typed results. This is the EXACT-NUMBERS path.

    WHEN TO USE: the user wants a number, a ratio, a trend, a comparison, a whole
    statement, or a screen/ranking:
      * single metric — revenue, PAT, margins, total assets, deposits, borrowings
        (FY or quarter); trends / YoY / QoQ / CAGR over years/quarters,
      * derived ratios — ROE, ROA, ROCE, D/E, current/quick ratio, interest
        coverage, P/E, P/B, EV/EBITDA, EPS, book value/share, working capital,
      * whole statements — balance sheet, P&L / income statement, key ratios,
      * comparisons — same metric across 2+ companies,
      * screening / ranking — "top 10 NBFCs by ROE", "midcap high-ROCE low-debt
        companies", "Nifty 50 with P/E > 60", "how many pharma companies".
    Indian listed companies only.

    WHEN NOT TO USE: live/intra-day price series (use ``stock_technicals``),
    filings narrative / what a company *said* (use ``stock_filings_read``),
    business-model overview (use ``bmc_*``), news/sentiment (use ``news_*``),
    analyst estimates / forecasts / advice, or non-Indian companies.

    **RESOLVE THE COMPANY FIRST.** The service no longer resolves names — it
    needs the ``master_securities`` ``security_id``:
      * ONE company  → call ``resolve_company`` first, pass its ``security_id``.
      * COMPARISON   → resolve EACH company, pass all ids as ``security_ids``
        (takes precedence over ``security_id``).
      * SCREEN / RANK / market-wide (no specific company named) → pass NEITHER
        id; the service infers the universe (sector / index / size band / all)
        from the question.
    Pass the user's ``question`` VERBATIM (the service parses the metric, period,
    operation, and any screen criteria itself — do not normalise or pre-fill).

    Handle the returned ``status``:
      * ``needs_clarification`` → the METRIC isn't in the catalog; show ``answer``
        and the ``suggestions`` (closest fields) and ask which the user wants.
        Do NOT answer from your own knowledge.
      * ``no_data`` → there's no value for that company/period; relay ``answer``.
      * otherwise (``ok``) → relay ``answer`` and render the structured fields.
    On a service failure the standard ``{"ok": False, ...}`` shape is returned
    with a ``next_action`` hint.

    Args:
        question: The user's natural-language question, verbatim (required).
        security_id: ``master_securities`` id from ``resolve_company`` for a
            SINGLE company. Omit for screens/rankings.
        security_ids: ids for a multi-company COMPARISON (resolve each first).
            Takes precedence over ``security_id``. Omit for screens/rankings.

    Returns:
        dict with ``status``, ``operation``, ``answer`` (NL), and the
        operation-specific fields: ``value``/``period``/``field``/``company``
        (lookup), ``series`` (trend), ``comparison`` (compare), ``ranking`` /
        ``matches`` (rank/screen), ``line_items`` (statement), ``attributes`` /
        ``count`` / ``names`` (classification), plus ``data_freshness``. On a
        clarification: ``needs_clarification: True`` + ``suggestions``.
    """
    payload: dict = {"question": question}
    if security_ids:
        payload["security_ids"] = security_ids
    elif security_id is not None:
        payload["security_id"] = security_id

    data = await _post("/ask", payload, _TIMEOUT)
    # Transport / HTTP failure already in the standard error shape.
    if data.get("ok") is False:
        return data

    status = data.get("status")

    # Metric-level clarification — the company is already resolved; this means
    # the requested METRIC isn't derivable. Surface it + the closest fields.
    if status == "needs_clarification":
        out: dict = {
            "status": "needs_clarification",
            "needs_clarification": True,
            "answer": data.get("answer"),
            "suggestions": data.get("suggestions") or [],
        }
        if "retry_count" in data:
            out["retry_count"] = data["retry_count"]
        return out

    # ok / no_data are both success envelopes. Keep the canonical structured
    # fields (operation-specific arrays pass through untouched); drop bulky/
    # irrelevant ones (provenance SQL, question echo, granularity, data_type).
    out = {k: data[k] for k in _KEEP_FIELDS if k in data and data[k] is not None}
    out["status"] = status or "ok"
    out["data_freshness"] = _data_freshness(data)
    if "retry_count" in data:
        out["retry_count"] = data["retry_count"]
    return out


# The registry's `python` adapter wraps each plain function here in a FunctionTool.
PRISM_FINANCIALS_TOOLS = [financials_query]
