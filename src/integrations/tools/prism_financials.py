"""prism-financials — the teammate-built numeric-Q&A service, as an agent tool.

The service (FastAPI) exposes one agent-relevant endpoint, ``POST /ask``: it
turns a natural-language finance question into safe, read-only Postgres SQL
over CMIE Prowess data and returns structured ``rows`` + the executed ``sql``.
This is PRISM's exact-numbers path — balance sheet, P&L, cash flow, quarterly,
shareholding, market multiples, and derived ratios (D/E, margins, CAGR, YoY,
sector rank). Narrative ("what did X *say*") still goes to ``stock_filings_read``.

We wire ONE typed tool (``financials_query``) rather than auto-generating from
an OpenAPI spec — the service has no spec yet, and the surface is a single
endpoint. (``GET /healthz`` is a liveness probe, not an agent tool.)

Base URL: ``settings.PRISM_FINANCIALS_URL`` (env ``PRISM_FINANCIALS_URL``; prod
``http://35.234.221.166:8000``). No caller auth today — the endpoint is open.
When the service adds ``X-API-Key`` auth, set ``PRISM_FINANCIALS_API_KEY`` and
the wrapper attaches the header automatically; until then it sends nothing.

The ``/ask`` contract is unusual: it ALWAYS returns HTTP 200 for four logical
shapes, with ``error: null`` on success. The wrapper branches on them:

  1. Normal       — ``rows`` non-empty, ``error`` null → pass through.
  2. Clarification — ``needs_clarification: true`` → pass through (still a
                     SUCCESS result; the agent re-asks the user). ``error`` is
                     null so ``is_error`` correctly reads it as non-error.
  3. NOT IN DATABASE refusal — ``rows: [{"note": "NOT IN DATABASE: …"}]``,
                     ``error`` null → pass through. The agent surfaces it and
                     does NOT retry or answer from its own knowledge.
  4. Error        — ``error`` is a non-empty string → convert to the standard
                     ``make_error`` shape so the agent gets a ``next_action``.
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

# Intake latency budget: recipe path 0.6-1.5s, text-to-SQL cold 8-15s, refusal
# ~12s. A single 30s ceiling covers the slow LLM-SQL path with headroom.
_TIMEOUT = 30.0


def _base_url() -> str:
    return (settings.PRISM_FINANCIALS_URL or "http://localhost:8000").rstrip("/")


def _auth_headers() -> dict[str, str]:
    """Attach X-API-Key only when the env var is set. No-op today (open
    endpoint); zero rework when the service turns auth on."""
    key = settings.PRISM_FINANCIALS_API_KEY
    return {"X-API-Key": key} if key else {}


async def _post(path: str, payload: dict, timeout: float) -> dict:
    """POST helper with transparent, graceful error reporting (Part-A).

    Transient transport failures (timeout / network error) get **one silent
    retry** after 250 ms; on retry success the response carries
    ``retry_count: 1`` so the runner can emit a ``ToolRetryEvent`` (↻ chip).
    4xx/5xx are never retried — those are bad input or upstream issues that
    won't fix themselves in 250 ms.
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
        # Defensive — both branches above return on the second failure.
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


def _latest_period_end(rows: list[dict]) -> str | None:
    """Max ISO ``period_end`` across rows — used as the data_freshness signal so
    the runner can emit a DataFreshnessEvent ("as of …")."""
    return max(
        (r.get("period_end") for r in rows if isinstance(r, dict) and r.get("period_end")),
        default=None,
    )


async def financials_query(
    question: str,
    answer_mode: str = "off",
    user_id: str | None = None,
) -> dict:
    """Answer a NUMERICAL question about an Indian listed company (CMIE Prowess)
    — exact figures, ratios, rankings, time-series, ownership %, and market
    multiples. Generates safe read-only SQL and returns structured ``rows`` plus
    the executed ``sql``. This is the EXACT-NUMBERS path.

    WHEN TO USE: the user wants a number, a ranking, a breakdown, or a trend —
      * balance sheet (assets, debt, equity, cash, reserves, working capital),
      * income statement (revenue, sales, EBITDA, PBT, PAT, margins) annual OR
        quarterly,
      * cash flow (operating / investing / financing, capex, dividends),
      * ownership (promoter / FII / DII / mutual-fund / retail %),
      * market multiples (P/E, P/B, market cap, EPS, dividend yield),
      * derived ratios (D/E, current ratio, ROCE, net profit margin, total
        debt, YoY growth, CAGR),
      * cross-company top-N / sector rankings / peer benchmarks.
    Coverage: balance sheet FY15 onward, P&L and cash flow FY17 onward,
    quarterly from Q1 FY18 onward. Indian listed companies only.

    WHEN NOT TO USE: live/intra-day or historical stock-price series (use
    ``stock_technicals`` for live), filings narrative / MD&A / what a company
    *said* (use ``stock_filings_read``), credit ratings / analyst estimates /
    forecasts, investment advice, general company facts (CEO, founding year),
    or any non-Indian company. The tool will refuse out-of-scope asks itself
    with a ``NOT IN DATABASE`` note — but routing those here wastes a round trip.

    **Pass the user's question VERBATIM.** Do not normalise case, strip aliases
    or punctuation, or reword it. The service's own resolver handles aliases
    ("HUL", "L&T", "M&M"), Indian fiscal quarters ("Q3 FY25"), and sector hints.

    Handle the four response shapes (see Returns):
      * ``needs_clarification: true`` → show ``clarification`` to the user
        verbatim and ask them to pick; do NOT choose a candidate yourself.
      * ``rows[0].note`` starts with "NOT IN DATABASE:" → surface that
        explanation; do NOT retry and do NOT answer from your own knowledge.
      * ``ok: False`` → follow ``next_action`` (the service was briefly down).
      * otherwise → render the ``rows``; cite the ``sql`` if useful.

    Args:
        question: The user's natural-language question, passed verbatim (required).
        answer_mode: "off" (default — return rows only; you write the prose),
            "table" (also get a ready markdown table), or "llm" (service writes
            analyst prose). Leave at "off" for PRISM — the agent owns the final
            cited answer.
        user_id: Optional session/user id — written to the service's audit log
            only; has no effect on the answer. Omit if you don't have one.

    Returns:
        On success, a dict with ``rows`` (the canonical data — do NOT re-rank or
        mutate before display), ``sql`` (the executed query, for citation),
        ``needs_clarification`` / ``clarification``, ``provider``,
        ``duration_ms``, and ``data_freshness`` (latest period in the rows).
        On a service-side failure returns the standard ``{"ok": False, ...}``
        error shape with a ``next_action`` hint.
    """
    payload: dict = {"question": question, "answer_mode": answer_mode, "debug": False}
    if user_id:
        payload["user_id"] = user_id

    data = await _post("/ask", payload, _TIMEOUT)
    # Transport / HTTP failure already in the standard error shape.
    if data.get("ok") is False:
        return data

    # Shape 4 — the service tried but hit a transient failure (LLM timeout, DB
    # blip). It puts a graceful message in ``answer``; we map ``error`` onto the
    # standard shape so the agent gets a structured next_action.
    if data.get("error"):
        return make_error(
            message="The financials service couldn't complete the query just now. Try again in a moment.",
            code="prism_financials_upstream_error",
            next_action="ask_user_to_retry_later",
            retriable=True,
            detail=str(data.get("error")),
        )

    # Shapes 1-3 are all SUCCESS envelopes (error is null). Trim operational
    # fields (debug, echoed answer) and keep the canonical rows + sql.
    rows = data.get("rows") or []
    out: dict = {
        "rows": rows,
        "sql": data.get("sql"),
        "needs_clarification": data.get("needs_clarification", False),
        "clarification": data.get("clarification"),
        "provider": data.get("provider"),
        "duration_ms": data.get("duration_ms"),
        "data_freshness": _latest_period_end(rows),
    }
    if "retry_count" in data:
        out["retry_count"] = data["retry_count"]
    return out


# The registry's `python` adapter wraps each plain function here in a FunctionTool.
PRISM_FINANCIALS_TOOLS = [financials_query]
