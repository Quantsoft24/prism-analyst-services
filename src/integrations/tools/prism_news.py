"""prism-news — the teammate-built Indian financial-news + sentiment service.

External FastAPI service (showtimeapp/NewsRSS) at ``settings.PRISM_NEWS_URL``
(env ``PRISM_NEWS_URL``; prod ``http://<gcp-host>:8001``). 82 Indian RSS feeds,
OpenAI sentiment scoring, and a 4,149-company alias master. The service offers
both REST and an MCP endpoint; we wire it as a ``python`` typed wrapper (NOT
``openapi`` / ``mcp``) — same convention as stock-chat / bmc / prism-financials.
Rationale: auto-generating from the OpenAPI spec would expose ~9 tools
(including ops-only ``/health`` / ``/stats``) and clutter the LLM's context.
A typed wrapper exposes only the 3-4 the agent actually needs, trims bulky
responses, and inherits the structured error contract (see _errors.py).

We expose **4 tools** over the REST surface:

  * ``news_sentiment``  → GET /news/summary    (per-company verdict + trend)
  * ``news_trending``   → GET /news/trending    (most-mentioned companies)
  * ``news_search``     → GET /news             (article list, company/sector)
  * ``news_compare``    → GET /news/compare      (multi-company side-by-side)

Coverage limits (surface these to the user, don't fabricate around them):
  * Indian NSE/BSE-listed names only — alias master is India-only.
  * 10-day (240h) max window; older docs exist upstream but aren't exposed.
  * Not for live prices/quotes (this is news), pre-IPO/private names, or
    non-Indian equities.

Latency reality (drives our timeouts): a company's FIRST sentiment query in a
fresh window triggers OpenAI scoring on up to 30 articles (cold 5-10s, worst
~12s). Subsequent calls are Mongo-cached and sub-second. Trending / headline
lists make zero LLM calls and are fast. Prod is occasionally bouncy mid-fetch
(a rare 503) — we do one silent retry, same as the other wrappers.

Base URL: ``settings.PRISM_NEWS_URL``. No caller auth today; when a gateway is
added, set ``PRISM_NEWS_API_KEY`` and the wrapper attaches ``X-API-Key``.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from src.config import settings
from src.integrations.tools._errors import make_error

logger = logging.getLogger(__name__)

# One-shot retry on transient transport blips — matches stock_chat._post /
# bmc._request / prism_financials._post. 250ms clears most TLS/DNS/load-spike
# blips. The intake notes prod is occasionally bouncy mid-fetch (a rare 503);
# 5xx is NOT retried here (it won't fix itself in 250ms) but the agent gets a
# retriable error so the runner's middleware can re-invoke once.
_RETRY_DELAY_S = 0.25

# Company-sentiment queries fire cold OpenAI scoring on first ask for a fresh
# company+window. The intake claimed ~12s worst-case, but live measurement
# showed the cold /news/summary path running PAST 20s while scoring ~24
# articles (it then caches → sub-second on every later call). So the first
# question about any company was timing out. We give the sentiment/compare
# paths a 40s ceiling — comfortably above observed cold latency, and still
# under the agent runner's 60s overall cap (a single cold news turn + the
# answer synthesis fits). Crucially, timeouts are NOT retried (see _get):
# a retry would double the wait to 80s and blow the 60s agent budget, and the
# server keeps computing + caching anyway, so the user's next ask is instant.
_SENTIMENT_TIMEOUT = 40.0
_COMPARE_TIMEOUT = 40.0
_FAST_TIMEOUT = 15.0

# Hard caps to keep tool inputs sane (and protect the upstream).
_MAX_HOURS = 240            # service exposes a 10-day window only
_DEFAULT_HOURS = 24
_VALID_SECTORS = frozenset({
    "BANKING", "TECH", "AUTO", "PHARMA", "ENERGY", "FMCG", "METALS", "REALTY",
})


def _base_url() -> str:
    return (settings.PRISM_NEWS_URL or "http://localhost:8014").rstrip("/")


def _auth_headers() -> dict[str, str]:
    """Attach X-API-Key only when the env var is set. No-op today (open
    endpoint); zero rework when a gateway is added."""
    key = settings.PRISM_NEWS_API_KEY
    return {"X-API-Key": key} if key else {}


def _clamp_hours(hours: int | None) -> int:
    """Clamp the window to the service's supported 1-240h range."""
    if not isinstance(hours, int) or hours <= 0:
        return _DEFAULT_HOURS
    return min(hours, _MAX_HOURS)


def _clamp_limit(limit: int | None, default: int, cap: int) -> int:
    """Clamp a result-count limit to [1, cap], falling back to ``default``."""
    if not isinstance(limit, int) or limit <= 0:
        return default
    return min(limit, cap)


async def _get(path: str, params: dict, timeout: float) -> dict:
    """GET helper with transparent, graceful error reporting (Part-A).

    Retry policy is split by failure kind:

      * **Timeout** → NOT retried. The sentiment path can legitimately take
        ~30-40s on a cold OpenAI scoring pass; retrying immediately just
        doubles the wait (and would blow the agent runner's 60s cap). The
        upstream keeps computing + caching server-side regardless, so the
        user's NEXT ask hits a warm cache and is instant. We surface a
        friendly "still gathering — ask again in a moment" message.
      * **Transport error** (connect/DNS/reset) → **one silent retry** after
        250 ms (a true transient blip). On retry success the response carries
        ``retry_count: 1`` so the runner can emit a ToolRetryEvent.
      * **4xx/5xx** → never retried (bad input / upstream issue).
    """
    url = f"{_base_url()}{path}"
    headers = _auth_headers()
    retry_count = 0

    # Timeouts: single attempt, no retry (see docstring).
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, params=params, headers=headers)
    except httpx.TimeoutException as exc:
        logger.warning("prism-news timed out at %s after %ss: %s", url, timeout, exc)
        return make_error(
            message=(
                "The news service is still gathering sentiment for that "
                "(it scores fresh articles on the first request). Ask again "
                "in a few seconds — it will be ready and instant."
            ),
            code="prism_news_timeout",
            next_action="ask_user_to_retry_later",
            retriable=True,
            detail=f"{path}: timed out after {timeout}s (cold-scoring)",
        )
    except httpx.RequestError:
        # Transport blip — one retry after a short pause.
        logger.warning("prism-news transport error at %s — retrying in %sms", url, int(_RETRY_DELAY_S * 1000))
        await asyncio.sleep(_RETRY_DELAY_S)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, params=params, headers=headers)
            retry_count = 1
        except httpx.TimeoutException as exc:
            logger.warning("prism-news timed out at %s on retry: %s", url, exc)
            return make_error(
                message=(
                    "The news service is still gathering sentiment for that. "
                    "Ask again in a few seconds — it will be ready."
                ),
                code="prism_news_timeout",
                next_action="ask_user_to_retry_later",
                retriable=True,
                detail=f"{path}: timed out after {timeout}s on retry",
            )
        except httpx.RequestError as exc:
            logger.warning("prism-news unreachable at %s after retry: %s", url, exc)
            return make_error(
                message="The news service is unreachable. The data team has been notified.",
                code="prism_news_unreachable",
                next_action="ask_user_to_retry_later",
                retriable=True,
                detail=f"{path}: {exc}",
            )

    if resp.status_code == 404:
        return make_error(
            message="The news service endpoint was not found (404 — the URL looks misconfigured).",
            code="prism_news_http_404",
            next_action="ask_user_to_retry_later",
            retriable=False,
            detail=f"GET {url} returned 404. Check PRISM_NEWS_URL env var.",
        )
    if resp.status_code >= 500:
        return make_error(
            message="The news service returned an internal error.",
            code=f"prism_news_http_{resp.status_code}",
            next_action="ask_user_to_retry_later",
            retriable=True,
            detail=resp.text,
        )
    if resp.status_code in (400, 422):
        return make_error(
            message="The news service rejected the request — the company or sector may be unrecognised.",
            code="prism_news_bad_request",
            next_action="ask_user_to_clarify",
            retriable=False,
            detail=resp.text,
        )
    if resp.status_code != 200:
        return make_error(
            message=f"The news service returned HTTP {resp.status_code}.",
            code=f"prism_news_http_{resp.status_code}",
            next_action="try_alternate_tool",
            retriable=False,
            detail=resp.text,
        )
    try:
        data = resp.json()
    except (ValueError, TypeError) as exc:
        return make_error(
            message="The news service returned a malformed response.",
            code="prism_news_bad_payload",
            next_action="try_alternate_tool",
            retriable=False,
            detail=str(exc),
        )
    if not isinstance(data, dict):
        return make_error(
            message="The news service returned an unexpected response shape.",
            code="prism_news_bad_payload",
            next_action="try_alternate_tool",
            retriable=False,
            detail=str(data)[:200],
        )
    if retry_count:
        data["retry_count"] = retry_count
    return data


def _trim_article(a: dict) -> dict:
    """Keep only the fields the LLM needs to cite an article; drop bulky/raw
    fields (description blobs, original_link, internal ids)."""
    if not isinstance(a, dict):
        return {}
    sentiment = a.get("sentiment")
    trimmed_sentiment = None
    if isinstance(sentiment, dict):
        trimmed_sentiment = {
            "label": sentiment.get("label"),
            "score": sentiment.get("score"),
            "provider": sentiment.get("provider"),
        }
    return {
        "title": a.get("title"),
        "source": a.get("source"),
        "published_ist": a.get("published_ist"),
        "link": a.get("link"),
        "companies": a.get("companies"),
        "sector": a.get("sector"),
        "sentiment": trimmed_sentiment,
    }


def _trim_headline(a: dict) -> dict:
    """Even leaner than _trim_article — for top_positive / top_negative lists
    inside a summary (title + source + link is enough to cite)."""
    if not isinstance(a, dict):
        return {}
    return {
        "title": a.get("title"),
        "source": a.get("source"),
        "link": a.get("link"),
    }


async def news_sentiment(company: str, hours: int = 24) -> dict:
    """Get the market-news SENTIMENT VERDICT for ONE Indian listed company —
    bullish / bearish / neutral, with the trend, article counts, and the
    strongest positive + negative headlines. Backed by OpenAI sentiment over
    live RSS feeds.

    WHEN TO USE: the user asks how a company is doing in the news, the market
    mood / reaction / outlook on a stock, whether sentiment is positive, or
    "what's the news on X" expecting a verdict ("How is Reliance doing today?",
    "Is TCS bullish?", "What's the sentiment on HDFC Bank?").

    WHEN NOT TO USE: exact financial figures (use ``financials_query``), what a
    filing SAID (use ``stock_filings_read``), live prices (use
    ``stock_technicals``), or non-Indian companies (coverage is India-only).

    Accepts aliases — "HDFC", "HDFC Bank", "HDFC Ltd" all resolve to HDFC Bank;
    "Reliance" → Reliance Industries; "TCS" → Tata Consultancy Services. Pass
    the company as the user said it; the service's alias master resolves it.

    LATENCY: a company's first query in a fresh window can take 5-10s while
    OpenAI scores articles; repeat calls are sub-second (cached).

    Args:
        company: Company name or alias (required). Single company only.
        hours: Look-back window, 1-240 (default 24). Clamped to that range.

    Returns:
        dict with ``company`` (resolved canonical name), ``trend``
        (bullish | bearish | neutral), ``avg_score``, ``total_articles``,
        ``sentiment_breakdown`` ({positive, negative, neutral}), ``trend_detail``
        (recent vs older half of the window), ``top_positive`` / ``top_negative``
        (headline + source + link), and ``provider`` ("openai" or "heuristic" —
        a "heuristic" verdict is directional but lower confidence; note that to
        the user). ``total_articles: 0`` means no recent news — tell the user
        plainly, do not fabricate. On a transport/service failure returns the
        standard ``{"ok": False, ...}`` error shape.
    """
    if not company or not company.strip():
        return make_error(
            message="Need a company name to look up news sentiment.",
            code="prism_news_missing_company",
            next_action="ask_user_to_clarify",
        )
    params = {"company": company.strip(), "hours": _clamp_hours(hours)}
    data = await _get("/news/summary", params, _SENTIMENT_TIMEOUT)
    if data.get("ok") is False:
        return data
    out = {
        "company": data.get("company") or company.strip(),
        "input": data.get("input"),
        "trend": data.get("trend"),
        "avg_score": data.get("avg_score"),
        "total_articles": data.get("total_articles", 0),
        "sentiment_breakdown": data.get("sentiment_breakdown"),
        "trend_detail": data.get("trend_detail"),
        "top_positive": [_trim_headline(a) for a in (data.get("top_positive") or [])][:3],
        "top_negative": [_trim_headline(a) for a in (data.get("top_negative") or [])][:3],
        "provider": data.get("provider"),
        "data_freshness": "live",
    }
    if "retry_count" in data:
        out["retry_count"] = data["retry_count"]
    return out


async def news_trending(hours: int = 24, limit: int = 10) -> dict:
    """Get the MOST-MENTIONED Indian listed companies in the news right now,
    each with aggregate sentiment and sector. Zero LLM cost; fast.

    WHEN TO USE: the user asks what's trending / hot / moving / in the news in
    Indian markets, or wants a market-pulse overview ("What's trending?",
    "What's moving in the markets today?", "What's in the news?").

    Args:
        hours: Look-back window, 1-240 (default 24). Clamped.
        limit: How many companies to return, default 10 (capped at 50).

    Returns:
        dict with ``hours`` and ``trending`` — a list of
        {company, mentions, sentiment (positive|negative|neutral),
        sentiment_breakdown, sector (8-chip code or null)}, ranked by mention
        count. Empty ``trending`` means a quiet window. On failure returns the
        standard error shape.
    """
    params = {"hours": _clamp_hours(hours), "limit": _clamp_limit(limit, 10, 50)}
    data = await _get("/news/trending", params, _FAST_TIMEOUT)
    if data.get("ok") is False:
        return data
    out = {
        "hours": data.get("hours", params["hours"]),
        "trending": data.get("trending") or [],
        "data_freshness": "live",
    }
    if "retry_count" in data:
        out["retry_count"] = data["retry_count"]
    return out


async def news_search(
    company: str | None = None,
    sector: str | None = None,
    hours: int = 24,
    limit: int = 20,
) -> dict:
    """List recent NEWS HEADLINES for an Indian listed company or sector. Use
    when the user wants a LIST of articles (not a verdict or a trend).

    WHEN TO USE: "Show me banking news", "Latest headlines on Tata Motors",
    "Any pharma news today?", "Recent news on Adani group" (pass the names as a
    comma-separated string in ``company``).

    WHEN NOT TO USE: a sentiment verdict on one company (use ``news_sentiment``);
    "what's trending" (use ``news_trending``).

    Args:
        company: Company name(s). Single name, or comma-separated for a group
            ("Adani Enterprises,Adani Ports,Adani Green Energy"). Aliases ok.
        sector: One of BANKING | TECH | AUTO | PHARMA | ENERGY | FMCG | METALS |
            REALTY. Invalid sectors are dropped (the call still runs unfiltered).
        hours: Look-back window 1-240 (default 24). Clamped.
        limit: Max articles (default 20, capped at 100).

    Returns:
        dict with ``total`` (full match count), ``returned``, and ``articles``
        (each: title, source, published_ist, link, companies, sector, sentiment
        — sentiment may be null on non-company queries; that's expected, not an
        error). ``total: 0`` → no news; tell the user, don't fabricate. On
        failure returns the standard error shape.
    """
    params: dict = {"hours": _clamp_hours(hours), "limit": _clamp_limit(limit, 20, 100)}
    if company and company.strip():
        params["company"] = company.strip()
    if sector and sector.strip():
        sec = sector.strip().upper()
        if sec in _VALID_SECTORS:
            params["sector"] = sec
        else:
            logger.info("prism-news: ignoring unrecognised sector %r", sector)
    data = await _get("/news", params, _FAST_TIMEOUT)
    if data.get("ok") is False:
        return data
    meta = data.get("meta") or {}
    articles = [_trim_article(a) for a in (data.get("articles") or [])]
    latest = next(
        (a.get("published_ist") for a in articles if a.get("published_ist")),
        None,
    )
    out = {
        "total": meta.get("total_results", len(articles)),
        "returned": meta.get("returned", len(articles)),
        "articles": articles,
        "sentiment_provider": meta.get("sentiment_provider"),
        "data_freshness": latest or "live",
    }
    if "retry_count" in data:
        out["retry_count"] = data["retry_count"]
    return out


async def news_compare(companies: list[str] | str, hours: int = 48) -> dict:
    """Compare NEWS SENTIMENT across MULTIPLE Indian listed companies, side by
    side, ranked best-to-worst by sentiment score. Use for peer mood checks.

    WHEN TO USE: "Compare sentiment on HDFC, ICICI and SBI", "Which IT stock has
    the best news right now — TCS, Infosys or Wipro?".

    Args:
        companies: A list of names, or a comma-separated string. Aliases ok.
            At least one required.
        hours: Look-back window 1-240 (default 48). Clamped.

    Returns:
        dict with ``companies`` echoed and ``comparison`` — each company's
        summary (trend, breakdown, avg_score), ranked by avg_score. On failure
        returns the standard error shape.
    """
    if isinstance(companies, str):
        names = [c.strip() for c in companies.split(",") if c.strip()]
    elif isinstance(companies, list):
        names = [str(c).strip() for c in companies if str(c).strip()]
    else:
        names = []
    if not names:
        return make_error(
            message="Need at least one company to compare news sentiment.",
            code="prism_news_missing_company",
            next_action="ask_user_to_clarify",
        )
    params = {"companies": ",".join(names), "hours": _clamp_hours(hours)}
    data = await _get("/news/compare", params, _COMPARE_TIMEOUT)
    if data.get("ok") is False:
        return data
    out = {
        "companies": names,
        "comparison": data.get("comparison") or data.get("results") or data.get("companies"),
        "data_freshness": "live",
    }
    if "retry_count" in data:
        out["retry_count"] = data["retry_count"]
    return out


# The registry's `python` adapter wraps each plain function here in a FunctionTool.
PRISM_NEWS_TOOLS = [news_sentiment, news_trending, news_search, news_compare]
