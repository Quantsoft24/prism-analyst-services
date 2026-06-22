"""bmc — the teammate-built Business Model Canvas service, as agent tools.

External FastAPI service at ``settings.BMC_URL`` (env ``BMC_URL``; prod
``http://35.234.221.166:8012``). Read-on-demand: no RAG, no embeddings, no
chunks. Built around the shared filings catalog (``filings_index``) — the
service picks the right filing(s) by metadata, downloads them, and emits a
9-block canvas with inline ``[Company | p.N]`` citations.

We expose **6 tools** mapping the agent-routable endpoints (the export
endpoint is a binary download — surfaced through the UI proxy, not as an
agent tool):

  * ``bmc_get``           → GET  /bmc/{ticker}                — preferred first
  * ``bmc_generate``      → POST /bmc/{ticker}/run            — refresh / first time
  * ``bmc_library``       → GET  /bmc/{ticker}/library
  * ``bmc_get_version``   → GET  /bmc/{ticker}/{version}
  * ``bmc_block_chat``    → POST /bmc/{ticker}/blocks/{block_id}/chat
  * ``bmc_diff``          → POST /bmc/{ticker}/diff           — temporal diff

Caching hint baked into the docstrings: the agent should try ``bmc_get`` first
and fall back to ``bmc_generate`` only when there's no cached canvas — saves
~$0.014 and ~25s per warm hit.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from src.config import settings
from src.integrations.tools._errors import make_error

logger = logging.getLogger(__name__)

# Per the intake: cold /run + /diff can take up to ~150s; other endpoints are
# DB-only and snappy. We pick 180s for the heavy ones, 30s for the rest.
_HEAVY_TIMEOUT = 180.0
_LIGHT_TIMEOUT = 30.0
# One-shot retry on transient transport blips — matches stock_chat._post.
_RETRY_DELAY_S = 0.25

# Canonical block IDs (in the order the service emits them).
BMC_BLOCK_IDS = (
    "customer_segments",
    "value_propositions",
    "channels",
    "customer_relationships",
    "revenue_streams",
    "key_resources",
    "key_activities",
    "key_partnerships",
    "cost_structure",
)


def _base_url() -> str:
    return (settings.BMC_URL or "http://localhost:8012").rstrip("/")


async def _request(method: str, path: str, *, timeout: float, body: dict | None = None,
                   params: dict | None = None) -> dict:
    """HTTP helper with transparent, graceful error reporting (Part-A).

    Transient transport failures (timeout / network) get **one silent
    retry** after 250 ms — same pattern as ``stock_chat._post``. On
    retry success the response carries ``retry_count: 1`` for the
    runner's ToolRetryEvent emission. 4xx + 5xx responses are NOT
    retried (those are either bad input or upstream issues that won't
    fix themselves in 250 ms).
    """
    url = f"{_base_url()}{path}"
    # BMC is a FIRM-WIDE shared library — pin every call to the single shared
    # firm_id (NOT this run's per-user firm) so the agent reads/writes the same
    # pool the /bmc page shows to everyone. Matches the proxy's get_bmc_firm_id.
    # GET → query param; POST → body.
    firm_id = settings.BMC_SHARED_FIRM_ID
    if firm_id:
        if method.upper() == "GET":
            params = {**(params or {}), "firm_id": firm_id}
        else:
            body = {**(body or {}), "firm_id": firm_id}
    retry_count = 0
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.request(method, url, json=body, params=params)
            if attempt == 2:
                retry_count = 1
            break
        except httpx.TimeoutException as exc:
            if attempt == 1:
                logger.warning(
                    "BMC service timed out at %s (attempt %d) — retrying in %sms",
                    url, attempt, int(_RETRY_DELAY_S * 1000),
                )
                await asyncio.sleep(_RETRY_DELAY_S)
                continue
            logger.warning("BMC service timed out at %s after retry: %s", url, exc)
            return make_error(
                message="The BMC service timed out building the canvas. Try again in a moment.",
                code="bmc_timeout",
                next_action="ask_user_to_retry_later",
                retriable=True,
                detail=str(exc),
            )
        except httpx.RequestError as exc:
            if attempt == 1:
                logger.warning(
                    "BMC service unreachable at %s (attempt %d) — retrying in %sms",
                    url, attempt, int(_RETRY_DELAY_S * 1000),
                )
                await asyncio.sleep(_RETRY_DELAY_S)
                continue
            logger.warning("BMC service unreachable at %s after retry: %s", url, exc)
            return make_error(
                message="The BMC service is unreachable.",
                code="bmc_unreachable",
                next_action="ask_user_to_retry_later",
                retriable=True,
                detail=f"{method} {path}: {exc}",
            )

    if resp.status_code == 404:
        return make_error(
            message="No business model canvas exists yet for this ticker. Use bmc_generate to build one.",
            code="bmc_not_found",
            next_action="try_alternate_tool",
            retriable=False,
            detail=resp.text,
        )
    if resp.status_code >= 500:
        return make_error(
            message="The BMC service returned an internal error.",
            code=f"bmc_http_{resp.status_code}",
            next_action="ask_user_to_retry_later",
            retriable=True,
            detail=resp.text,
        )
    if resp.status_code != 200:
        return make_error(
            message=f"The BMC service returned HTTP {resp.status_code}.",
            code=f"bmc_http_{resp.status_code}",
            next_action="give_up_gracefully",
            retriable=False,
            detail=resp.text,
        )
    data = resp.json()
    if retry_count and isinstance(data, dict):
        data["retry_count"] = retry_count
    return data


def _trim_bmc(data: dict) -> dict:
    """Strip bulky operational fields from a full BMC response so the LLM gets
    the model output, not the service's diagnostics."""
    if data.get("ok") is False or "error" in data:
        return data
    return {
        "ticker": data.get("ticker"),
        "company_name": data.get("company_name"),
        "version": data.get("version"),
        "status": data.get("status"),
        "overall_confidence": data.get("overall_confidence"),
        "blocks": data.get("blocks"),
        "selected_filings": [
            {
                "slot": f.get("slot"),
                "category": f.get("category"),
                "announcement_dt": f.get("announcement_dt"),
                "page_count": f.get("page_count"),
                # Keep the source-PDF link so the UI can render "View full source
                # filing" and so evidence markers deep-link to the cited page.
                "pdf_url": f.get("pdf_url"),
            }
            for f in (data.get("selected_filings") or [])
        ],
        "gaps": data.get("gaps"),
        "needs_clarification": data.get("needs_clarification"),
        "clarification": data.get("clarification"),
    }


async def bmc_get(ticker: str) -> dict:
    """Fetch the LATEST cached Business Model Canvas for an Indian NSE/BSE-listed
    company — cheap (no LLM, ~100 ms). ALWAYS try this FIRST when the user asks
    for "the BMC of X" / "@bmc X" / "business overview of X". Only call
    ``bmc_generate`` when this returns an error indicating no canvas exists.

    WHEN TO USE: the user asks what a company does, how it makes money, who
    its customers/partners/channels are, what its cost drivers are; "business
    model" / "business overview" / "BMC <ticker>"; canvas exists already.

    WHEN NOT TO USE: exact financial numbers (those go via PRISM's
    compute_*/NRE); a one-shot question about a single filing (use
    stock_filings_read).

    Args:
        ticker: ticker (e.g. "TCS"), NSE symbol ("HDFCBANK"), partial name
            ("Tata Consultancy"), or full canonical name. The service's fuzzy
            resolver picks the best match.

    Returns:
        Trimmed BMC dict with ``ticker``, ``company_name``, ``version``,
        ``status`` (complete | partial | no_evidence | failed),
        ``overall_confidence``, the 9 ``blocks`` (each with
        ``summary_bullets``, ``status``, ``confidence``, and
        ``evidence`` rows containing ``marker``, ``newsid``, ``page``,
        ``excerpt``), ``selected_filings``, and ``gaps``/``needs_clarification``
        if any. On error returns ``{"error": ...}`` — typically meaning no
        canvas exists yet (call ``bmc_generate`` then).
    """
    data = await _request("GET", f"/bmc/{ticker}", timeout=_LIGHT_TIMEOUT)
    return _trim_bmc(data)


async def bmc_generate(
    ticker: str,
    fiscal_period: str | None = None,
    security_id: int | None = None,
) -> dict:
    """Generate and persist a NEW Business Model Canvas (immutable new version).
    Cold ~25–35 s; ~$0.014 in upstream LLM cost. Use this ONLY when a canvas
    doesn't exist yet, or the user explicitly asks to "refresh" / "regenerate".

    Resolve the company FIRST via ``resolve_company`` and pass its ``security_id``
    here — this pins the EXACT NSE/BSE entity (matched against
    ``filings_index.security_id_bse OR security_id_nse``) and skips the service's
    fuzzy resolver, so we never silently build the canvas for the wrong "Reliance".
    Pass the resolved canonical symbol (e.g. ``"RELIANCE"``) as ``ticker`` — never
    the user's raw term.

    Args:
        ticker: resolved symbol / canonical name. Used as the persistence key.
        fiscal_period: optional FY anchor (e.g. ``"2025"`` for FY24-25). When
            omitted, the latest Annual Report is used.
        security_id: integer fast-path from ``resolve_company`` — bypasses the
            fuzzy ticker resolver.

    Returns:
        Same trimmed BMC shape as ``bmc_get``, with a fresh ``version``.
    """
    body: dict = {}
    if fiscal_period:
        body["fiscal_period"] = fiscal_period
    if security_id is not None:
        body["security_id"] = security_id
    data = await _request("POST", f"/bmc/{ticker}/run", body=body, timeout=_HEAVY_TIMEOUT)
    return _trim_bmc(data)


async def bmc_library(ticker: str) -> dict:
    """List every saved BMC version (header-only) for a company. Cheap, no LLM.
    Use when the user asks "show me all versions / history of X's BMC".

    Returns:
        ``{"versions": [{"version", "fiscal_period", "status",
        "overall_confidence", "created_at", "bmc_id"}]}`` or ``{"error": ...}``.
    """
    data = await _request("GET", f"/bmc/{ticker}/library", timeout=_LIGHT_TIMEOUT)
    if data.get("ok") is False or "error" in data:
        return data
    # The upstream may return a bare list or a dict — normalize.
    if isinstance(data, list):
        return {"versions": data}
    return data


async def bmc_get_version(ticker: str, version: int) -> dict:
    """Fetch one SPECIFIC saved BMC version (not latest). Use when the user
    asks about a historical canvas ("the v2 BMC of TCS")."""
    data = await _request("GET", f"/bmc/{ticker}/{version}", timeout=_LIGHT_TIMEOUT)
    return _trim_bmc(data)


async def bmc_block_chat(
    ticker: str,
    block_id: str,
    user_message: str,
    version: int | None = None,
) -> dict:
    """Ask a follow-up question scoped to ONE block of a company's BMC. The
    answer is grounded only in that block's citation evidence — perfect for
    "I have questions about Reliance's Key Partnerships block".

    Args:
        ticker: same flexible matcher.
        block_id: one of the canonical IDs — ``customer_segments``,
            ``value_propositions``, ``channels``, ``customer_relationships``,
            ``revenue_streams``, ``key_resources``, ``key_activities``,
            ``key_partnerships``, ``cost_structure``.
        user_message: the user's follow-up question.
        version: optional version number — defaults to latest.

    Returns:
        ``{"answer", "used_markers", "evidence_missing", "evidence",
        "history"}`` or ``{"error": ...}``.
    """
    if block_id not in BMC_BLOCK_IDS:
        return make_error(
            message=f"Invalid block_id {block_id!r}. Must be one of {list(BMC_BLOCK_IDS)}.",
            code="bmc_invalid_block_id",
            next_action="ask_user_to_clarify",
        )
    body: dict = {"user_message": user_message}
    if version is not None:
        body["version"] = version
    return await _request(
        "POST",
        f"/bmc/{ticker}/blocks/{block_id}/chat",
        body=body,
        timeout=_LIGHT_TIMEOUT,
    )


async def bmc_diff(ticker: str, period_a: str, period_b: str) -> dict:
    """Temporal diff: compare a company's Business Model Canvas across two
    fiscal periods (FY to-year strings like ``"2024"`` and ``"2026"``). Auto-
    generates either side if not yet cached, and caches the diff itself.

    WHEN TO USE: "compare TCS's business model from FY24 to FY26", "how has
    Reliance's business evolved between FY22 and FY26".

    Returns:
        ``{"block_diffs": [...9 entries...], "narrative", "a", "b",
        "from_cache"}`` or ``{"error": ...}``.
    """
    if period_a == period_b:
        return make_error(
            message="period_a and period_b must be different fiscal years for a diff.",
            code="bmc_diff_same_period",
            next_action="ask_user_to_clarify",
        )
    return await _request(
        "POST",
        f"/bmc/{ticker}/diff",
        body={"period_a": period_a, "period_b": period_b},
        timeout=_HEAVY_TIMEOUT,
    )


# The integration registry's `python` adapter wraps each plain function here in
# a FunctionTool. Six tools total.
BMC_TOOLS = [
    bmc_get,
    bmc_generate,
    bmc_library,
    bmc_get_version,
    bmc_block_chat,
    bmc_diff,
]
