"""BMC router — thin proxy to the external BMC service (``BMC_URL``).

PRISM's own RAG-based BMC was retired (2026-05-24); the teammate-built ``bmc``
HTTP service (FastAPI on port 8012, read-on-demand, owns its own 5 tables in
the shared Postgres) is now the source of truth. This router preserves the
``/api/v1/bmc/*`` API surface our frontend already speaks (BMCView, 3D
explorer, evidence panel, export buttons, @bmc chat intent) so nothing in the
UI changes — every call passes through to ``BMC_URL`` and the upstream
response is forwarded verbatim.

Endpoints forwarded (per the intake spec):

  POST /{ticker}/run                            — generate + persist
  GET  /{ticker}                                — latest (cheap, no LLM)
  GET  /{ticker}/library                        — all versions
  GET  /{ticker}/{version}                      — specific version
  POST /{ticker}/blocks/{block_id}/chat         — per-block drill-down chat
  GET  /{ticker}/{version}/export?format=...    — json | pdf  (xlsx → 501)
  POST /{ticker}/diff                           — temporal diff

Tenant: the PRISM backend's authenticated firm is injected as ``firm_id`` into
every BMC service request body / query param so per-firm canvas history stays
consistent across services.

Why proxy at all (vs frontend → BMC_URL direct)?
  * Single CORS/auth surface (PRISM)  — when real auth lands, BMC inherits it.
  * Single audit-log surface          — BMC calls show up in PRISM's logs.
  * No new env var leaked to the browser bundle.
"""

from __future__ import annotations

from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel

from src.config import settings
from src.core.auth import get_current_firm_id

router = APIRouter(prefix="/bmc", tags=["Business Model Canvas"])

# Timeouts per the intake spec. /run + /diff can cold-build (downloads + reads +
# LLM extraction); the rest are DB-only and snappy.
_HEAVY_TIMEOUT = 180.0   # POST /run, POST /diff
_LIGHT_TIMEOUT = 30.0    # everything else


def _bmc_url(path: str) -> str:
    return f"{settings.BMC_URL.rstrip('/')}{path}"


async def _forward_json(
    method: str,
    path: str,
    *,
    timeout: float,
    body: dict | None = None,
    params: dict | None = None,
) -> Any:
    """Call the BMC service and forward the JSON response. Upstream status codes
    propagate to the caller verbatim so the frontend can distinguish 404 from
    422 from 502 the same way it would talking directly to the service."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, _bmc_url(path), json=body, params=params)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"BMC service unreachable: {exc}",
        ) from exc
    if resp.status_code >= 400:
        detail: Any
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)
    return resp.json()


async def get_bmc_firm_id(
    _firm: Annotated[str, Depends(get_current_firm_id)],
) -> str:
    """BMC is a FIRM-WIDE shared library — every canvas is visible to everyone
    (guest or signed-in), not scoped to the caller. We still resolve the
    principal (keeps the auth surface / rate-limit identical) but IGNORE its
    per-user firm and return the single shared firm_id, so all reads + writes hit
    one pool. This also makes BMC immune to the per-user firm_id churn the Neon
    DB fallbacks introduce."""
    return settings.BMC_SHARED_FIRM_ID


# ── Request models — match the upstream API; we inject firm_id ────────────────


class BMCRunRequest(BaseModel):
    fiscal_period: str | None = None
    # Integer fast-path (BSE/NSE security_id) — pins the exact entity and skips
    # the BMC service's fuzzy ticker resolver. Forwarded verbatim in the body.
    security_id: int | None = None


class BMCChatRequest(BaseModel):
    user_message: str
    user_id: str | None = None
    version: int | None = None


class BMCDiffRequest(BaseModel):
    period_a: str
    period_b: str
    refresh: bool = False


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("/{ticker}/run", summary="Generate a new BMC version (read-on-demand)")
async def run_bmc(
    ticker: str,
    body: BMCRunRequest,
    firm_id: Annotated[str, Depends(get_bmc_firm_id)],
) -> Any:
    return await _forward_json(
        "POST",
        f"/bmc/{ticker}/run",
        body={**body.model_dump(exclude_none=True), "firm_id": firm_id},
        timeout=_HEAVY_TIMEOUT,
    )


@router.get("/library", summary="All saved canvases for this firm (latest per company)")
async def list_all_canvases(
    firm_id: Annotated[str, Depends(get_bmc_firm_id)],
) -> Any:
    # MUST be declared BEFORE GET /{ticker} so the literal "library" segment is
    # not captured as a ticker (FastAPI matches in declaration order). Upstream
    # returns {firm_id, total, entries:[...]}, forwarded verbatim.
    return await _forward_json(
        "GET", "/bmc/library", params={"firm_id": firm_id}, timeout=_LIGHT_TIMEOUT
    )


@router.get("/{ticker}", summary="Latest persisted BMC for this firm + ticker")
async def get_latest_bmc(
    ticker: str,
    firm_id: Annotated[str, Depends(get_bmc_firm_id)],
) -> Any:
    return await _forward_json(
        "GET", f"/bmc/{ticker}", params={"firm_id": firm_id}, timeout=_LIGHT_TIMEOUT
    )


@router.get("/{ticker}/library", summary="All saved versions (header-level)")
async def list_versions(
    ticker: str,
    firm_id: Annotated[str, Depends(get_bmc_firm_id)],
) -> Any:
    return await _forward_json(
        "GET", f"/bmc/{ticker}/library", params={"firm_id": firm_id}, timeout=_LIGHT_TIMEOUT
    )


@router.post(
    "/{ticker}/blocks/{block_id}/chat",
    summary="Drill-down chat scoped to one BMC block",
)
async def chat_about_block(
    ticker: str,
    block_id: str,
    body: BMCChatRequest,
    firm_id: Annotated[str, Depends(get_bmc_firm_id)],
) -> Any:
    payload = body.model_dump(exclude_none=True)
    # The upstream tracks chat history in `bmc_chats`. Until real users exist,
    # key the thread to the firm so users in the same firm share continuity.
    payload.setdefault("user_id", firm_id)
    return await _forward_json(
        "POST",
        f"/bmc/{ticker}/blocks/{block_id}/chat",
        body={**payload, "firm_id": firm_id},
        timeout=_LIGHT_TIMEOUT,
    )


@router.post("/{ticker}/diff", summary="Temporal diff between two fiscal periods")
async def diff_bmc(
    ticker: str,
    body: BMCDiffRequest,
    firm_id: Annotated[str, Depends(get_bmc_firm_id)],
) -> Any:
    return await _forward_json(
        "POST",
        f"/bmc/{ticker}/diff",
        body={**body.model_dump(), "firm_id": firm_id},
        timeout=_HEAVY_TIMEOUT,
    )


@router.get(
    "/{ticker}/export",
    summary="Export the latest canvas (JSON | PDF; XLSX is 501 upstream)",
)
async def export_latest(
    ticker: str,
    firm_id: Annotated[str, Depends(get_bmc_firm_id)],
    format: str = Query("pdf", pattern="^(json|pdf|xlsx)$"),
) -> Response:
    """Fetch latest version then forward the export bytes."""
    latest = await _forward_json(
        "GET", f"/bmc/{ticker}", params={"firm_id": firm_id}, timeout=_LIGHT_TIMEOUT
    )
    version = latest.get("version")
    if version is None:
        raise HTTPException(status_code=404, detail=f"No BMC yet for {ticker}.")
    return await _export_version(ticker, int(version), firm_id, format)


@router.get(
    "/{ticker}/{version}/export",
    summary="Export a specific canvas version (JSON | PDF; XLSX is 501 upstream)",
)
async def export_version(
    ticker: str,
    version: int,
    firm_id: Annotated[str, Depends(get_bmc_firm_id)],
    format: str = Query("pdf", pattern="^(json|pdf|xlsx)$"),
) -> Response:
    return await _export_version(ticker, version, firm_id, format)


async def _export_version(ticker: str, version: int, firm_id: str, fmt: str) -> Response:
    """Forward export bytes verbatim — the upstream sets the right Content-Type
    (application/pdf, application/json, or returns 501 for xlsx)."""
    url = _bmc_url(f"/bmc/{ticker}/{version}/export")
    try:
        async with httpx.AsyncClient(timeout=_LIGHT_TIMEOUT) as client:
            resp = await client.get(url, params={"format": fmt, "firm_id": firm_id})
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502, detail=f"BMC service unreachable: {exc}"
        ) from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)
    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type", "application/octet-stream"),
        headers={"Content-Disposition": resp.headers.get("content-disposition", "")},
    )


# Specific-version GET — declared AFTER /library and /export so the path
# doesn't capture those literals (FastAPI matches in declaration order).
@router.get("/{ticker}/{version}", summary="Specific persisted version")
async def get_version(
    ticker: str,
    version: int,
    firm_id: Annotated[str, Depends(get_bmc_firm_id)],
) -> Any:
    return await _forward_json(
        "GET",
        f"/bmc/{ticker}/{version}",
        params={"firm_id": firm_id},
        timeout=_LIGHT_TIMEOUT,
    )
