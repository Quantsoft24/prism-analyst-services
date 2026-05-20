"""Debug endpoint for the ModelRouter — exposed only when ``DEBUG=true``.

Useful for confirming a deployment's tier wiring, checking which API keys
were loaded, and getting a snapshot of cool-down state during local
diagnosis. Never exposed in production — production observability lives
in ``agent_runs`` + Sentry / OTel traces.

Mounted conditionally in ``main.py``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.config import settings
from src.services.model_router import get_router

router = APIRouter(prefix="/router", tags=["Debug"])


@router.get(
    "/health",
    summary="ModelRouter health snapshot (DEBUG only)",
    description=(
        "Returns the configured tier chains, number of API keys loaded, and "
        "router strategy. Available only when the server is started with "
        "``DEBUG=true``. Never returns the API keys themselves."
    ),
)
async def router_health() -> dict:
    if not settings.DEBUG:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found.",
        )
    try:
        return get_router().health()
    except RuntimeError as exc:
        # Router not initialized — return a 503 so monitoring can distinguish
        # "router missing" from "router unhealthy".
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
