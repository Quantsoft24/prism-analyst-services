"""Integrations endpoints — registered agent tools + per-firm enable/disable.

  GET /api/v1/integrations            → list every integration with this firm's
                                         effective enabled state + load health.
  PUT /api/v1/integrations/{name}     → toggle one integration ON/OFF for the firm.

The *available* integrations come from ``config/integrations.yml`` (the catalog,
built at startup). Firm toggles are persisted in ``firm_integrations`` (default
ON). Per-user toggles + add-from-UI are deferred to the auth/user-profile slice.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.auth import get_current_firm_id
from src.core.database import get_session
from src.integrations import get_registry
from src.integrations.firm_state import integration_view
from src.repositories.integration_repo import IntegrationRepository

router = APIRouter(prefix="/integrations", tags=["Integrations"])


class IntegrationToggle(BaseModel):
    enabled: bool


@router.get(
    "",
    summary="List registered agent integrations + this firm's enabled state",
    description=(
        "Every integration in the registry with its source type, load status "
        "('ok' | 'error' | 'disabled'), tool count, build error (if any), and "
        "this firm's effective ``enabled`` flag (default ON)."
    ),
)
async def list_integrations(
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> dict:
    registry = get_registry()
    if registry is None:
        return {"integrations": [], "total": 0, "ready": False, "tool_count": 0}
    view = await integration_view(firm_id, session)
    return {
        "integrations": view,
        "total": len(view),
        "ready": True,
        "tool_count": sum(h["tool_count"] for h in view),
    }


@router.put(
    "/{name}",
    summary="Enable or disable an integration for the firm",
    description="Persists a firm-level override. Takes effect on the next agent run.",
    responses={404: {"description": "Unknown integration name."}},
)
async def toggle_integration(
    name: str,
    body: IntegrationToggle,
    session: Annotated[AsyncSession, Depends(get_session)],
    firm_id: Annotated[str, Depends(get_current_firm_id)],
) -> dict:
    registry = get_registry()
    known = set(registry.names()) if registry is not None else set()
    if name not in known:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown integration {name!r}. Known: {sorted(known)}",
        )
    await IntegrationRepository(session).set_override(firm_id, name, body.enabled)
    return {"name": name, "enabled": body.enabled}
