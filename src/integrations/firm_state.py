"""Resolve a firm's *effective* integration state by layering DB overrides over
the registry catalog (default ON, per Part-A).

This is the seam the per-user layer will extend later: enabled = user-override
?? firm-override ?? default(True). For now it's firm + default.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.integrations import get_registry
from src.repositories.integration_repo import IntegrationRepository


async def enabled_integration_names(firm_id: str, session: AsyncSession) -> list[str]:
    """Integration names this firm should get — every loaded integration minus
    the ones the firm explicitly turned off."""
    registry = get_registry()
    if registry is None:
        return []
    overrides = await IntegrationRepository(session).get_overrides(firm_id)
    return [n for n in registry.names() if overrides.get(n, True)]


async def integration_view(firm_id: str, session: AsyncSession) -> list[dict]:
    """Per-integration health + this firm's effective enabled flag — for the
    Settings UI / GET /integrations."""
    registry = get_registry()
    if registry is None:
        return []
    overrides = await IntegrationRepository(session).get_overrides(firm_id)
    view: list[dict] = []
    for h in registry.health():
        loaded_ok = h["status"] == "ok"
        # Firm can only toggle integrations that actually loaded; default ON.
        effective = loaded_ok and overrides.get(h["name"], True)
        view.append({**h, "enabled": effective})
    return view
