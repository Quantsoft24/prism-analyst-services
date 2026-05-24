"""Data access for firm-level integration overrides (``firm_integrations``).

Default is ON (Part-A), so a row exists only when a firm has changed an
integration's state. The resolver layers: firm-override → default(True).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.integration import FirmIntegration


class IntegrationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_overrides(self, firm_id: str) -> dict[str, bool]:
        """Map of integration name → enabled, for this firm's explicit overrides."""
        rows = await self._session.execute(
            select(FirmIntegration.name, FirmIntegration.enabled).where(
                FirmIntegration.firm_id == firm_id
            )
        )
        return {name: enabled for name, enabled in rows.all()}

    async def set_override(self, firm_id: str, name: str, enabled: bool) -> None:
        """Upsert a firm's enable/disable for one integration."""
        stmt = (
            insert(FirmIntegration)
            .values(firm_id=firm_id, name=name, enabled=enabled)
            .on_conflict_do_update(
                constraint="uq_firm_integration",
                set_={"enabled": enabled},
            )
        )
        await self._session.execute(stmt)
        await self._session.commit()
