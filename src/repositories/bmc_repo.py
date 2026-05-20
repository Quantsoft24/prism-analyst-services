"""Data access for Business Model Canvas analyses."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models.bmc import BMCAnalysis, BMCBlock


class BMCRepository:
    """CRUD + versioning for BMC analyses."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def next_version(self, firm_id: str, ticker: str) -> int:
        """Return the next version number for (firm, ticker) — 1 if none yet.

        Re-generating a canvas never overwrites: it creates a new version so
        a firm can diff how a company's business model changed over time.
        """
        stmt = select(func.max(BMCAnalysis.version)).where(
            BMCAnalysis.firm_id == firm_id, BMCAnalysis.ticker == ticker.upper()
        )
        current = (await self._session.execute(stmt)).scalar_one_or_none()
        return (current or 0) + 1

    async def add(self, analysis: BMCAnalysis) -> BMCAnalysis:
        self._session.add(analysis)
        await self._session.flush()
        return analysis

    async def get_latest(self, firm_id: str, ticker: str) -> BMCAnalysis | None:
        """Most recent version for (firm, ticker), with blocks + evidence loaded."""
        stmt = (
            select(BMCAnalysis)
            .where(BMCAnalysis.firm_id == firm_id, BMCAnalysis.ticker == ticker.upper())
            .order_by(BMCAnalysis.version.desc())
            .limit(1)
            .options(selectinload(BMCAnalysis.blocks).selectinload(BMCBlock.evidence))
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_version(self, firm_id: str, ticker: str, version: int) -> BMCAnalysis | None:
        stmt = (
            select(BMCAnalysis)
            .where(
                BMCAnalysis.firm_id == firm_id,
                BMCAnalysis.ticker == ticker.upper(),
                BMCAnalysis.version == version,
            )
            .options(selectinload(BMCAnalysis.blocks).selectinload(BMCBlock.evidence))
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list_versions(self, firm_id: str, ticker: str) -> list[BMCAnalysis]:
        """All versions for (firm, ticker), newest first (header only, no blocks)."""
        stmt = (
            select(BMCAnalysis)
            .where(BMCAnalysis.firm_id == firm_id, BMCAnalysis.ticker == ticker.upper())
            .order_by(BMCAnalysis.version.desc())
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def get_by_id(self, bmc_id: uuid.UUID) -> BMCAnalysis | None:
        stmt = (
            select(BMCAnalysis)
            .where(BMCAnalysis.id == bmc_id)
            .options(selectinload(BMCAnalysis.blocks).selectinload(BMCBlock.evidence))
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()
