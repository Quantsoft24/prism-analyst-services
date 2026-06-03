"""Persistence for saved custom factors + strategies (PRISM primary DB).

Firm-scoped (``firm_id`` slug) with a nullable ``created_by`` for per-user
filtering once auth populates it. Mutators flush only — the request-scoped
session dependency commits.
"""

from __future__ import annotations

import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.portfolio import PortfolioCustomFactor, PortfolioStrategy


class CustomFactorStore:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self, *, firm_id: str, name: str, expression: str, direction: str,
        normalization: str, created_by: uuid.UUID | None = None,
    ) -> PortfolioCustomFactor:
        cf = PortfolioCustomFactor(
            firm_id=firm_id, name=name, expression=expression,
            direction=direction, normalization=normalization, created_by=created_by,
        )
        self.session.add(cf)
        await self.session.flush()
        return cf

    async def list(self, firm_id: str) -> list[PortfolioCustomFactor]:
        stmt = (
            select(PortfolioCustomFactor)
            .where(PortfolioCustomFactor.firm_id == firm_id)
            .order_by(PortfolioCustomFactor.created_at.desc())
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def get(self, firm_id: str, cf_id: uuid.UUID) -> PortfolioCustomFactor | None:
        stmt = select(PortfolioCustomFactor).where(
            PortfolioCustomFactor.id == cf_id, PortfolioCustomFactor.firm_id == firm_id
        )
        return (await self.session.execute(stmt)).scalars().first()

    async def delete(self, firm_id: str, cf_id: uuid.UUID) -> bool:
        res = await self.session.execute(
            delete(PortfolioCustomFactor).where(
                PortfolioCustomFactor.id == cf_id, PortfolioCustomFactor.firm_id == firm_id
            )
        )
        return (res.rowcount or 0) > 0


class StrategyStore:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self, *, firm_id: str, name: str, config: dict,
        created_by: uuid.UUID | None = None,
    ) -> PortfolioStrategy:
        s = PortfolioStrategy(firm_id=firm_id, name=name, config=config, created_by=created_by)
        self.session.add(s)
        await self.session.flush()
        return s

    async def list(self, firm_id: str) -> list[PortfolioStrategy]:
        stmt = (
            select(PortfolioStrategy)
            .where(PortfolioStrategy.firm_id == firm_id)
            .order_by(PortfolioStrategy.updated_at.desc())
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def get(self, firm_id: str, sid: uuid.UUID) -> PortfolioStrategy | None:
        stmt = select(PortfolioStrategy).where(
            PortfolioStrategy.id == sid, PortfolioStrategy.firm_id == firm_id
        )
        return (await self.session.execute(stmt)).scalars().first()

    async def delete(self, firm_id: str, sid: uuid.UUID) -> bool:
        res = await self.session.execute(
            delete(PortfolioStrategy).where(
                PortfolioStrategy.id == sid, PortfolioStrategy.firm_id == firm_id
            )
        )
        return (res.rowcount or 0) > 0
