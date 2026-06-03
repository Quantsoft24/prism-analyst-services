"""Backtest-job persistence (PRISM primary DB).

The worker claims queued jobs with ``SELECT … FOR UPDATE SKIP LOCKED`` so any
number of worker replicas can run safely without double-processing. Progress and
results stream back to the same row, so the API reads live status from one place.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete as sa_delete
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.portfolio import (
    BT_FAILED,
    BT_QUEUED,
    BT_RUNNING,
    BT_SUCCEEDED,
    PortfolioBacktest,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ── API-side (request-scoped session; the dependency commits) ────────────

    async def create(
        self, *, firm_id: str, name: str | None, spec: dict, strategy_hash: str,
        created_by: uuid.UUID | None = None,
    ) -> PortfolioBacktest:
        job = PortfolioBacktest(
            firm_id=firm_id, name=name, spec=spec, strategy_hash=strategy_hash,
            created_by=created_by, status=BT_QUEUED, progress=0.0,
        )
        self.session.add(job)
        await self.session.flush()      # populate id; the request dep commits
        return job

    async def find_cached(self, firm_id: str, strategy_hash: str) -> PortfolioBacktest | None:
        """A previously succeeded backtest with the same spec → instant reuse."""
        stmt = (
            select(PortfolioBacktest)
            .where(
                PortfolioBacktest.firm_id == firm_id,
                PortfolioBacktest.strategy_hash == strategy_hash,
                PortfolioBacktest.status == BT_SUCCEEDED,
            )
            .order_by(PortfolioBacktest.created_at.desc())
            .limit(1)
        )
        return (await self.session.execute(stmt)).scalars().first()

    async def get(self, firm_id: str, job_id: uuid.UUID) -> PortfolioBacktest | None:
        stmt = select(PortfolioBacktest).where(
            PortfolioBacktest.id == job_id, PortfolioBacktest.firm_id == firm_id
        )
        return (await self.session.execute(stmt)).scalars().first()

    async def list_for_firm(self, firm_id: str, limit: int = 50) -> list[PortfolioBacktest]:
        stmt = (
            select(PortfolioBacktest)
            .where(PortfolioBacktest.firm_id == firm_id)
            .order_by(PortfolioBacktest.created_at.desc())
            .limit(limit)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def delete(self, firm_id: str, job_id: uuid.UUID) -> bool:
        """Delete a firm's backtest job. Returns False if it didn't exist."""
        res = await self.session.execute(
            sa_delete(PortfolioBacktest).where(
                PortfolioBacktest.id == job_id, PortfolioBacktest.firm_id == firm_id
            )
        )
        return (res.rowcount or 0) > 0

    # ── Worker-side (own session; commits internally) ────────────────────────

    async def claim_next(self) -> PortfolioBacktest | None:
        """Atomically claim the oldest queued job (skip rows locked by peers)."""
        stmt = (
            select(PortfolioBacktest)
            .where(PortfolioBacktest.status == BT_QUEUED)
            .order_by(PortfolioBacktest.created_at.asc())
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        job = (await self.session.execute(stmt)).scalars().first()
        if job is None:
            await self.session.rollback()
            return None
        job.status = BT_RUNNING
        job.started_at = _utcnow()
        job.stage = "Starting"
        job.progress = 0.0
        await self.session.commit()
        return job

    async def set_progress(self, job_id: uuid.UUID, progress: float, stage: str) -> None:
        await self.session.execute(
            update(PortfolioBacktest)
            .where(PortfolioBacktest.id == job_id)
            .values(progress=progress, stage=stage, updated_at=_utcnow())
        )
        await self.session.commit()

    async def finish_success(self, job_id: uuid.UUID, result: dict) -> None:
        await self.session.execute(
            update(PortfolioBacktest)
            .where(PortfolioBacktest.id == job_id)
            .values(
                status=BT_SUCCEEDED, progress=1.0, stage="Done", result=result,
                error=None, finished_at=_utcnow(), updated_at=_utcnow(),
            )
        )
        await self.session.commit()

    async def finish_error(self, job_id: uuid.UUID, error: str) -> None:
        await self.session.execute(
            update(PortfolioBacktest)
            .where(PortfolioBacktest.id == job_id)
            .values(
                status=BT_FAILED, error=error[:4000], finished_at=_utcnow(),
                updated_at=_utcnow(),
            )
        )
        await self.session.commit()

    async def reclaim_stale(self, timeout_min: int = 30) -> int:
        """Requeue RUNNING jobs that haven't progressed in ``timeout_min`` (a
        worker died mid-run). Progress updates bump ``updated_at``, so a live job
        is never reclaimed."""
        cutoff = _utcnow() - timedelta(minutes=timeout_min)
        res = await self.session.execute(
            update(PortfolioBacktest)
            .where(PortfolioBacktest.status == BT_RUNNING, PortfolioBacktest.updated_at < cutoff)
            .values(status=BT_QUEUED, stage="Requeued after stall", updated_at=_utcnow())
        )
        await self.session.commit()
        return res.rowcount or 0
