"""Backtest worker process.

Long-running loop that claims queued ``pb_backtests`` jobs (``FOR UPDATE SKIP
LOCKED``), runs the engine against the read-only investment RDS, streams progress
back, and persists the result. Restart-safe (stalled RUNNING jobs are requeued)
and horizontally scalable — run N replicas. Deployed as its own container:

    python -m src.portfolio.worker

It shares the codebase + env with the API but runs separately so heavy backtests
never block request handling.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy.ext.asyncio import async_sessionmaker

from src.core.database import (
    FAILOVER_ERRORS,
    dispose_engine,
    ensure_engine,
    get_sessionmaker,
)
from src.core.investment_database import (
    dispose_investment_engine,
    get_investment_sessionmaker,
    init_investment_engine,
)
from src.portfolio.backtest import run_backtest
from src.portfolio.job_repo import JobRepository
from src.portfolio.repository import PortfolioRepository
from src.portfolio.serialize import result_to_dict, spec_from_dict

logger = logging.getLogger("portfolio.worker")

POLL_INTERVAL_S = 2.0
STALE_TIMEOUT_MIN = 30


async def process_one(primary_sm: async_sessionmaker, invest_sm: async_sessionmaker) -> bool:
    """Claim + run one job. Returns True if a job was processed."""
    async with primary_sm() as jobs_session:
        repo = JobRepository(jobs_session)
        job = await repo.claim_next()
        if job is None:
            return False
        job_id, spec_dict = job.id, job.spec
        logger.info("claimed backtest %s", job_id)

        async def progress(p: float, stage: str) -> None:
            try:
                await repo.set_progress(job_id, p, stage)
            except Exception:  # noqa: BLE001 — progress is best-effort
                logger.warning("progress update failed for %s", job_id, exc_info=True)

        try:
            spec = spec_from_dict(spec_dict)
            async with invest_sm() as invest_session:
                prepo = PortfolioRepository(invest_session)
                result = await run_backtest(prepo, spec, progress=progress)
            await repo.finish_success(job_id, result_to_dict(result))
            logger.info("backtest %s succeeded", job_id)
        except Exception as exc:  # noqa: BLE001 — record any failure on the job
            logger.exception("backtest %s failed", job_id)
            await repo.finish_error(job_id, f"{type(exc).__name__}: {exc}")
        return True


async def worker_loop(poll_interval: float = POLL_INTERVAL_S) -> None:
    # Failover-aware boot (same as the API lifespan): probe the primary DB and
    # rotate to a configured fallback if it's unreachable (e.g. a capped Neon
    # project). Without this the worker binds to a dead primary and crash-loops.
    await ensure_engine()
    init_investment_engine()
    invest_sm = get_investment_sessionmaker()
    logger.info("portfolio backtest worker started")
    try:
        while True:
            try:
                # Re-fetch the sessionmaker each tick so a runtime DB failover
                # (which rebuilds the engine) is picked up rather than reusing a
                # stale binding to the dead DB.
                primary_sm = get_sessionmaker()
                processed = await process_one(primary_sm, invest_sm)
                if not processed:
                    # Idle tick: requeue any stalled jobs, then wait.
                    async with primary_sm() as s:
                        n = await JobRepository(s).reclaim_stale(STALE_TIMEOUT_MIN)
                        if n:
                            logger.warning("requeued %d stalled backtest(s)", n)
                    await asyncio.sleep(poll_interval)
            except FAILOVER_ERRORS as exc:
                # Primary DB dropped connections mid-loop — rotate to a healthy
                # fallback and keep going instead of crashing the process.
                logger.warning(
                    "worker DB connection error (%s) — failing over and retrying",
                    type(exc).__name__,
                )
                await ensure_engine()
                await asyncio.sleep(poll_interval)
    finally:
        await dispose_engine()
        await dispose_investment_engine()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(worker_loop())


if __name__ == "__main__":
    main()
