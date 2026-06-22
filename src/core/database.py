"""Async PostgreSQL database engine, session factory, and FastAPI dependency.

The engine is created lazily on app startup and disposed on shutdown — see
``src/main.py``. Tests build their own engine bound to a transactional fixture
in ``tests/conftest.py`` and override the ``get_session`` dependency.

Multi-URL failover
------------------
``settings.async_database_urls`` is an ordered list: the primary ``DATABASE_URL``
followed by any ``DATABASE_URL_FALLBACKS``. When the active database stops
accepting connections — e.g. a Neon free-tier project whose monthly compute
allowance is spent (the compute endpoint gets *disabled*, not just suspended) —
the app rotates to the next URL automatically:

  * **Startup** (``ensure_engine``) probes the active URL with ``SELECT 1`` and
    rotates through the list until one connects, so the app boots on a healthy
    DB even if the primary is capped.
  * **Runtime** — if the active DB drops connections mid-request, that one
    request fails with a connection error; ``get_session`` / ``session_scope``
    catch it, rotate to a healthy URL, and the *next* request lands cleanly.

The fallbacks are independent databases (data is NOT replicated), so a failover
lands on a separate dataset — acceptable only where losing data is fine
(internal/dev tooling). On process restart the index resets to 0, so the app
prefers the primary again (handy when its allowance resets).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import settings

logger = logging.getLogger(__name__)

# Module-level singletons. ``None`` until ``init_engine()`` runs.
_engine: AsyncEngine | None = None
_SessionLocal: async_sessionmaker[AsyncSession] | None = None
# Index into ``settings.async_database_urls`` for the currently-active DB.
_active_idx: int = 0

# Error classes that mean "this database isn't accepting our connection" and we
# should try the next URL. Broad on purpose — a capped Neon endpoint can surface
# as any of these (connect refused, endpoint disabled, too many connections).
FAILOVER_ERRORS = (OperationalError, InterfaceError, DBAPIError, OSError, ConnectionError)


def _build_engine(url: str) -> AsyncEngine:
    return create_async_engine(
        url,
        echo=settings.DB_ECHO,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_pre_ping=settings.DB_POOL_PRE_PING,
        connect_args=settings.db_connect_args,
    )


def init_engine() -> AsyncEngine:
    """Create the global async engine + session factory for the active DB URL.

    Idempotent: safe to call multiple times in tests; subsequent calls return
    the existing engine. Does NOT probe connectivity — call ``ensure_engine()``
    at startup for failover-aware boot.
    """
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine

    urls = settings.async_database_urls
    _engine = _build_engine(urls[_active_idx])
    _SessionLocal = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    return _engine


async def dispose_engine() -> None:
    """Close all pooled connections. Called on app shutdown."""
    global _engine, _SessionLocal, _active_idx
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _SessionLocal = None
    _active_idx = 0


async def _switch_to(idx: int) -> None:
    """Dispose the current engine and rebuild it against URL ``idx``."""
    global _engine, _SessionLocal, _active_idx
    if _engine is not None:
        try:
            await _engine.dispose()
        except Exception:  # noqa: BLE001 — disposing a dead engine can itself raise
            pass
    _active_idx = idx
    _engine = _build_engine(settings.async_database_urls[idx])
    _SessionLocal = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def _can_connect(engine: AsyncEngine) -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001 — any failure here means "try the next URL"
        logger.warning("DB connectivity probe failed: %s", exc)
        return False


async def ensure_engine() -> None:
    """Ensure the active engine can connect; otherwise rotate through fallbacks.

    Called at startup and after a runtime connection failure. If no URL is
    reachable, leaves the engine pointing at the last-tried URL and logs — the
    next DB call then raises a clear error.
    """
    if _engine is None:
        init_engine()
    assert _engine is not None
    if await _can_connect(_engine):
        return

    urls = settings.async_database_urls
    n = len(urls)
    if n <= 1:
        logger.error("Primary database unreachable and no fallbacks configured.")
        return
    for step in range(1, n):
        idx = (_active_idx + step) % n
        logger.warning("DB failover: trying URL index %d of %d…", idx, n)
        await _switch_to(idx)
        if await _can_connect(_engine):
            logger.warning("DB failover succeeded → now on URL index %d of %d.", idx, n)
            return
    logger.error("All %d database URLs are unreachable.", n)


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Return the configured session factory, initializing if necessary."""
    if _SessionLocal is None:
        init_engine()
    assert _SessionLocal is not None
    return _SessionLocal


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields a request-scoped async DB session.

    Committed on success, rolled back on any exception, always closed. If the
    active database drops the connection (e.g. a Neon project just hit its
    compute cap), the failing request raises, and we rotate to a healthy DB so
    the *next* request lands cleanly (failover is connection-level, not a
    mid-transaction replay).
    """
    sm = get_sessionmaker()
    try:
        async with sm() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    except FAILOVER_ERRORS as exc:
        logger.warning("DB connection error (%s) — attempting failover.", type(exc).__name__)
        await ensure_engine()
        raise


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Context manager for non-FastAPI code paths (scripts, background tasks)."""
    sm = get_sessionmaker()
    try:
        async with sm() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    except FAILOVER_ERRORS as exc:
        logger.warning("DB connection error (%s) — attempting failover.", type(exc).__name__)
        await ensure_engine()
        raise
