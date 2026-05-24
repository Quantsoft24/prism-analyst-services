"""Async PostgreSQL database engine, session factory, and FastAPI dependency.

The engine is created lazily on app startup and disposed on shutdown — see
``src/main.py``. Tests build their own engine bound to a transactional fixture
in ``tests/conftest.py`` and override the ``get_session`` dependency.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import settings

# Module-level singletons. ``None`` until ``init_engine()`` runs.
_engine: AsyncEngine | None = None
_SessionLocal: async_sessionmaker[AsyncSession] | None = None


def init_engine() -> AsyncEngine:
    """Create the global async engine and session factory.

    Idempotent: safe to call multiple times in tests; subsequent calls return
    the existing engine.
    """
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine

    _engine = create_async_engine(
        settings.async_database_url,
        echo=settings.DB_ECHO,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_pre_ping=settings.DB_POOL_PRE_PING,
        connect_args=settings.db_connect_args,
    )
    _SessionLocal = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    return _engine


async def dispose_engine() -> None:
    """Close all pooled connections. Called on app shutdown."""
    global _engine, _SessionLocal
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _SessionLocal = None


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Return the configured session factory, initializing if necessary."""
    if _SessionLocal is None:
        init_engine()
    assert _SessionLocal is not None
    return _SessionLocal


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields a request-scoped async DB session.

    The session is committed on success and rolled back on any exception.
    Always closed at the end of the request.
    """
    sm = get_sessionmaker()
    async with sm() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Context manager for non-FastAPI code paths (scripts, background tasks)."""
    sm = get_sessionmaker()
    async with sm() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
