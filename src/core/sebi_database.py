"""Read-only secondary engine for the SEBI regulatory Postgres.

Backs the Regulatory Lens feature. Main table ``content`` (~40k rows of SEBI
circulars / regulations / orders with an AI-enriched ``ai_tags`` JSON column),
plus ``weekly_summaries`` (digest) and ``insight_feed`` (AI signals). These are
OWNED externally and exposed via a read-only ``frontend`` role — we treat them
as strictly read-only.

Parallels ``src/core/investment_database.py``:
  * Uses its own ``DeclarativeBase`` (``SebiBase``) so PRISM's Alembic history
    never sees these models (no risk of accidental CREATE/DROP). We don't define
    ORM models for the single external table — the router runs raw ``text()``
    SELECTs — but the base keeps the engine isolated.
  * Initialised only when the SEBI DB is configured (``SEBI_DB_*`` parts or
    ``SEBI_DATABASE_URL``). If unset, the engine stays ``None`` and the
    Regulatory Lens endpoints degrade gracefully (503).
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
from sqlalchemy.orm import DeclarativeBase

from src.config import settings


class SebiBase(DeclarativeBase):
    """Separate ORM base for the SEBI DB — keeps these models out of PRISM's
    primary ``Base.metadata`` so Alembic never tries to migrate them."""


# Module-level singletons. Stay None until ``init_sebi_engine()`` runs OR the
# URL isn't configured (in which case ``get_sebi_session`` raises a clear error).
_sebi_engine: AsyncEngine | None = None
_SebiSessionLocal: async_sessionmaker[AsyncSession] | None = None


def init_sebi_engine() -> AsyncEngine | None:
    """Create the SEBI engine + session factory. Returns ``None`` if no SEBI URL
    is configured (allows tests / minimal deploys to skip it)."""
    global _sebi_engine, _SebiSessionLocal
    if _sebi_engine is not None:
        return _sebi_engine
    url = settings.async_sebi_database_url
    if not url:
        return None

    _sebi_engine = create_async_engine(
        url,
        echo=settings.DB_ECHO,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_pre_ping=settings.DB_POOL_PRE_PING,
        connect_args=settings.sebi_connect_args,
    )
    _SebiSessionLocal = async_sessionmaker(
        bind=_sebi_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    return _sebi_engine


async def dispose_sebi_engine() -> None:
    """Close pooled connections on shutdown."""
    global _sebi_engine, _SebiSessionLocal
    if _sebi_engine is not None:
        await _sebi_engine.dispose()
    _sebi_engine = None
    _SebiSessionLocal = None


def is_sebi_configured() -> bool:
    return bool(settings.async_sebi_database_url)


def get_sebi_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _SebiSessionLocal is None:
        init_sebi_engine()
    if _SebiSessionLocal is None:
        raise RuntimeError(
            "SEBI DB is not configured. Set SEBI_DB_* (or SEBI_DATABASE_URL) "
            "in .env to enable the Regulatory Lens feature."
        )
    return _SebiSessionLocal


async def get_sebi_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: read-only session against the SEBI DB.

    We never commit — these queries are SELECT-only. A rollback on exception
    keeps the connection clean for the next request.
    """
    sm = get_sebi_sessionmaker()
    async with sm() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def sebi_session_scope() -> AsyncIterator[AsyncSession]:
    """Same as ``get_sebi_session`` but for non-FastAPI callers (e.g. a future
    agent tool wrapper)."""
    sm = get_sebi_sessionmaker()
    async with sm() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
