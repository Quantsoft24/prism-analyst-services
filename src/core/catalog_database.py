"""Read-only secondary engine for the catalog Postgres (stock_chat DB).

PRISM's company lookup (and future filings-catalog reads) hit this engine
instead of duplicating tables in PRISM's own DB. Tables here are OWNED by
the stock-chat / bmc services — we treat them as read-only.

Parallels ``src/core/database.py`` but:
  * Uses a different ``DeclarativeBase`` (``CatalogBase``) so PRISM's Alembic
    history never sees these models (no risk of accidental CREATE/DROP).
  * Initialised only when ``CATALOG_DATABASE_URL`` (or the back-compat
    ``POSTGRES_URL``) is set. If neither is set, the catalog engine stays
    ``None`` and consumers should degrade gracefully.
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


class CatalogBase(DeclarativeBase):
    """Separate ORM base for the catalog DB — keeps these models out of
    PRISM's primary ``Base.metadata`` so Alembic never tries to migrate them."""


# Module-level singletons. Stay None until ``init_catalog_engine()`` runs OR
# the URL isn't set (in which case they stay None and ``get_catalog_session``
# raises a clear error).
_catalog_engine: AsyncEngine | None = None
_CatalogSessionLocal: async_sessionmaker[AsyncSession] | None = None


def init_catalog_engine() -> AsyncEngine | None:
    """Create the catalog engine + session factory. Returns ``None`` if no
    catalog URL is configured (allows tests / minimal deploys to skip it)."""
    global _catalog_engine, _CatalogSessionLocal
    if _catalog_engine is not None:
        return _catalog_engine
    url = settings.async_catalog_database_url
    if not url:
        return None

    _catalog_engine = create_async_engine(
        url,
        echo=settings.DB_ECHO,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_pre_ping=settings.DB_POOL_PRE_PING,
        connect_args=settings.db_connect_args,
    )
    _CatalogSessionLocal = async_sessionmaker(
        bind=_catalog_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    return _catalog_engine


async def dispose_catalog_engine() -> None:
    """Close pooled connections on shutdown."""
    global _catalog_engine, _CatalogSessionLocal
    if _catalog_engine is not None:
        await _catalog_engine.dispose()
    _catalog_engine = None
    _CatalogSessionLocal = None


def is_catalog_configured() -> bool:
    return bool(settings.async_catalog_database_url)


def get_catalog_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _CatalogSessionLocal is None:
        init_catalog_engine()
    if _CatalogSessionLocal is None:
        raise RuntimeError(
            "Catalog DB is not configured. Set CATALOG_DATABASE_URL (or "
            "POSTGRES_URL) in .env to enable company / catalog lookups."
        )
    return _CatalogSessionLocal


async def get_catalog_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: read-only session against the catalog DB.

    We don't commit — these queries are SELECT-only. A rollback on exception
    keeps the connection clean for the next request.
    """
    sm = get_catalog_sessionmaker()
    async with sm() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def catalog_session_scope() -> AsyncIterator[AsyncSession]:
    """Same as ``get_catalog_session`` but for non-FastAPI callers (agent tools)."""
    sm = get_catalog_sessionmaker()
    async with sm() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
