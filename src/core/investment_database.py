"""Read-only secondary engine for the investment Postgres (AWS RDS).

Backs the Stock Dashboard. Two tables only — ``master_securities`` (NSE/BSE
security master) and ``prices_and_securities`` (daily OHLC / volume / value /
market-cap). These are OWNED externally; we treat them as strictly read-only.

Like the other read-only secondaries (SEBI), it:
  * Uses its own ``DeclarativeBase`` (``InvestmentBase``) so PRISM's Alembic
    history never sees these models (no risk of accidental CREATE/DROP).
  * Initialised only when the investment DB is configured (``INVESTMENT_DB_*``
    parts or ``INVESTMENT_DATABASE_URL``). If unset, the engine stays ``None``
    and the Stock Dashboard endpoints degrade gracefully (503).
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


class InvestmentBase(DeclarativeBase):
    """Separate ORM base for the investment DB — keeps these models out of
    PRISM's primary ``Base.metadata`` so Alembic never tries to migrate them."""


# Module-level singletons. Stay None until ``init_investment_engine()`` runs OR
# the URL isn't configured (in which case ``get_investment_session`` raises a
# clear error).
_investment_engine: AsyncEngine | None = None
_InvestmentSessionLocal: async_sessionmaker[AsyncSession] | None = None


def init_investment_engine() -> AsyncEngine | None:
    """Create the investment engine + session factory. Returns ``None`` if no
    investment URL is configured (allows tests / minimal deploys to skip it)."""
    global _investment_engine, _InvestmentSessionLocal
    if _investment_engine is not None:
        return _investment_engine
    url = settings.async_investment_database_url
    if not url:
        return None

    _investment_engine = create_async_engine(
        url,
        echo=settings.DB_ECHO,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_pre_ping=settings.DB_POOL_PRE_PING,
        connect_args=settings.investment_connect_args,
    )
    _InvestmentSessionLocal = async_sessionmaker(
        bind=_investment_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    return _investment_engine


async def dispose_investment_engine() -> None:
    """Close pooled connections on shutdown."""
    global _investment_engine, _InvestmentSessionLocal
    if _investment_engine is not None:
        await _investment_engine.dispose()
    _investment_engine = None
    _InvestmentSessionLocal = None


def is_investment_configured() -> bool:
    return bool(settings.async_investment_database_url)


def get_investment_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _InvestmentSessionLocal is None:
        init_investment_engine()
    if _InvestmentSessionLocal is None:
        raise RuntimeError(
            "Investment DB is not configured. Set INVESTMENT_DB_* (or "
            "INVESTMENT_DATABASE_URL) in .env to enable the Stock Dashboard."
        )
    return _InvestmentSessionLocal


async def get_investment_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: read-only session against the investment DB.

    We never commit — these queries are SELECT-only. A rollback on exception
    keeps the connection clean for the next request.
    """
    sm = get_investment_sessionmaker()
    async with sm() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def investment_session_scope() -> AsyncIterator[AsyncSession]:
    """Same as ``get_investment_session`` but for non-FastAPI callers."""
    sm = get_investment_sessionmaker()
    async with sm() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
