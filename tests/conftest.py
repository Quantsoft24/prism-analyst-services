"""Shared pytest fixtures — async DB engine, session, and FastAPI client.

Strategy:
  * One engine per test session, pointed at a TEST_DATABASE_URL.
  * Each test runs inside a SAVEPOINT that gets rolled back at teardown,
    so tests are fast and independent without re-running migrations.
  * The FastAPI ``get_session`` dependency is overridden to yield the
    same session the test uses, so any data the test seeds is visible
    to the endpoint under test.

Required for these tests to run:
  * A Postgres reachable via ``TEST_DATABASE_URL`` env var
    (defaults to ``postgresql+asyncpg://postgres:postgres@localhost:5432/prism_test``).
  * Migrations already applied to that database
    (CI runs ``alembic upgrade head`` before pytest; locally you do the same).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.database import get_session
from src.main import app

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/prism_test",
)


@pytest_asyncio.fixture(scope="session")
async def engine():
    """Single async engine for the whole test session."""
    eng = create_async_engine(TEST_DATABASE_URL, echo=False, future=True)
    try:
        yield eng
    finally:
        await eng.dispose()


@pytest_asyncio.fixture
async def db_session(engine) -> AsyncIterator[AsyncSession]:
    """Per-test session wrapped in a transaction that's rolled back at teardown.

    Uses the SQLAlchemy "join-an-external-transaction" pattern so the test
    sees committed data (e.g., from migrations + seed) but anything the
    test itself writes is discarded.
    """
    async with engine.connect() as connection:
        trans = await connection.begin()
        SessionLocal = async_sessionmaker(
            bind=connection,
            class_=AsyncSession,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
        async with SessionLocal() as session:
            try:
                yield session
            finally:
                await session.close()
        await trans.rollback()


@pytest_asyncio.fixture
async def client(db_session) -> AsyncIterator[AsyncClient]:
    """FastAPI ASGI client whose DB sessions are the same one the test uses."""

    async def _override_get_session():
        yield db_session

    app.dependency_overrides[get_session] = _override_get_session
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides.pop(get_session, None)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Headers that satisfy the dev-mode auth stub. Adjust when real auth lands."""
    return {"X-Dev-Firm": "QUANTSOFT"}
