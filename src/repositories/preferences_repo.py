"""Data access for per-user preferences (``user_preferences``).

One JSONB row per user. ``get`` returns ``{}`` when none exists yet (callers
treat that as "all defaults"); ``upsert`` replaces the blob (the router does the
merge so PATCH semantics stay explicit).
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.user_preferences import UserPreference


class PreferencesRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, user_id: uuid.UUID) -> dict:
        """This user's preference blob, or ``{}`` if none stored yet."""
        row = await self._session.execute(
            select(UserPreference.prefs).where(UserPreference.user_id == user_id)
        )
        prefs = row.scalar_one_or_none()
        return prefs or {}

    async def upsert(self, user_id: uuid.UUID, prefs: dict) -> dict:
        """Insert or replace this user's preference blob."""
        stmt = (
            insert(UserPreference)
            .values(user_id=user_id, prefs=prefs)
            .on_conflict_do_update(
                index_elements=[UserPreference.user_id],
                set_={"prefs": prefs},
            )
        )
        await self._session.execute(stmt)
        await self._session.commit()
        return prefs
