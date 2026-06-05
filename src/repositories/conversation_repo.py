"""Read access for chat conversation history (derived from ``agent_runs``).

MVP with no new table: a "conversation" is the set of ``agent_runs`` sharing a
``session_id``. Scoped to the user (``user_id``) when authenticated, else to the
firm (dev / pre-auth). Title = first user message; preview = latest answer.
A dedicated ``conversations`` table (rename/pin/share) is a later phase.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.agent_run import AgentRun
from src.models.chat_conversation import ChatConversation

_TITLE_MAX = 120
_PREVIEW_MAX = 200


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def _scope(self, firm_id: str, user_id: uuid.UUID | None):
        # user_id is globally unique → scope by it when present; else by firm.
        return AgentRun.user_id == user_id if user_id is not None else AgentRun.firm_id == firm_id

    async def list_conversations(
        self, *, firm_id: str, user_id: uuid.UUID | None, limit: int = 30
    ) -> list[dict[str, Any]]:
        scope = self._scope(firm_id, user_id)
        visible = AgentRun.hidden_at.is_(None)  # exclude soft-deleted runs

        # 1) Most-recently-active sessions + turn counts.
        agg = (
            select(
                AgentRun.session_id,
                func.max(AgentRun.created_at).label("last_activity"),
                func.count().label("turns"),
            )
            .where(scope, visible)
            .group_by(AgentRun.session_id)
            .order_by(func.max(AgentRun.created_at).desc())
            .limit(limit)
        )
        rows = (await self._session.execute(agg)).all()
        if not rows:
            return []
        session_ids = [r.session_id for r in rows]
        meta = {r.session_id: (r.last_activity, r.turns) for r in rows}

        # 2) Per-session title (first user message) + preview (latest answer).
        detail = (
            select(
                AgentRun.session_id,
                AgentRun.user_input,
                AgentRun.final_answer,
                AgentRun.agent_name,
            )
            .where(AgentRun.session_id.in_(session_ids))
            .where(scope, visible)
            .order_by(AgentRun.created_at.asc())
        )
        title: dict[str, str] = {}
        agent: dict[str, str | None] = {}
        preview: dict[str, str] = {}
        for d in (await self._session.execute(detail)).all():
            if d.session_id not in title:
                title[d.session_id] = d.user_input
                agent[d.session_id] = d.agent_name
            if d.final_answer:
                preview[d.session_id] = d.final_answer

        # User-renamed titles override the derived (first-message) title.
        overlay = await self._session.execute(
            select(ChatConversation.session_id, ChatConversation.title).where(
                ChatConversation.session_id.in_(session_ids)
            )
        )
        renamed = {sid: t for sid, t in overlay.all()}

        out: list[dict[str, Any]] = []
        for sid in session_ids:
            last_activity, turns = meta[sid]
            out.append(
                {
                    "session_id": sid,
                    "title": (renamed.get(sid) or title.get(sid) or "Untitled")[:_TITLE_MAX],
                    "turns": turns,
                    "last_activity": last_activity,
                    "preview": (preview.get(sid) or "")[:_PREVIEW_MAX],
                    "agent_name": agent.get(sid),
                }
            )
        return out

    async def get_conversation(
        self, *, session_id: str, firm_id: str, user_id: uuid.UUID | None
    ) -> list[AgentRun]:
        scope = self._scope(firm_id, user_id)
        q = (
            select(AgentRun)
            .where(AgentRun.session_id == session_id)
            .where(scope, AgentRun.hidden_at.is_(None))
            .order_by(AgentRun.created_at.asc())
        )
        return list((await self._session.execute(q)).scalars().all())

    async def hide_conversation(
        self, *, session_id: str, firm_id: str, user_id: uuid.UUID | None
    ) -> int:
        """Soft-delete: mark every (still-visible) run in the session as hidden.
        The audit rows are preserved. Returns how many runs were hidden."""
        scope = self._scope(firm_id, user_id)
        stmt = (
            update(AgentRun)
            .where(AgentRun.session_id == session_id, scope, AgentRun.hidden_at.is_(None))
            .values(hidden_at=datetime.now(timezone.utc))
        )
        res = await self._session.execute(stmt)
        await self._session.commit()
        return res.rowcount or 0

    async def set_title(
        self, *, session_id: str, firm_id: str, user_id: uuid.UUID | None, title: str
    ) -> bool:
        """Rename a conversation (overlay). Returns False if the session isn't
        the caller's (no visible run in scope) — so users can't retitle others'."""
        owns = await self._session.scalar(
            select(AgentRun.id)
            .where(AgentRun.session_id == session_id, self._scope(firm_id, user_id))
            .limit(1)
        )
        if owns is None:
            return False
        stmt = (
            pg_insert(ChatConversation)
            .values(session_id=session_id, firm_id=firm_id, user_id=user_id, title=title)
            .on_conflict_do_update(
                index_elements=[ChatConversation.session_id], set_={"title": title}
            )
        )
        await self._session.execute(stmt)
        await self._session.commit()
        return True
