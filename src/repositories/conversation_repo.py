"""Read access for chat conversation history (derived from ``agent_runs``).

MVP with no new table: a "conversation" is the set of ``agent_runs`` sharing a
``session_id``. Scoping:
  * signed-in  → by ``user_id`` (globally unique).
  * guest      → all guests share the ``__anonymous__`` firm, so they MUST be
                 isolated per browser via ``client_key`` (the X-Guest-Id sent by
                 the client; never the shared firm). No client_key → match
                 nothing (can't safely identify the guest).
  * dev/no-auth→ by ``firm_id`` (the dev firm).
Title = first user message; preview = latest answer.
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, false, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.principal import ANONYMOUS_FIRM
from src.models.agent_run import AgentRun
from src.models.chat_conversation import ChatConversation
from src.models.message_feedback import MessageFeedback

_TITLE_MAX = 120
_PREVIEW_MAX = 200


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def _scope(self, *, firm_id: str, user_id: uuid.UUID | None, client_key: str | None):
        if user_id is not None:
            return AgentRun.user_id == user_id
        if firm_id == ANONYMOUS_FIRM:
            # Guests share the anonymous firm → isolate per browser. Without a
            # client_key we cannot identify the guest, so return NOTHING rather
            # than leak every guest's history.
            return AgentRun.client_key == client_key if client_key else false()
        return AgentRun.firm_id == firm_id

    async def list_conversations(
        self,
        *,
        firm_id: str,
        user_id: uuid.UUID | None,
        client_key: str | None = None,
        limit: int = 30,
        offset: int = 0,
        q: str | None = None,
        archived: bool = False,
    ) -> list[dict[str, Any]]:
        scope = self._scope(firm_id=firm_id, user_id=user_id, client_key=client_key)
        visible = AgentRun.hidden_at.is_(None)  # exclude soft-deleted runs

        conds = [scope, visible]
        # Archived view vs the default list. The LEFT JOIN below means a session
        # with no overlay row has archived_at = NULL → treated as not-archived.
        conds.append(
            ChatConversation.archived_at.is_not(None)
            if archived
            else ChatConversation.archived_at.is_(None)
        )
        # Optional full-text-ish search: keep a session if ANY of its runs match
        # the query in the question or the answer, or its (renamed) title matches.
        # The agg's own `scope` still gates ownership, so the title subquery
        # needn't re-scope. ILIKE is fine at MVP volume (add a Postgres FTS index
        # later if history grows large).
        if q and q.strip():
            pattern = f"%{q.strip()}%"
            content_ids = select(AgentRun.session_id).where(
                scope,
                visible,
                or_(AgentRun.user_input.ilike(pattern), AgentRun.final_answer.ilike(pattern)),
            )
            title_ids = select(ChatConversation.session_id).where(
                ChatConversation.title.ilike(pattern)
            )
            conds.append(AgentRun.session_id.in_(content_ids.union(title_ids)))

        # 1) Pinned-first, then most-recently-active, with turn counts. The overlay
        #    is LEFT-joined so we get is_pinned / archived_at / renamed title in one
        #    query (and can paginate with limit/offset).
        agg = (
            select(
                AgentRun.session_id,
                func.max(AgentRun.created_at).label("last_activity"),
                func.count().label("turns"),
                ChatConversation.is_pinned.label("is_pinned"),
                ChatConversation.title.label("overlay_title"),
            )
            .select_from(AgentRun)
            .join(
                ChatConversation,
                ChatConversation.session_id == AgentRun.session_id,
                isouter=True,
            )
            .where(*conds)
            .group_by(AgentRun.session_id, ChatConversation.is_pinned, ChatConversation.title)
            .order_by(
                func.coalesce(ChatConversation.is_pinned, false()).desc(),
                func.max(AgentRun.created_at).desc(),
            )
            .limit(limit)
            .offset(offset)
        )
        rows = (await self._session.execute(agg)).all()
        if not rows:
            return []
        session_ids = [r.session_id for r in rows]
        meta = {r.session_id: r for r in rows}

        # 2) Per-session derived title (first user message) + preview (latest answer).
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

        out: list[dict[str, Any]] = []
        for sid in session_ids:
            r = meta[sid]
            out.append(
                {
                    "session_id": sid,
                    # Renamed (overlay) title wins; else the first user message.
                    "title": (r.overlay_title or title.get(sid) or "Untitled")[:_TITLE_MAX],
                    "turns": r.turns,
                    "last_activity": r.last_activity,
                    "preview": (preview.get(sid) or "")[:_PREVIEW_MAX],
                    "agent_name": agent.get(sid),
                    "is_pinned": bool(r.is_pinned),
                }
            )
        return out

    async def get_conversation(
        self,
        *,
        session_id: str,
        firm_id: str,
        user_id: uuid.UUID | None,
        client_key: str | None = None,
    ) -> list[AgentRun]:
        scope = self._scope(firm_id=firm_id, user_id=user_id, client_key=client_key)
        q = (
            select(AgentRun)
            .where(AgentRun.session_id == session_id)
            .where(scope, AgentRun.hidden_at.is_(None))
            .order_by(AgentRun.created_at.asc())
        )
        return list((await self._session.execute(q)).scalars().all())

    async def hide_conversation(
        self,
        *,
        session_id: str,
        firm_id: str,
        user_id: uuid.UUID | None,
        client_key: str | None = None,
    ) -> int:
        """Soft-delete: mark every (still-visible) run in the session as hidden.
        The audit rows are preserved. Returns how many runs were hidden."""
        scope = self._scope(firm_id=firm_id, user_id=user_id, client_key=client_key)
        stmt = (
            update(AgentRun)
            .where(AgentRun.session_id == session_id, scope, AgentRun.hidden_at.is_(None))
            .values(hidden_at=datetime.now(timezone.utc))
        )
        res = await self._session.execute(stmt)
        await self._session.commit()
        return res.rowcount or 0

    async def _owns(
        self, *, session_id: str, firm_id: str, user_id: uuid.UUID | None, client_key: str | None
    ) -> bool:
        """Whether the caller owns this session (has ≥1 visible run in scope) —
        so users can't mutate others' conversations."""
        owns = await self._session.scalar(
            select(AgentRun.id)
            .where(
                AgentRun.session_id == session_id,
                self._scope(firm_id=firm_id, user_id=user_id, client_key=client_key),
            )
            .limit(1)
        )
        return owns is not None

    async def _upsert_overlay(
        self, *, session_id: str, firm_id: str, user_id: uuid.UUID | None, values: dict[str, Any]
    ) -> None:
        """Create-or-update the overlay row for this session with ``values``
        (title / is_pinned / archived_at). A new row leaves the others at their
        defaults (title NULL → list falls back to the first message)."""
        stmt = (
            pg_insert(ChatConversation)
            .values(session_id=session_id, firm_id=firm_id, user_id=user_id, **values)
            .on_conflict_do_update(index_elements=[ChatConversation.session_id], set_=values)
        )
        await self._session.execute(stmt)
        await self._session.commit()

    async def set_title(
        self,
        *,
        session_id: str,
        firm_id: str,
        user_id: uuid.UUID | None,
        title: str,
        client_key: str | None = None,
    ) -> bool:
        """Rename a conversation (overlay). False if not the caller's."""
        if not await self._owns(
            session_id=session_id, firm_id=firm_id, user_id=user_id, client_key=client_key
        ):
            return False
        await self._upsert_overlay(
            session_id=session_id, firm_id=firm_id, user_id=user_id, values={"title": title}
        )
        return True

    async def set_pinned(
        self,
        *,
        session_id: str,
        firm_id: str,
        user_id: uuid.UUID | None,
        pinned: bool,
        client_key: str | None = None,
    ) -> bool:
        """Pin/unpin a conversation (sticks it to the top). False if not the caller's."""
        if not await self._owns(
            session_id=session_id, firm_id=firm_id, user_id=user_id, client_key=client_key
        ):
            return False
        await self._upsert_overlay(
            session_id=session_id, firm_id=firm_id, user_id=user_id, values={"is_pinned": pinned}
        )
        return True

    async def set_archived(
        self,
        *,
        session_id: str,
        firm_id: str,
        user_id: uuid.UUID | None,
        archived: bool,
        client_key: str | None = None,
    ) -> bool:
        """Archive/unarchive a conversation (hidden from the default list). False
        if not the caller's."""
        if not await self._owns(
            session_id=session_id, firm_id=firm_id, user_id=user_id, client_key=client_key
        ):
            return False
        value = datetime.now(timezone.utc) if archived else None
        await self._upsert_overlay(
            session_id=session_id, firm_id=firm_id, user_id=user_id, values={"archived_at": value}
        )
        return True

    # ── Read-only public share ───────────────────────────────────────────────

    async def create_or_get_share(
        self,
        *,
        session_id: str,
        firm_id: str,
        user_id: uuid.UUID | None,
        client_key: str | None = None,
    ) -> dict[str, Any] | None:
        """Owner-only. Return the conversation's active share (idempotent): if a
        non-revoked token already exists, return it unchanged; otherwise mint a
        token and FREEZE the current set of visible turns into ``shared_run_ids``
        so later private turns never leak into the link. ``None`` if not the
        caller's conversation."""
        if not await self._owns(
            session_id=session_id, firm_id=firm_id, user_id=user_id, client_key=client_key
        ):
            return None
        # Reuse an existing, un-revoked share so the link is stable.
        existing = (
            await self._session.execute(
                select(
                    ChatConversation.share_token, ChatConversation.shared_at
                ).where(
                    ChatConversation.session_id == session_id,
                    ChatConversation.share_token.is_not(None),
                    ChatConversation.revoked_at.is_(None),
                )
            )
        ).first()
        if existing is not None and existing.share_token:
            return {"token": existing.share_token, "shared_at": existing.shared_at}

        # Freeze the exact turns shared (visible runs in this session, in order).
        run_ids = (
            await self._session.execute(
                select(AgentRun.id)
                .where(
                    AgentRun.session_id == session_id,
                    AgentRun.hidden_at.is_(None),
                    self._scope(firm_id=firm_id, user_id=user_id, client_key=client_key),
                )
                .order_by(AgentRun.created_at.asc())
            )
        ).scalars().all()
        token = secrets.token_urlsafe(24)
        now = datetime.now(timezone.utc)
        await self._upsert_overlay(
            session_id=session_id,
            firm_id=firm_id,
            user_id=user_id,
            values={
                "share_token": token,
                "shared_at": now,
                "revoked_at": None,
                "shared_run_ids": [str(r) for r in run_ids],
            },
        )
        return {"token": token, "shared_at": now}

    async def revoke_share(
        self,
        *,
        session_id: str,
        firm_id: str,
        user_id: uuid.UUID | None,
        client_key: str | None = None,
    ) -> bool:
        """Owner-only. Kill the public link (the token stops resolving). Keeps the
        overlay row. ``False`` if not the caller's conversation."""
        if not await self._owns(
            session_id=session_id, firm_id=firm_id, user_id=user_id, client_key=client_key
        ):
            return False
        await self._upsert_overlay(
            session_id=session_id,
            firm_id=firm_id,
            user_id=user_id,
            values={"revoked_at": datetime.now(timezone.utc)},
        )
        return True

    async def get_shared_snapshot(self, token: str) -> dict[str, Any] | None:
        """PUBLIC (no principal scope). Resolve a share token → the frozen,
        read-only snapshot: ``{title, shared_at, runs}``. ``None`` if the token is
        unknown, revoked, or every shared turn has since been deleted."""
        overlay = (
            await self._session.execute(
                select(
                    ChatConversation.session_id,
                    ChatConversation.title,
                    ChatConversation.shared_at,
                    ChatConversation.shared_run_ids,
                ).where(
                    ChatConversation.share_token == token,
                    ChatConversation.revoked_at.is_(None),
                )
            )
        ).first()
        if overlay is None:
            return None
        run_ids = [uuid.UUID(r) for r in (overlay.shared_run_ids or [])]
        if not run_ids:
            return None
        # Fetch the EXACT frozen turns by id (still honour hidden_at so deleting
        # the conversation kills the share). No principal scope — the ids were
        # owner-captured at share time, so this is safe and public.
        runs = list(
            (
                await self._session.execute(
                    select(AgentRun)
                    .where(AgentRun.id.in_(run_ids), AgentRun.hidden_at.is_(None))
                    .order_by(AgentRun.created_at.asc())
                )
            )
            .scalars()
            .all()
        )
        if not runs:
            return None
        title = overlay.title or (runs[0].user_input or "Shared conversation")[:_TITLE_MAX]
        return {"title": title, "shared_at": overlay.shared_at, "runs": runs}

    async def upsert_feedback(
        self,
        *,
        agent_run_id: uuid.UUID,
        firm_id: str,
        user_id: uuid.UUID | None,
        rating: int,
        reasons: list[str],
        comment: str | None,
        client_key: str | None = None,
    ) -> bool:
        """Record (or replace) the caller's rating of one answer. False if the
        run isn't the caller's (can't rate someone else's turn). One row per run
        — re-rating upserts."""
        owns = await self._session.scalar(
            select(AgentRun.id)
            .where(
                AgentRun.id == agent_run_id,
                self._scope(firm_id=firm_id, user_id=user_id, client_key=client_key),
            )
            .limit(1)
        )
        if owns is None:
            return False
        stmt = (
            pg_insert(MessageFeedback)
            .values(
                agent_run_id=agent_run_id,
                firm_id=firm_id,
                user_id=user_id,
                client_key=client_key,
                rating=rating,
                reasons=reasons,
                comment=comment,
            )
            .on_conflict_do_update(
                constraint="uq_message_feedback_agent_run",
                set_={
                    "rating": rating,
                    "reasons": reasons,
                    "comment": comment,
                    "user_id": user_id,
                    "client_key": client_key,
                },
            )
        )
        await self._session.execute(stmt)
        await self._session.commit()
        return True

    async def clear_feedback(
        self,
        *,
        agent_run_id: uuid.UUID,
        firm_id: str,
        user_id: uuid.UUID | None,
        client_key: str | None = None,
    ) -> bool:
        """Remove the caller's rating of one answer (toggle 👍/👎 back to
        neutral). False if the run isn't the caller's; idempotent otherwise
        (no row → success)."""
        owns = await self._session.scalar(
            select(AgentRun.id)
            .where(
                AgentRun.id == agent_run_id,
                self._scope(firm_id=firm_id, user_id=user_id, client_key=client_key),
            )
            .limit(1)
        )
        if owns is None:
            return False
        await self._session.execute(
            delete(MessageFeedback).where(MessageFeedback.agent_run_id == agent_run_id)
        )
        await self._session.commit()
        return True

    async def get_feedback_for_runs(
        self, run_ids: list[uuid.UUID]
    ) -> dict[uuid.UUID, dict[str, Any]]:
        """``{agent_run_id: {rating, reasons, comment}}`` for the given runs —
        used to surface each turn's rating on replay."""
        if not run_ids:
            return {}
        rows = await self._session.execute(
            select(
                MessageFeedback.agent_run_id,
                MessageFeedback.rating,
                MessageFeedback.reasons,
                MessageFeedback.comment,
            ).where(MessageFeedback.agent_run_id.in_(run_ids))
        )
        return {
            r.agent_run_id: {"rating": r.rating, "reasons": r.reasons or [], "comment": r.comment}
            for r in rows.all()
        }
