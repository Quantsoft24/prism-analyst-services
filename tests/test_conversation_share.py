"""Phase 9 — read-only public share link.

Verifies the share lifecycle: owner mints a token (idempotent), the snapshot is
FROZEN at share time (continuing privately never leaks), non-owners can't share,
revoke kills the link, and re-sharing mints a fresh token. DB-backed (needs the
test Postgres with migration 0019 applied).
"""

from __future__ import annotations

import pytest

from src.models.agent_run import AgentRun
from src.repositories.conversation_repo import ConversationRepository

FIRM = "SHARETESTFIRM"
OTHER = "SHAREOTHERFIRM"
SESS = "sharetest_sess"


def _run(ui: str, fa: str, firm_id: str = FIRM) -> AgentRun:
    return AgentRun(
        firm_id=firm_id, session_id=SESS, agent_name="company_intel",
        user_input=ui, final_answer=fa, status="complete",
    )


@pytest.mark.asyncio
async def test_create_is_idempotent_and_snapshots_turns(db_session) -> None:
    db_session.add_all([_run("q1", "a1"), _run("q2", "a2")])
    await db_session.flush()
    repo = ConversationRepository(db_session)

    sh = await repo.create_or_get_share(session_id=SESS, firm_id=FIRM, user_id=None)
    assert sh and sh["token"]
    # Same link on re-share (stable URL).
    sh2 = await repo.create_or_get_share(session_id=SESS, firm_id=FIRM, user_id=None)
    assert sh2["token"] == sh["token"]

    snap = await repo.get_shared_snapshot(sh["token"])
    assert snap is not None
    assert [t.user_input for t in snap["runs"]] == ["q1", "q2"]
    assert snap["title"] == "q1"


@pytest.mark.asyncio
async def test_snapshot_is_frozen_continuing_privately_does_not_leak(db_session) -> None:
    db_session.add_all([_run("q1", "a1"), _run("q2", "a2")])
    await db_session.flush()
    repo = ConversationRepository(db_session)
    sh = await repo.create_or_get_share(session_id=SESS, firm_id=FIRM, user_id=None)

    # Continue the conversation AFTER sharing → must not appear in the link.
    db_session.add(_run("q3-private", "a3"))
    await db_session.flush()

    snap = await repo.get_shared_snapshot(sh["token"])
    assert [t.user_input for t in snap["runs"]] == ["q1", "q2"]  # still frozen at 2


@pytest.mark.asyncio
async def test_non_owner_cannot_share(db_session) -> None:
    db_session.add(_run("q1", "a1"))
    await db_session.flush()
    repo = ConversationRepository(db_session)
    # A different firm can't mint a share for this conversation.
    assert await repo.create_or_get_share(session_id=SESS, firm_id=OTHER, user_id=None) is None


@pytest.mark.asyncio
async def test_revoke_kills_link_and_reshare_mints_new_token(db_session) -> None:
    db_session.add(_run("q1", "a1"))
    await db_session.flush()
    repo = ConversationRepository(db_session)
    sh = await repo.create_or_get_share(session_id=SESS, firm_id=FIRM, user_id=None)

    assert await repo.revoke_share(session_id=SESS, firm_id=FIRM, user_id=None)
    assert await repo.get_shared_snapshot(sh["token"]) is None  # revoked → gone

    sh2 = await repo.create_or_get_share(session_id=SESS, firm_id=FIRM, user_id=None)
    assert sh2["token"] != sh["token"]  # re-share mints a fresh token
    assert await repo.get_shared_snapshot(sh2["token"]) is not None


@pytest.mark.asyncio
async def test_unknown_token_returns_none(db_session) -> None:
    repo = ConversationRepository(db_session)
    assert await repo.get_shared_snapshot("does-not-exist") is None


@pytest.mark.asyncio
async def test_deleting_conversation_kills_share(db_session) -> None:
    db_session.add_all([_run("q1", "a1")])
    await db_session.flush()
    repo = ConversationRepository(db_session)
    sh = await repo.create_or_get_share(session_id=SESS, firm_id=FIRM, user_id=None)

    # Hiding (soft-delete) every run → the shared snapshot has nothing to show.
    await repo.hide_conversation(session_id=SESS, firm_id=FIRM, user_id=None)
    assert await repo.get_shared_snapshot(sh["token"]) is None
