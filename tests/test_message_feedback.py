"""Phase 5 — per-answer feedback (👍/👎).

Verifies the `message_feedback` overlay: a caller can rate one of their answers,
re-rating upserts (one row per run), reasons/comment persist, the rating surfaces
for replay, and a caller cannot rate an answer that isn't theirs (scope guard).
DB-backed (needs the test Postgres with migration 0018 applied).
"""

from __future__ import annotations

import uuid

import pytest

from src.models.agent_run import AgentRun
from src.repositories.conversation_repo import ConversationRepository

FIRM = "FBTESTFIRM"
OTHER_FIRM = "FBOTHERFIRM"


def _run(session_id: str, firm_id: str = FIRM) -> AgentRun:
    return AgentRun(
        firm_id=firm_id,
        session_id=session_id,
        agent_name="company_intel",
        user_input="what are TCS margins?",
        final_answer="ok",
        status="complete",
    )


@pytest.mark.asyncio
async def test_upsert_and_surface_rating(db_session) -> None:
    run = _run("fb1")
    db_session.add(run)
    await db_session.flush()
    repo = ConversationRepository(db_session)

    # 👎 with reasons + comment.
    assert await repo.upsert_feedback(
        agent_run_id=run.id,
        firm_id=FIRM,
        user_id=None,
        rating=-1,
        reasons=["Inaccurate", "Wrong source"],
        comment="cited the wrong filing",
    )
    fb = await repo.get_feedback_for_runs([run.id])
    assert fb[run.id]["rating"] == -1
    assert fb[run.id]["reasons"] == ["Inaccurate", "Wrong source"]
    assert fb[run.id]["comment"] == "cited the wrong filing"


@pytest.mark.asyncio
async def test_re_rating_upserts_one_row(db_session) -> None:
    run = _run("fb2")
    db_session.add(run)
    await db_session.flush()
    repo = ConversationRepository(db_session)

    assert await repo.upsert_feedback(
        agent_run_id=run.id, firm_id=FIRM, user_id=None, rating=-1,
        reasons=["Incomplete"], comment=None,
    )
    # Change their mind → 👍. One row per run (unique constraint) → overwrites.
    assert await repo.upsert_feedback(
        agent_run_id=run.id, firm_id=FIRM, user_id=None, rating=1,
        reasons=[], comment=None,
    )
    fb = await repo.get_feedback_for_runs([run.id])
    assert fb[run.id]["rating"] == 1
    assert fb[run.id]["reasons"] == []


@pytest.mark.asyncio
async def test_cannot_rate_another_principals_answer(db_session) -> None:
    run = _run("fb3", firm_id=OTHER_FIRM)
    db_session.add(run)
    await db_session.flush()
    repo = ConversationRepository(db_session)

    # FIRM tries to rate OTHER_FIRM's run → refused, nothing recorded.
    assert not await repo.upsert_feedback(
        agent_run_id=run.id, firm_id=FIRM, user_id=None, rating=-1,
        reasons=[], comment=None,
    )
    assert await repo.get_feedback_for_runs([run.id]) == {}


@pytest.mark.asyncio
async def test_unknown_run_returns_false(db_session) -> None:
    repo = ConversationRepository(db_session)
    assert not await repo.upsert_feedback(
        agent_run_id=uuid.uuid4(), firm_id=FIRM, user_id=None, rating=1,
        reasons=[], comment=None,
    )


@pytest.mark.asyncio
async def test_get_feedback_empty_for_unrated(db_session) -> None:
    run = _run("fb4")
    db_session.add(run)
    await db_session.flush()
    repo = ConversationRepository(db_session)
    assert await repo.get_feedback_for_runs([run.id]) == {}
    assert await repo.get_feedback_for_runs([]) == {}


@pytest.mark.asyncio
async def test_clear_feedback_toggles_back_to_neutral(db_session) -> None:
    run = _run("fb5")
    db_session.add(run)
    await db_session.flush()
    repo = ConversationRepository(db_session)
    await repo.upsert_feedback(
        agent_run_id=run.id, firm_id=FIRM, user_id=None, rating=1, reasons=[], comment=None
    )
    # Clear → the row is removed (back to neutral). Idempotent on a second call.
    assert await repo.clear_feedback(agent_run_id=run.id, firm_id=FIRM, user_id=None)
    assert await repo.get_feedback_for_runs([run.id]) == {}
    assert await repo.clear_feedback(agent_run_id=run.id, firm_id=FIRM, user_id=None)


@pytest.mark.asyncio
async def test_cannot_clear_another_principals_feedback(db_session) -> None:
    run = _run("fb6", firm_id=OTHER_FIRM)
    db_session.add(run)
    await db_session.flush()
    repo = ConversationRepository(db_session)
    # FIRM can't clear a rating on OTHER_FIRM's run.
    assert not await repo.clear_feedback(agent_run_id=run.id, firm_id=FIRM, user_id=None)
