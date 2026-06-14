"""Phase 3 — pin / archive / pagination.

Verifies the conversation overlay flags: pinned conversations sort first,
archived ones drop out of the default list and appear in the archived view, and
`is_pinned` surfaces in the summary. DB-backed (needs the test Postgres with
migration 0017 applied).
"""

from __future__ import annotations

import pytest

from src.models.agent_run import AgentRun
from src.repositories.conversation_repo import ConversationRepository

FIRM = "PINTESTFIRM"


def _run(session_id: str, user_input: str) -> AgentRun:
    return AgentRun(
        firm_id=FIRM,
        session_id=session_id,
        agent_name="company_intel",
        user_input=user_input,
        final_answer="ok",
        status="complete",
    )


@pytest.mark.asyncio
async def test_pin_sorts_first_and_surfaces_flag(db_session) -> None:
    db_session.add_all([_run("pa1", "first chat"), _run("pa2", "second chat")])
    await db_session.flush()
    repo = ConversationRepository(db_session)

    # pa2 is more recent insert but both share ~same time; pin pa1 → it leads.
    assert await repo.set_pinned(firm_id=FIRM, user_id=None, session_id="pa1", pinned=True)
    listed = await repo.list_conversations(firm_id=FIRM, user_id=None)
    assert listed[0]["session_id"] == "pa1"
    assert listed[0]["is_pinned"] is True
    assert any(c["session_id"] == "pa2" and c["is_pinned"] is False for c in listed)


@pytest.mark.asyncio
async def test_archive_excludes_from_default_and_shows_in_archived(db_session) -> None:
    db_session.add_all([_run("ar1", "keep me"), _run("ar2", "archive me")])
    await db_session.flush()
    repo = ConversationRepository(db_session)

    assert await repo.set_archived(firm_id=FIRM, user_id=None, session_id="ar2", archived=True)

    default = {c["session_id"] for c in await repo.list_conversations(firm_id=FIRM, user_id=None)}
    assert default == {"ar1"}  # archived one is hidden

    archived = {
        c["session_id"]
        for c in await repo.list_conversations(firm_id=FIRM, user_id=None, archived=True)
    }
    assert archived == {"ar2"}  # only the archived one

    # Unarchive → back in the default list.
    assert await repo.set_archived(firm_id=FIRM, user_id=None, session_id="ar2", archived=False)
    back = {c["session_id"] for c in await repo.list_conversations(firm_id=FIRM, user_id=None)}
    assert back == {"ar1", "ar2"}


@pytest.mark.asyncio
async def test_offset_pagination(db_session) -> None:
    db_session.add_all([_run(f"pg{i}", f"chat {i}") for i in range(5)])
    await db_session.flush()
    repo = ConversationRepository(db_session)

    page1 = await repo.list_conversations(firm_id=FIRM, user_id=None, limit=2, offset=0)
    page2 = await repo.list_conversations(firm_id=FIRM, user_id=None, limit=2, offset=2)
    assert len(page1) == 2 and len(page2) == 2
    assert {c["session_id"] for c in page1}.isdisjoint({c["session_id"] for c in page2})
