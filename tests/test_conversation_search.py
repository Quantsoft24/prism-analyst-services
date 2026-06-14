"""Phase 2 — conversation search.

Verifies ``ConversationRepository.list_conversations(q=…)`` filters by question
text AND answer text (not just title), scoped. Uses the rolled-back ``db_session``
fixture (needs the test Postgres with migrations applied).
"""

from __future__ import annotations

import pytest

from src.models.agent_run import AgentRun
from src.repositories.conversation_repo import ConversationRepository

FIRM = "SEARCHTESTFIRM"


def _run(session_id: str, user_input: str, final_answer: str) -> AgentRun:
    return AgentRun(
        firm_id=FIRM,
        session_id=session_id,
        agent_name="company_intel",
        user_input=user_input,
        final_answer=final_answer,
        status="complete",
    )


@pytest.mark.asyncio
async def test_search_matches_question_and_answer(db_session) -> None:
    db_session.add_all(
        [
            _run("ss1", "How is TCS doing on margins?", "TCS margins improved YoY."),
            _run("ss2", "What about Reliance refining?", "Reliance refining was strong."),
        ]
    )
    await db_session.flush()
    repo = ConversationRepository(db_session)

    # match on the QUESTION text
    by_q = await repo.list_conversations(firm_id=FIRM, user_id=None, q="margins")
    assert {c["session_id"] for c in by_q} == {"ss1"}

    # match on the ANSWER text (not just the title/first message)
    by_a = await repo.list_conversations(firm_id=FIRM, user_id=None, q="refining")
    assert {c["session_id"] for c in by_a} == {"ss2"}

    # no query → both conversations
    allc = await repo.list_conversations(firm_id=FIRM, user_id=None)
    assert {c["session_id"] for c in allc} == {"ss1", "ss2"}

    # no match → empty
    none = await repo.list_conversations(firm_id=FIRM, user_id=None, q="zzz-no-match")
    assert none == []

    # blank query is ignored (treated as no filter)
    blank = await repo.list_conversations(firm_id=FIRM, user_id=None, q="   ")
    assert {c["session_id"] for c in blank} == {"ss1", "ss2"}
