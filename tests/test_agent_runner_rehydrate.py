"""Phase 1B — context continuity on resume.

Unit tests for ``AgentRunner`` session rehydration (DB-free): the principal
scope clause (security) and seeding a fresh ADK session from the persisted
transcript (the ADK-API-sensitive part). The full end-to-end (refresh →
contextual follow-up) is a live integration check.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from src.auth.principal import ANONYMOUS_FIRM
from src.models.agent_run import AgentRun
from src.services.agent_runner import AgentRunner


def _runner(**kw) -> AgentRunner:
    agent = SimpleNamespace(name="company_intel", model=None, model_tier="fast")
    return AgentRunner(agent=agent, firm_id=kw.pop("firm_id", "QUANTSOFT"), **kw)


def test_scope_clause_signed_in_by_user() -> None:
    uid = uuid.uuid4()
    clause = _runner(user_id=uid)._scope_clause()
    assert "agent_runs.user_id" in str(clause)


def test_scope_clause_guest_by_client_key() -> None:
    clause = _runner(firm_id=ANONYMOUS_FIRM, client_key="guest-123")._scope_clause()
    assert "agent_runs.client_key" in str(clause)


def test_scope_clause_guest_without_client_key_matches_nothing() -> None:
    # No client_key for an anonymous principal → must fail safe (match nothing),
    # never fall through to a broad firm match that could leak guest history.
    clause = _runner(firm_id=ANONYMOUS_FIRM, client_key=None)._scope_clause()
    assert "agent_runs.firm_id" not in str(clause)


def test_scope_clause_dev_by_firm() -> None:
    clause = _runner(firm_id="QUANTSOFT")._scope_clause()
    assert "agent_runs.firm_id" in str(clause)
    assert AgentRun.firm_id is not None  # sanity


@pytest.mark.asyncio
async def test_rehydrate_seeds_prior_turns_into_adk_session() -> None:
    from google.adk.sessions import InMemorySessionService

    runner = _runner()

    async def _fake_prior():
        return [("How is TCS doing?", "TCS looks bullish."), ("And its margins?", "Margins improved YoY.")]

    runner._load_prior_turns = _fake_prior  # type: ignore[assignment]

    svc = InMemorySessionService()
    sess = await svc.create_session(app_name="prism", user_id="QUANTSOFT", session_id="sess_x")
    await runner._rehydrate_session(svc, sess)

    got = await svc.get_session(app_name="prism", user_id="QUANTSOFT", session_id="sess_x")
    assert len(got.events) == 4  # 2 turns × (user + model)
    assert got.events[0].author == "user" and got.events[0].content.role == "user"
    assert got.events[0].content.parts[0].text == "How is TCS doing?"
    assert got.events[1].author == "company_intel" and got.events[1].content.role == "model"
    assert got.events[1].content.parts[0].text == "TCS looks bullish."


@pytest.mark.asyncio
async def test_rehydrate_noop_when_no_prior_turns() -> None:
    from google.adk.sessions import InMemorySessionService

    runner = _runner()

    async def _none():
        return []

    runner._load_prior_turns = _none  # type: ignore[assignment]

    svc = InMemorySessionService()
    sess = await svc.create_session(app_name="prism", user_id="QUANTSOFT", session_id="sess_y")
    await runner._rehydrate_session(svc, sess)

    got = await svc.get_session(app_name="prism", user_id="QUANTSOFT", session_id="sess_y")
    assert len(got.events) == 0  # genuinely new chat — nothing seeded
