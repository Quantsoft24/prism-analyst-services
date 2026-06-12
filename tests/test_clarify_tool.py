"""Tests for the agent-driven clarification tool — shape + security_id guard."""

from __future__ import annotations

from src.services import company_resolver as cr
from src.tools.clarify_tool import CLARIFY_MARKER, request_clarification


def _grp(sid):
    return cr.CompanyGroup(
        isin="INE002A01018", name="Reliance Industries Ltd.", symbol="RELIANCE",
        sector="Oil, Gas & Consumable Fuels", security_id_nse=sid,
    )


async def test_single_select_keeps_valid_options(monkeypatch):
    async def fake_get(sid):
        return _grp(sid)

    monkeypatch.setattr(cr, "get_by_security_id", fake_get)
    out = await request_clarification(
        "Which Reliance did you mean?",
        [{"label": "Reliance Industries Ltd.", "value": 2228, "hint": "RELIANCE · NSE/BSE"}],
        "single_select",
    )
    payload = out[CLARIFY_MARKER]
    assert payload["mode"] == "single_select"
    assert payload["allow_search"] is True
    assert payload["options"][0]["value"] == 2228
    assert payload["options"][0]["label"] == "Reliance Industries Ltd."


async def test_unknown_security_id_dropped_falls_back_to_open_text(monkeypatch):
    async def fake_get(_sid):
        return None  # id not in the master → must be dropped (can't mis-route)

    monkeypatch.setattr(cr, "get_by_security_id", fake_get)
    out = await request_clarification("pick", [{"label": "Bogus", "value": 999999}], "single_select")
    payload = out[CLARIFY_MARKER]
    assert payload["options"] == []
    assert payload["mode"] == "open_text"  # no valid options → still ask, as text


async def test_open_text_needs_no_options():
    out = await request_clarification("Which fiscal year — FY25 or FY26?", None, "open_text")
    payload = out[CLARIFY_MARKER]
    assert payload["mode"] == "open_text"
    assert payload["question"].startswith("Which fiscal year")


async def test_bad_mode_defaults_to_single_select(monkeypatch):
    async def fake_get(sid):
        return _grp(sid)

    monkeypatch.setattr(cr, "get_by_security_id", fake_get)
    out = await request_clarification("?", [{"label": "X", "value": 1}], "nonsense_mode")
    assert out[CLARIFY_MARKER]["mode"] == "single_select"


def test_clarification_event_serializes_for_sse():
    """The runner emits ClarificationEvent; the chat router must serialize it to
    an `event: clarification` SSE frame the frontend can dispatch on."""
    import json
    import uuid

    from src.routers.chat import _serialize_event
    from src.schemas.chat import ClarificationEvent, ClarificationOption

    ev = ClarificationEvent(
        agent_run_id=uuid.uuid4(),
        question="Which Reliance did you mean?",
        mode="single_select",
        options=[
            ClarificationOption(id="2228", label="Reliance Industries Ltd.",
                                hint="RELIANCE · NSE/BSE", value=2228),
        ],
    )
    frame = _serialize_event(ev)
    assert frame["event"] == "clarification"
    data = json.loads(frame["data"])
    assert data["type"] == "clarification"
    assert data["mode"] == "single_select"
    assert data["allow_search"] is True
    assert data["options"][0]["value"] == 2228
