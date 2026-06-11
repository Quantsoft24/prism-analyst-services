"""Tests for the stock-chat wrapper's v3 contract, response-shape routing, and
HTTP error handling.

Pure unit tests — no live service. ``httpx.AsyncClient`` is replaced with a
fake that yields scripted responses (or raises transport errors), so we can
exercise all response shapes plus the transport/HTTP failure paths without a
network call. Mirrors the test_prism_financials.py pattern.
"""

from __future__ import annotations

import asyncio

import httpx

from src.integrations.tools import stock_chat as sc
from src.integrations.tools._errors import is_error


class _FakeResp:
    def __init__(self, status_code: int = 200, json_data: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text

    def json(self) -> dict:
        return self._json


def _make_client_factory(behaviors: list, calls: list):
    """Return a fake ``AsyncClient`` class. Each ``post`` consumes the next
    behavior — either a ``_FakeResp`` to return or an ``Exception`` to raise.
    The wrapper builds a fresh client per retry attempt, so one behavior maps
    to one attempt."""
    it = iter(behaviors)

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> bool:
            return False

        async def post(self, url, json=None, headers=None):
            calls.append({"url": url, "json": json})
            behavior = next(it)
            if isinstance(behavior, Exception):
                raise behavior
            return behavior

    return _Client


async def _noop_sleep(*_a, **_k) -> None:
    return None


def _invoke(monkeypatch, behaviors: list, **kwargs):
    """Patch httpx + the retry sleep, then run stock_filings_read.
    Returns ``(result, calls)``."""
    calls: list = []
    monkeypatch.setattr(sc.httpx, "AsyncClient", _make_client_factory(behaviors, calls))
    monkeypatch.setattr(sc.asyncio, "sleep", _noop_sleep)
    kwargs.setdefault("question", "q")
    result = asyncio.run(sc.stock_filings_read(**kwargs))
    return result, calls


# ── Payload construction tests ─────────────────────────────────────────────


def test_question_only_sends_minimal_payload(monkeypatch) -> None:
    """When only ``question`` is given, the payload has exactly 2 keys."""
    _, calls = _invoke(
        monkeypatch,
        [_FakeResp(200, {"answer": "test", "selected_filings": []})],
        question="What did ITC discuss about sustainability?",
    )
    sent = calls[0]["json"]
    assert sent["question"] == "What did ITC discuss about sustainability?"
    # Default synthesise=False → upstream returns evidence passages (with pdf_url
    # + page) that our agent composes the answer from; enables citation deep-links.
    assert sent["synthesise"] is False
    assert "company" not in sent
    # The forbidden v2 fields must NEVER appear in the payload.
    for forbidden in ("category", "period", "date_from", "date_to", "max_filings",
                      "companies", "text_match", "industry"):
        assert forbidden not in sent, f"Forbidden field {forbidden!r} found in payload"


def test_single_security_id_passed(monkeypatch) -> None:
    _, calls = _invoke(
        monkeypatch,
        [_FakeResp(200, {"answer": "test", "selected_filings": []})],
        question="What risks did they flag?",
        security_id=2718,
    )
    assert calls[0]["json"]["security_id"] == 2718
    assert "company" not in calls[0]["json"]


def test_security_ids_list_passed(monkeypatch) -> None:
    _, calls = _invoke(
        monkeypatch,
        [_FakeResp(200, {"answer": "test", "selected_filings": []})],
        question="Compare their board outcomes",
        security_ids=[1259, 1081],
    )
    assert calls[0]["json"]["security_ids"] == [1259, 1081]
    # security_ids takes precedence — no single security_id key.
    assert "security_id" not in calls[0]["json"]


def test_synthesise_false_forwarded(monkeypatch) -> None:
    _, calls = _invoke(
        monkeypatch,
        [_FakeResp(200, {"answer": None, "evidence": [{"quote": "x"}], "selected_filings": []})],
        question="q",
        synthesise=False,
    )
    assert calls[0]["json"]["synthesise"] is False


def test_only_question_synthesise_security_id_in_payload(monkeypatch) -> None:
    """The payload carries only question + synthesise + the resolved id —
    catalog filters stay out (the service's planner derives them)."""
    _, calls = _invoke(
        monkeypatch,
        [_FakeResp(200, {"answer": "ok", "selected_filings": []})],
        question="q",
        security_id=2228,
    )
    sent = calls[0]["json"]
    assert set(sent.keys()) == {"question", "synthesise", "security_id"}


# ── Response shape tests ───────────────────────────────────────────────────


def _full_response():
    """A realistic v3 response with all fields."""
    return {
        "question": "What did ITC say about sustainability?",
        "plan": {"companies": ["ITC"], "category": "Annual Report", "rationale": "specific AR"},
        "resolved_companies": ["ITC Ltd"],
        "candidates_considered": 3,
        "needs_clarification": False,
        "clarification_question": None,
        "selected_filings": [
            {
                "newsid": "n1", "company_name": "ITC Ltd",
                "category": "Annual Report", "headline": "Annual Report 2025",
                "announcement_dt": "2025-06-27T10:00:00",
                "why_selected": "latest AR", "read_ok": True, "from_cache": True,
                "page_count": 428, "sections_read": ["Sustainability"],
                "is_scanned": False, "error": None,
            }
        ],
        "answer": "ITC discussed sustainability... [ITC Ltd | p.6]",
        "evidence": [
            {"newsid": "n1", "company_name": "ITC Ltd", "page": 6,
             "quote": "sustainability quote", "citation": "[ITC Ltd | p.6]"}
        ],
        "document_excerpts": [{"newsid": "n1", "text": "very long text..."}],
        "token_usage": {"plan_in": 320, "answer_out": 373},
        "timings_ms": {"plan_ms": 520, "answer_ms": 10298},
    }


def test_normal_response_trimmed_correctly(monkeypatch) -> None:
    result, _ = _invoke(monkeypatch, [_FakeResp(200, _full_response())])

    assert is_error(result) is False
    # Answer and evidence pass through.
    assert "[ITC Ltd | p.6]" in result["answer"]
    assert result["evidence"][0]["citation"] == "[ITC Ltd | p.6]"
    # Enriched fields pass through (new in v3).
    assert result["plan"]["category"] == "Annual Report"
    assert result["candidates_considered"] == 3
    # Selected filings are enriched.
    sf = result["selected_filings"][0]
    assert sf["category"] == "Annual Report"
    assert sf["is_scanned"] is False
    assert sf["why_selected"] == "latest AR"
    assert sf["sections_read"] == ["Sustainability"]
    # Data freshness computed from latest filing.
    assert result["data_freshness"] == "2025-06-27T10:00:00"
    # Bulky operational fields are trimmed.
    assert "document_excerpts" not in result
    assert "token_usage" not in result
    assert "timings_ms" not in result


def test_clarification_passthrough(monkeypatch) -> None:
    resp = _full_response()
    resp["needs_clarification"] = True
    resp["clarification_question"] = "Did you mean ITC Ltd or ITC Hotels?"
    resp["answer"] = None
    result, _ = _invoke(monkeypatch, [_FakeResp(200, resp)])

    assert is_error(result) is False
    assert result["needs_clarification"] is True
    assert "ITC Ltd or ITC Hotels" in result["clarification_question"]


def test_scanned_pdf_preserved(monkeypatch) -> None:
    resp = _full_response()
    resp["selected_filings"][0]["is_scanned"] = True
    resp["selected_filings"][0]["read_ok"] = False
    result, _ = _invoke(monkeypatch, [_FakeResp(200, resp)])

    sf = result["selected_filings"][0]
    assert sf["is_scanned"] is True
    assert sf["read_ok"] is False


def test_data_freshness_picks_latest(monkeypatch) -> None:
    resp = _full_response()
    resp["selected_filings"].append({
        "newsid": "n2", "company_name": "ITC Ltd",
        "announcement_dt": "2026-01-15T10:00:00",
        "read_ok": True, "page_count": 10,
    })
    result, _ = _invoke(monkeypatch, [_FakeResp(200, resp)])
    assert result["data_freshness"] == "2026-01-15T10:00:00"


# ── Error handling tests ──────────────────────────────────────────────────


def test_http_422_is_bad_request(monkeypatch) -> None:
    result, _ = _invoke(monkeypatch, [_FakeResp(422, text="missing question")])
    assert is_error(result) is True
    assert result["error_code"] == "stock_chat_bad_request"
    assert result["next_action"] == "ask_user_to_clarify"
    assert result["retriable"] is False


def test_http_500_is_retriable(monkeypatch) -> None:
    result, _ = _invoke(monkeypatch, [_FakeResp(503, text="upstream down")])
    assert is_error(result) is True
    assert result["error_code"] == "stock_chat_http_503"
    assert result["next_action"] == "ask_user_to_retry_later"
    assert result["retriable"] is True


def test_timeout_then_success_tags_retry_count(monkeypatch) -> None:
    behaviors = [
        httpx.TimeoutException("slow"),
        _FakeResp(200, _full_response()),
    ]
    result, calls = _invoke(monkeypatch, behaviors)

    assert is_error(result) is False
    assert result["retry_count"] == 1
    assert len(calls) == 2


def test_timeout_twice_returns_timeout_error(monkeypatch) -> None:
    behaviors = [httpx.TimeoutException("slow"), httpx.TimeoutException("slow")]
    result, calls = _invoke(monkeypatch, behaviors)

    assert is_error(result) is True
    assert result["error_code"] == "stock_chat_timeout"
    assert result["retriable"] is True
    assert len(calls) == 2


def test_connect_error_twice_returns_unreachable(monkeypatch) -> None:
    behaviors = [httpx.ConnectError("no route"), httpx.ConnectError("no route")]
    result, _ = _invoke(monkeypatch, behaviors)

    assert is_error(result) is True
    assert result["error_code"] == "stock_chat_unreachable"
    assert result["retriable"] is True
