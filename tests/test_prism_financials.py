"""Tests for the prism-financials wrapper's response-shape routing and HTTP
error handling.

Pure unit tests — no live service. ``httpx.AsyncClient`` is replaced with a
fake that yields scripted responses (or raises transport errors), so we can
exercise all four ``/ask`` reply shapes plus the transport/HTTP failure paths
without a network call. The end-to-end ADK wiring is covered elsewhere.
"""

from __future__ import annotations

import asyncio

import httpx

from src.integrations.tools import prism_financials as pf
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
            calls.append({"url": url, "json": json, "headers": headers})
            behavior = next(it)
            if isinstance(behavior, Exception):
                raise behavior
            return behavior

    return _Client


async def _noop_sleep(*_a, **_k) -> None:
    return None


def _invoke(monkeypatch, behaviors: list, *, api_key: str = "", **kwargs):
    """Patch httpx + the retry sleep + the API-key setting, then run the tool.
    Returns ``(result, calls)``."""
    calls: list = []
    monkeypatch.setattr(pf.httpx, "AsyncClient", _make_client_factory(behaviors, calls))
    monkeypatch.setattr(pf.asyncio, "sleep", _noop_sleep)
    monkeypatch.setattr(pf.settings, "PRISM_FINANCIALS_API_KEY", api_key)
    kwargs.setdefault("question", "q")
    result = asyncio.run(pf.financials_query(**kwargs))
    return result, calls


# ── Shape 1: normal answer ─────────────────────────────────────────────────


def test_normal_passthrough_and_freshness(monkeypatch) -> None:
    resp = _FakeResp(200, {
        "question": "q",
        "rows": [{"company_name": "Reliance", "period_end": "2024-03-31", "debt_to_equity": 0.44}],
        "sql": "SELECT ...",
        "answer": "",
        "needs_clarification": False,
        "clarification": None,
        "provider": "openai",
        "duration_ms": 700,
        "debug": None,
    })
    result, _ = _invoke(monkeypatch, [resp])

    assert is_error(result) is False
    assert result["rows"][0]["debt_to_equity"] == 0.44
    assert result["sql"] == "SELECT ..."
    assert result["data_freshness"] == "2024-03-31"
    assert result["provider"] == "openai"
    # Operational fields are trimmed.
    assert "answer" not in result
    assert "debug" not in result


def test_freshness_picks_latest_period_across_rows(monkeypatch) -> None:
    resp = _FakeResp(200, {
        "rows": [
            {"period_end": "2022-03-31", "value": 1},
            {"period_end": "2024-03-31", "value": 3},
            {"period_end": "2023-03-31", "value": 2},
        ],
        "error": None,
    })
    result, _ = _invoke(monkeypatch, [resp])
    assert result["data_freshness"] == "2024-03-31"


def test_question_sent_verbatim_with_defaults(monkeypatch) -> None:
    q = "What is Reliance Industries' Debt-to-Equity ratio in FY24?"
    _, calls = _invoke(monkeypatch, [_FakeResp(200, {"rows": [], "error": None})], question=q)
    sent = calls[0]["json"]
    assert sent["question"] == q  # untouched — no normalisation
    assert sent["answer_mode"] == "off"
    assert sent["debug"] is False
    assert "user_id" not in sent


def test_user_id_passed_through_when_given(monkeypatch) -> None:
    _, calls = _invoke(
        monkeypatch, [_FakeResp(200, {"rows": [], "error": None})],
        question="q", user_id="session-123",
    )
    assert calls[0]["json"]["user_id"] == "session-123"


def test_answer_mode_override_is_forwarded(monkeypatch) -> None:
    _, calls = _invoke(
        monkeypatch, [_FakeResp(200, {"rows": [], "error": None})],
        question="q", answer_mode="table",
    )
    assert calls[0]["json"]["answer_mode"] == "table"


# ── Shape 2: clarification (still a SUCCESS envelope) ───────────────────────


def test_clarification_passthrough_is_not_error(monkeypatch) -> None:
    resp = _FakeResp(200, {
        "rows": [],
        "sql": None,
        "needs_clarification": True,
        "clarification": "Which Reliance did you mean?\n  1. Industries\n  2. Power",
        "error": None,
    })
    result, _ = _invoke(monkeypatch, [resp])

    assert is_error(result) is False  # error stays null → not a failure
    assert result["needs_clarification"] is True
    assert "Which Reliance" in result["clarification"]


# ── Shape 3: NOT IN DATABASE refusal (deliberate, not an error) ─────────────


def test_not_in_database_passes_through_not_as_error(monkeypatch) -> None:
    resp = _FakeResp(200, {
        "rows": [{"note": "NOT IN DATABASE: stock-price time series isn't loaded."}],
        "error": None,
        "needs_clarification": False,
    })
    result, _ = _invoke(monkeypatch, [resp])

    assert is_error(result) is False
    assert result["rows"][0]["note"].startswith("NOT IN DATABASE")
    assert result["data_freshness"] is None  # no period_end on a refusal row


# ── Shape 4: service-side error → standard make_error shape ─────────────────


def test_error_shape_becomes_structured_error(monkeypatch) -> None:
    resp = _FakeResp(200, {"rows": [], "error": "RateLimitError: 429 from OpenAI"})
    result, _ = _invoke(monkeypatch, [resp])

    assert is_error(result) is True
    assert result["error_code"] == "prism_financials_upstream_error"
    assert result["next_action"] == "ask_user_to_retry_later"
    assert result["retriable"] is True
    assert "RateLimitError" in result["detail"]


# ── HTTP status handling ────────────────────────────────────────────────────


def test_http_422_is_bad_request(monkeypatch) -> None:
    result, _ = _invoke(monkeypatch, [_FakeResp(422, text="missing question")])
    assert result["error_code"] == "prism_financials_bad_request"
    assert result["next_action"] == "ask_user_to_clarify"
    assert result["retriable"] is False


def test_http_500_is_retriable(monkeypatch) -> None:
    result, _ = _invoke(monkeypatch, [_FakeResp(503, text="upstream down")])
    assert result["error_code"] == "prism_financials_http_503"
    assert result["next_action"] == "ask_user_to_retry_later"
    assert result["retriable"] is True


# ── Transport failures + one-shot retry ─────────────────────────────────────


def test_timeout_then_success_tags_retry_count(monkeypatch) -> None:
    behaviors = [
        httpx.TimeoutException("slow"),
        _FakeResp(200, {"rows": [{"period_end": "2024-03-31", "v": 1}], "error": None}),
    ]
    result, calls = _invoke(monkeypatch, behaviors)

    assert is_error(result) is False
    assert result["retry_count"] == 1
    assert len(calls) == 2  # first attempt + retry


def test_timeout_twice_returns_timeout_error(monkeypatch) -> None:
    behaviors = [httpx.TimeoutException("slow"), httpx.TimeoutException("slow")]
    result, calls = _invoke(monkeypatch, behaviors)

    assert is_error(result) is True
    assert result["error_code"] == "prism_financials_timeout"
    assert result["next_action"] == "ask_user_to_retry_later"
    assert result["retriable"] is True
    assert len(calls) == 2


def test_connect_error_twice_returns_unreachable(monkeypatch) -> None:
    behaviors = [httpx.ConnectError("no route"), httpx.ConnectError("no route")]
    result, _ = _invoke(monkeypatch, behaviors)

    assert result["error_code"] == "prism_financials_unreachable"
    assert result["retriable"] is True


# ── Auth header (sent only when the env var is set) ─────────────────────────


def test_api_key_header_attached_when_set(monkeypatch) -> None:
    _, calls = _invoke(
        monkeypatch, [_FakeResp(200, {"rows": [], "error": None})], api_key="s3cret",
    )
    assert calls[0]["headers"].get("X-API-Key") == "s3cret"


def test_no_auth_header_when_key_unset(monkeypatch) -> None:
    _, calls = _invoke(
        monkeypatch, [_FakeResp(200, {"rows": [], "error": None})], api_key="",
    )
    assert calls[0]["headers"] == {}
