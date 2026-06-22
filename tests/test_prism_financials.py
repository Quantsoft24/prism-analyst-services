"""Tests for the prism-financials wrapper (security_id migration, 2026-06).

Pure unit tests — no live service. ``httpx.AsyncClient`` is replaced with a
fake that yields scripted responses (or raises transport errors), so we exercise
the new ``status``-based reply shapes (ok / no_data / needs_clarification), the
id-routing in the request payload, and the HTTP/transport failure paths without
a network call. End-to-end ADK wiring is covered elsewhere.
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
    """Fake ``AsyncClient`` — each ``post`` consumes the next behavior (a
    ``_FakeResp`` to return or an ``Exception`` to raise); one behavior per
    attempt (the wrapper builds a fresh client per retry)."""
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
    """Patch httpx + retry sleep + API-key setting, run the tool, return
    ``(result, calls)``."""
    calls: list = []
    monkeypatch.setattr(pf.httpx, "AsyncClient", _make_client_factory(behaviors, calls))
    monkeypatch.setattr(pf.asyncio, "sleep", _noop_sleep)
    monkeypatch.setattr(pf.settings, "PRISM_FINANCIALS_API_KEY", api_key)
    kwargs.setdefault("question", "q")
    result = asyncio.run(pf.financials_query(**kwargs))
    return result, calls


# ── Request payload: id routing ─────────────────────────────────────────────


def test_question_sent_verbatim_no_extra_fields(monkeypatch) -> None:
    q = "What was Reliance Industries' net profit in FY24?"
    _, calls = _invoke(monkeypatch, [_FakeResp(200, {"status": "ok", "operation": "lookup"})], question=q)
    sent = calls[0]["json"]
    assert sent["question"] == q  # untouched — no normalisation
    # New contract: no answer_mode / debug / user_id, and no ids unless supplied.
    assert "answer_mode" not in sent and "debug" not in sent and "user_id" not in sent
    assert "security_id" not in sent and "security_ids" not in sent


def test_security_id_in_payload(monkeypatch) -> None:
    _, calls = _invoke(
        monkeypatch, [_FakeResp(200, {"status": "ok", "operation": "lookup"})],
        question="net profit FY2024", security_id=1081,
    )
    assert calls[0]["json"]["security_id"] == 1081
    assert "security_ids" not in calls[0]["json"]


def test_security_ids_take_precedence(monkeypatch) -> None:
    _, calls = _invoke(
        monkeypatch, [_FakeResp(200, {"status": "ok", "operation": "lookup"})],
        question="compare margins", security_id=9, security_ids=[2718, 1267],
    )
    sent = calls[0]["json"]
    assert sent["security_ids"] == [2718, 1267]
    assert "security_id" not in sent  # security_ids wins


def test_screen_omits_both_ids(monkeypatch) -> None:
    # Market-wide screen/rank → neither id; the service infers the universe.
    _, calls = _invoke(monkeypatch, [_FakeResp(200, {"status": "ok", "operation": "rank", "ranking": []})],
                       question="top 10 NBFCs by ROE")
    assert "security_id" not in calls[0]["json"] and "security_ids" not in calls[0]["json"]


# ── status: ok ──────────────────────────────────────────────────────────────


def test_lookup_keeps_structured_fields_drops_provenance(monkeypatch) -> None:
    resp = _FakeResp(200, {
        "status": "ok", "operation": "lookup", "question": "q",
        "value": 65446.5, "period": "FY2024",
        "field": {"key": "net_profit", "label": "Net Profit (PAT)", "unit": "₹ cr"},
        "company": {"security_id": 1081, "name": "H D F C Bank Ltd.", "symbol": "HDFCBANK"},
        "answer": "…", "provenance": {"sql": ["SELECT ..."]}, "granularity": "annual",
    })
    result, _ = _invoke(monkeypatch, [resp], security_id=1081)
    assert is_error(result) is False
    assert result["status"] == "ok"
    assert result["operation"] == "lookup"
    assert result["value"] == 65446.5
    assert result["field"]["label"] == "Net Profit (PAT)"
    assert result["data_freshness"] == "FY2024"  # from `period`
    # Trimmed: provenance (no SQL viewer), question echo, granularity dropped.
    assert "provenance" not in result and "question" not in result and "granularity" not in result


def test_trend_freshness_is_latest_series_period(monkeypatch) -> None:
    resp = _FakeResp(200, {
        "status": "ok", "operation": "trend",
        "series": [
            {"period": "FY2022", "value": 2},
            {"period": "FY2024", "value": 4},
            {"period": "FY2023", "value": 3},
        ],
    })
    result, _ = _invoke(monkeypatch, [resp], security_id=1081)
    assert result["operation"] == "trend"
    assert len(result["series"]) == 3
    assert result["data_freshness"] == "FY2024"


def test_compare_passes_comparison_through(monkeypatch) -> None:
    resp = _FakeResp(200, {
        "status": "ok", "operation": "lookup",  # service tags compares as lookup
        "comparison": [
            {"security_id": 2718, "name": "TCS", "value": 19.1, "period": "FY2025"},
            {"security_id": 1267, "name": "Infosys", "value": 16.4, "period": "FY2025"},
        ],
    })
    result, _ = _invoke(monkeypatch, [resp], security_ids=[2718, 1267])
    assert len(result["comparison"]) == 2
    assert result["comparison"][0]["name"] == "TCS"


# ── status: no_data (success envelope, not an error) ────────────────────────


def test_no_data_is_not_an_error(monkeypatch) -> None:
    resp = _FakeResp(200, {"status": "no_data", "operation": "lookup",
                           "answer": "No value for that period."})
    result, _ = _invoke(monkeypatch, [resp], security_id=1081)
    assert is_error(result) is False
    assert result["status"] == "no_data"


# ── status: needs_clarification (metric-level, carries suggestions) ─────────


def test_needs_clarification_returns_suggestions(monkeypatch) -> None:
    resp = _FakeResp(200, {
        "status": "needs_clarification",
        "answer": "Free cash flow isn't in the catalog. Did you mean …?",
        "suggestions": [{"key": "cash", "label": "Cash & Bank Balance"}],
    })
    result, _ = _invoke(monkeypatch, [resp], security_id=1081)
    assert is_error(result) is False
    assert result["needs_clarification"] is True
    assert result["suggestions"][0]["key"] == "cash"


# ── HTTP status handling ────────────────────────────────────────────────────


def test_http_404_means_security_not_found(monkeypatch) -> None:
    # 404 now means the supplied security_id doesn't exist (resolver mismatch).
    result, _ = _invoke(monkeypatch, [_FakeResp(404, text='{"detail":"security_id 999 not found"}')],
                        security_id=999)
    assert is_error(result) is True
    assert result["error_code"] == "prism_financials_security_not_found"
    assert result["next_action"] == "ask_user_to_clarify"
    assert result["retriable"] is False
    assert "999" in result["detail"]


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
        _FakeResp(200, {"status": "ok", "operation": "lookup", "value": 1, "period": "FY2024"}),
    ]
    result, calls = _invoke(monkeypatch, behaviors, security_id=1)
    assert is_error(result) is False
    assert result["retry_count"] == 1
    assert len(calls) == 2  # first attempt + retry


def test_timeout_twice_returns_timeout_error(monkeypatch) -> None:
    behaviors = [httpx.TimeoutException("slow"), httpx.TimeoutException("slow")]
    result, calls = _invoke(monkeypatch, behaviors)
    assert is_error(result) is True
    assert result["error_code"] == "prism_financials_timeout"
    assert result["retriable"] is True
    assert len(calls) == 2


def test_connect_error_twice_returns_unreachable(monkeypatch) -> None:
    behaviors = [httpx.ConnectError("no route"), httpx.ConnectError("no route")]
    result, _ = _invoke(monkeypatch, behaviors)
    assert result["error_code"] == "prism_financials_unreachable"
    assert result["retriable"] is True


# ── Auth header (sent only when the env var is set) ─────────────────────────


def test_api_key_header_attached_when_set(monkeypatch) -> None:
    _, calls = _invoke(monkeypatch, [_FakeResp(200, {"status": "ok", "operation": "lookup"})], api_key="s3cret")
    assert calls[0]["headers"].get("X-API-Key") == "s3cret"


def test_no_auth_header_when_key_unset(monkeypatch) -> None:
    _, calls = _invoke(monkeypatch, [_FakeResp(200, {"status": "ok", "operation": "lookup"})], api_key="")
    assert calls[0]["headers"] == {}
