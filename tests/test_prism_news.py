"""Tests for the prism-news wrapper's response trimming, clamping, and HTTP
error handling.

Pure unit tests — no live service. ``httpx.AsyncClient`` is replaced with a
fake that yields scripted responses (or raises transport errors), so we can
exercise the happy paths, empty-result paths, and the transport/HTTP failure
paths without a network call. Mirrors the test_prism_financials.py pattern.
"""

from __future__ import annotations

import asyncio

import httpx

from src.integrations.tools import prism_news as pn
from src.integrations.tools._errors import is_error


class _FakeResp:
    def __init__(self, status_code: int = 200, json_data=None, text: str = "") -> None:
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text

    def json(self):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json


def _make_client_factory(behaviors: list, calls: list):
    """Fake ``AsyncClient`` whose ``get`` consumes the next behavior — either a
    ``_FakeResp`` to return or an ``Exception`` to raise. The wrapper builds a
    fresh client per retry attempt, so one behavior maps to one attempt."""
    it = iter(behaviors)

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> bool:
            return False

        async def get(self, url, params=None, headers=None):
            calls.append({"url": url, "params": params, "headers": headers})
            behavior = next(it)
            if isinstance(behavior, Exception):
                raise behavior
            return behavior

    return _Client


async def _noop_sleep(*_a, **_k) -> None:
    return None


def _invoke(monkeypatch, behaviors, fn, *, api_key: str = "", **kwargs):
    calls: list = []
    monkeypatch.setattr(pn.httpx, "AsyncClient", _make_client_factory(behaviors, calls))
    monkeypatch.setattr(pn.asyncio, "sleep", _noop_sleep)
    monkeypatch.setattr(pn.settings, "PRISM_NEWS_API_KEY", api_key)
    result = asyncio.run(fn(**kwargs))
    return result, calls


# ── news_sentiment ──────────────────────────────────────────────────────────


def test_sentiment_happy_path_trims_and_passes_verdict(monkeypatch) -> None:
    resp = _FakeResp(200, {
        "company": "HDFC Bank",
        "input": "HDFC",
        "trend": "bullish",
        "avg_score": 0.72,
        "total_articles": 45,
        "sentiment_breakdown": {"positive": 28, "negative": 8, "neutral": 9},
        "trend_detail": {"recent_half": {"positive": 16}, "older_half": {"positive": 12}},
        "top_positive": [{"title": "Q4 beat", "source": "ET", "link": "x", "junk": "drop"}],
        "top_negative": [],
        "provider": "openai",
    })
    result, _ = _invoke(monkeypatch, [resp], pn.news_sentiment, company="HDFC")
    assert is_error(result) is False
    assert result["company"] == "HDFC Bank"
    assert result["trend"] == "bullish"
    assert result["total_articles"] == 45
    # top_positive trimmed to title/source/link only
    assert result["top_positive"][0] == {"title": "Q4 beat", "source": "ET", "link": "x"}
    assert result["provider"] == "openai"
    assert result["data_freshness"] == "live"


def test_sentiment_empty_is_not_error(monkeypatch) -> None:
    resp = _FakeResp(200, {"company": "Foo Ltd", "total_articles": 0,
                           "sentiment_breakdown": {}, "trend": "neutral"})
    result, _ = _invoke(monkeypatch, [resp], pn.news_sentiment, company="Foo")
    assert is_error(result) is False
    assert result["total_articles"] == 0


def test_sentiment_missing_company_clarifies(monkeypatch) -> None:
    result, calls = _invoke(monkeypatch, [], pn.news_sentiment, company="  ")
    assert is_error(result) is True
    assert result["error_code"] == "prism_news_missing_company"
    assert result["next_action"] == "ask_user_to_clarify"
    assert calls == []  # no HTTP call made


def test_sentiment_clamps_hours(monkeypatch) -> None:
    resp = _FakeResp(200, {"company": "X", "total_articles": 1})
    _, calls = _invoke(monkeypatch, [resp], pn.news_sentiment, company="X", hours=9999)
    assert calls[0]["params"]["hours"] == 240  # clamped to max


# ── news_trending ───────────────────────────────────────────────────────────


def test_trending_happy_path(monkeypatch) -> None:
    resp = _FakeResp(200, {"hours": 24, "trending": [
        {"company": "Reliance Industries", "mentions": 19, "sentiment": "positive",
         "sector": "ENERGY", "sentiment_breakdown": {"positive": 12}},
    ]})
    result, _ = _invoke(monkeypatch, [resp], pn.news_trending, hours=24, limit=10)
    assert is_error(result) is False
    assert result["trending"][0]["company"] == "Reliance Industries"
    assert result["data_freshness"] == "live"


def test_trending_empty_is_not_error(monkeypatch) -> None:
    resp = _FakeResp(200, {"hours": 24, "trending": []})
    result, _ = _invoke(monkeypatch, [resp], pn.news_trending)
    assert is_error(result) is False
    assert result["trending"] == []


def test_trending_clamps_limit(monkeypatch) -> None:
    resp = _FakeResp(200, {"trending": []})
    _, calls = _invoke(monkeypatch, [resp], pn.news_trending, limit=999)
    assert calls[0]["params"]["limit"] == 50  # capped


# ── news_search ─────────────────────────────────────────────────────────────


def test_search_trims_articles_and_reads_meta(monkeypatch) -> None:
    resp = _FakeResp(200, {
        "meta": {"total_results": 64, "returned": 2, "sentiment_provider": "openai"},
        "articles": [
            {"title": "A", "source": "ET", "published_ist": "2026-05-30 10:00:00 IST",
             "link": "l1", "companies": ["HDFC Bank"], "sector": "BANKING",
             "description": "long blob", "original_link": "g", "id": "x",
             "sentiment": {"label": "positive", "score": 0.8, "provider": "openai", "junk": 1}},
            {"title": "B", "source": "Mint", "published_ist": "2026-05-30 09:00:00 IST",
             "link": "l2", "companies": [], "sector": None, "sentiment": None},
        ],
    })
    result, _ = _invoke(monkeypatch, [resp], pn.news_search, sector="BANKING")
    assert is_error(result) is False
    assert result["total"] == 64
    art = result["articles"][0]
    # trimmed — no description / original_link / id
    assert set(art.keys()) == {"title", "source", "published_ist", "link", "companies", "sector", "sentiment"}
    assert art["sentiment"] == {"label": "positive", "score": 0.8, "provider": "openai"}
    # data_freshness = first article's timestamp
    assert result["data_freshness"] == "2026-05-30 10:00:00 IST"


def test_search_drops_invalid_sector(monkeypatch) -> None:
    resp = _FakeResp(200, {"meta": {"total_results": 0}, "articles": []})
    _, calls = _invoke(monkeypatch, [resp], pn.news_search, sector="CRYPTO")
    assert "sector" not in calls[0]["params"]  # invalid sector dropped


def test_search_valid_sector_uppercased(monkeypatch) -> None:
    resp = _FakeResp(200, {"meta": {}, "articles": []})
    _, calls = _invoke(monkeypatch, [resp], pn.news_search, sector="banking")
    assert calls[0]["params"]["sector"] == "BANKING"


def test_search_empty_is_not_error(monkeypatch) -> None:
    resp = _FakeResp(200, {"meta": {"total_results": 0}, "articles": []})
    result, _ = _invoke(monkeypatch, [resp], pn.news_search, company="Nonexistent")
    assert is_error(result) is False
    assert result["total"] == 0
    assert result["articles"] == []


# ── news_compare ────────────────────────────────────────────────────────────


def test_compare_string_input_splits(monkeypatch) -> None:
    resp = _FakeResp(200, {"comparison": [{"company": "HDFC Bank", "avg_score": 0.7}]})
    result, calls = _invoke(monkeypatch, [resp], pn.news_compare, companies="HDFC, ICICI, SBI")
    assert is_error(result) is False
    assert calls[0]["params"]["companies"] == "HDFC,ICICI,SBI"
    assert result["companies"] == ["HDFC", "ICICI", "SBI"]


def test_compare_list_input(monkeypatch) -> None:
    resp = _FakeResp(200, {"comparison": []})
    _, calls = _invoke(monkeypatch, [resp], pn.news_compare, companies=["TCS", "Infosys"])
    assert calls[0]["params"]["companies"] == "TCS,Infosys"


def test_compare_empty_clarifies(monkeypatch) -> None:
    result, calls = _invoke(monkeypatch, [], pn.news_compare, companies="  ,  ")
    assert is_error(result) is True
    assert result["error_code"] == "prism_news_missing_company"
    assert calls == []


# ── HTTP error handling ─────────────────────────────────────────────────────


def test_http_404_signals_misconfig(monkeypatch) -> None:
    result, _ = _invoke(monkeypatch, [_FakeResp(404, text="not found")],
                        pn.news_trending)
    assert result["error_code"] == "prism_news_http_404"
    assert result["retriable"] is False
    assert "PRISM_NEWS_URL" in result["detail"]


def test_http_500_retriable(monkeypatch) -> None:
    result, _ = _invoke(monkeypatch, [_FakeResp(503, text="down")], pn.news_trending)
    assert result["error_code"] == "prism_news_http_503"
    assert result["next_action"] == "ask_user_to_retry_later"
    assert result["retriable"] is True


def test_http_422_clarifies(monkeypatch) -> None:
    result, _ = _invoke(monkeypatch, [_FakeResp(422, text="bad sector")],
                        pn.news_search, sector="BANKING")
    assert result["error_code"] == "prism_news_bad_request"
    assert result["next_action"] == "ask_user_to_clarify"


def test_malformed_json_is_handled(monkeypatch) -> None:
    result, _ = _invoke(monkeypatch, [_FakeResp(200, json_data=ValueError("boom"))],
                        pn.news_trending)
    assert result["error_code"] == "prism_news_bad_payload"


def test_non_dict_payload_is_handled(monkeypatch) -> None:
    result, _ = _invoke(monkeypatch, [_FakeResp(200, json_data=["a", "list"])],
                        pn.news_trending)
    assert result["error_code"] == "prism_news_bad_payload"


# ── failure handling: timeout (no retry) vs transport (one retry) ───────────


def test_timeout_returns_immediately_no_retry(monkeypatch) -> None:
    # Timeouts are NOT retried — the cold-scoring path can take ~40s and a
    # retry would double the wait + blow the agent's 60s cap. One attempt,
    # then a friendly "ask again in a moment" error.
    behaviors = [httpx.TimeoutException("slow")]
    result, calls = _invoke(monkeypatch, behaviors, pn.news_trending)
    assert result["error_code"] == "prism_news_timeout"
    assert result["retriable"] is True
    assert result["next_action"] == "ask_user_to_retry_later"
    assert len(calls) == 1  # single attempt — no timeout-retry


def test_transport_error_then_success_tags_retry(monkeypatch) -> None:
    # A true transport blip (connect error) IS retried once.
    behaviors = [httpx.ConnectError("blip"), _FakeResp(200, {"trending": []})]
    result, calls = _invoke(monkeypatch, behaviors, pn.news_trending)
    assert is_error(result) is False
    assert result["retry_count"] == 1
    assert len(calls) == 2


def test_transport_error_twice_unreachable(monkeypatch) -> None:
    behaviors = [httpx.ConnectError("x"), httpx.ConnectError("x")]
    result, calls = _invoke(monkeypatch, behaviors, pn.news_trending)
    assert result["error_code"] == "prism_news_unreachable"
    assert result["retriable"] is True
    assert len(calls) == 2


def test_transport_error_then_timeout_on_retry(monkeypatch) -> None:
    # Transport blip → retry → retry itself times out → timeout error.
    behaviors = [httpx.ConnectError("x"), httpx.TimeoutException("slow")]
    result, calls = _invoke(monkeypatch, behaviors, pn.news_trending)
    assert result["error_code"] == "prism_news_timeout"
    assert len(calls) == 2


# ── auth header ─────────────────────────────────────────────────────────────


def test_api_key_header_when_set(monkeypatch) -> None:
    _, calls = _invoke(monkeypatch, [_FakeResp(200, {"trending": []})],
                       pn.news_trending, api_key="s3cret")
    assert calls[0]["headers"].get("X-API-Key") == "s3cret"


def test_no_auth_header_when_unset(monkeypatch) -> None:
    _, calls = _invoke(monkeypatch, [_FakeResp(200, {"trending": []})],
                       pn.news_trending, api_key="")
    assert calls[0]["headers"] == {}
