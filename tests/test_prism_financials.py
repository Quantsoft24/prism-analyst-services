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
    # When the upstream returns ``needs_clarification: true``, the wrapper
    # now auto-disambiguates by re-calling with the top candidate. This
    # test exercises the case where BOTH calls return clarification (e.g.
    # the question has multi-entity ambiguity that one retry can't
    # resolve). The wrapper should preserve the ORIGINAL clarification
    # so the agent has the most informative context to forward to the user.
    first = _FakeResp(200, {
        "rows": [],
        "sql": None,
        "needs_clarification": True,
        "clarification": "Which Reliance did you mean?\n  1. Industries\n  2. Power",
        "error": None,
    })
    second = _FakeResp(200, {
        "rows": [],
        "sql": None,
        "needs_clarification": True,
        "clarification": "And which Wipro?\n  1. Wipro Limited\n  2. Wipro Enterprises",
        "error": None,
    })
    result, calls = _invoke(monkeypatch, [first, second])

    assert is_error(result) is False  # error stays null → not a failure
    assert result["needs_clarification"] is True
    # Original clarification preserved (retry couldn't resolve → keep first).
    assert "Which Reliance" in result["clarification"]
    # Exactly 2 calls — wrapper does not loop further.
    assert len(calls) == 2
    # No auto_disambiguated_to chip because the retry didn't succeed.
    assert "auto_disambiguated_to" not in result


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


def test_http_404_signals_misconfigured_url(monkeypatch) -> None:
    # 404 from this service is almost always a misconfigured PRISM_FINANCIALS_URL
    # (e.g. unset → wrapper hits PRISM's own :8000). We surface it specifically
    # so the agent stops instead of pretending an alternate tool could answer.
    result, _ = _invoke(monkeypatch, [_FakeResp(404, text="Not Found")])
    assert result["error_code"] == "prism_financials_http_404"
    assert result["next_action"] == "ask_user_to_retry_later"
    assert result["retriable"] is False
    assert "PRISM_FINANCIALS_URL" in result["detail"]


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


# ── _extract_top_candidate (clarification parser) ────────────────────────────


class TestExtractTopCandidate:
    """Parser must reliably pull the #1-ranked candidate from the upstream's
    clarification text. Failure modes degrade safely to ``None`` so the
    wrapper skips the auto-retry rather than picking garbage."""

    def test_canonical_format_returns_top_name(self) -> None:
        text = (
            "Your question mentions a company that could mean several things. Which one did you mean?\n\n"
            "  1. Tata Consultancy Services Ltd. (NSE: TCS, prowess_id=11536)\n"
            "  2. TCS Education Society (no NSE listing, prowess_id=98765)\n"
            "  3. Tata Consultancy Cyber Security Ltd. (BSE: 543210)\n"
        )
        assert pf._extract_top_candidate(text) == "Tata Consultancy Services Ltd."

    def test_paren_variant_picks_correct_name(self) -> None:
        text = "  1) Reliance Industries Ltd. (NSE: RELIANCE, prowess_id=500325)\n  2) Reliance Power Ltd. (...)\n"
        assert pf._extract_top_candidate(text) == "Reliance Industries Ltd."

    def test_bare_format_without_parenthetical_still_parsed(self) -> None:
        text = "  1. Wipro Limited\n  2. Wipro Enterprises Ltd\n"
        assert pf._extract_top_candidate(text) == "Wipro Limited"

    def test_empty_string_returns_none(self) -> None:
        assert pf._extract_top_candidate("") is None

    def test_no_numbered_list_returns_none(self) -> None:
        text = "There are multiple possible matches but I'm not listing them."
        assert pf._extract_top_candidate(text) is None

    def test_non_string_input_returns_none(self) -> None:
        # Defensive — clarification field could in theory be null/missing.
        assert pf._extract_top_candidate(None) is None  # type: ignore[arg-type]
        assert pf._extract_top_candidate(123) is None  # type: ignore[arg-type]

    def test_very_short_match_rejected(self) -> None:
        # Defensive: a malformed clarification like "1. ok" shouldn't pick.
        text = "  1. ok (some details)\n"
        assert pf._extract_top_candidate(text) is None


# ── Auto-disambiguation (the integration of wrapper + retry) ─────────────────


class TestAutoDisambiguation:
    """The wrapper's primary defence against the financials_query ambiguity
    gate. When the first call returns ``needs_clarification: true`` and the
    parser can extract a top candidate, the wrapper silently re-calls with
    the candidate prepended. The retry's response replaces the first if it
    resolved successfully; otherwise the original clarification surfaces."""

    def test_clarification_triggers_silent_retry_with_top_candidate(self, monkeypatch) -> None:
        # First call → clarification about TCS. Second call → real rows.
        first = _FakeResp(200, {
            "rows": [],
            "sql": None,
            "needs_clarification": True,
            "clarification": "  1. Tata Consultancy Services Ltd. (NSE: TCS)\n  2. TCS Education\n",
            "error": None,
        })
        second = _FakeResp(200, {
            "rows": [{"company_name": "Tata Consultancy Services Ltd.",
                      "period_end": "2025-03-31", "revenue": 246060.0}],
            "sql": "SELECT ...",
            "needs_clarification": False,
            "error": None,
        })
        result, calls = _invoke(monkeypatch, [first, second], question="TCS revenue FY25")

        # Two upstream calls; the second has the top candidate prepended.
        assert len(calls) == 2
        assert "Tata Consultancy Services Ltd." in calls[1]["json"]["question"]
        # The wrapper returns the SECOND response — rows are present.
        assert result["needs_clarification"] is False
        assert len(result["rows"]) == 1
        # Auto-disambig field tells the agent what we picked.
        assert result["auto_disambiguated_to"] == "Tata Consultancy Services Ltd."

    def test_unparseable_clarification_does_not_retry(self, monkeypatch) -> None:
        # Parser returns None → wrapper skips the retry and surfaces the
        # original clarification cleanly (the agent then forwards it).
        resp = _FakeResp(200, {
            "rows": [],
            "sql": None,
            "needs_clarification": True,
            "clarification": "I have multiple matches but didn't list them.",
            "error": None,
        })
        result, calls = _invoke(monkeypatch, [resp])

        # Only ONE upstream call — no retry attempted.
        assert len(calls) == 1
        assert result["needs_clarification"] is True
        assert "auto_disambiguated_to" not in result

    def test_retry_still_ambiguous_returns_original_clarification(self, monkeypatch) -> None:
        # Multi-entity ambiguity: retry picks TCS but tool now needs to
        # clarify Wipro. Cap at 1 retry → wrapper surfaces the retry's
        # clarification (or the original — either path tells the agent
        # to forward to the user). We assert the wrapper didn't fire a
        # THIRD call.
        first = _FakeResp(200, {
            "rows": [],
            "sql": None,
            "needs_clarification": True,
            "clarification": "  1. Tata Consultancy Services Ltd. (NSE: TCS)\n",
            "error": None,
        })
        second = _FakeResp(200, {
            "rows": [],
            "sql": None,
            "needs_clarification": True,
            "clarification": "  1. Wipro Limited (NSE: WIPRO)\n",
            "error": None,
        })
        result, calls = _invoke(monkeypatch, [first, second])

        # Exactly TWO upstream calls — no third. The wrapper doesn't loop.
        assert len(calls) == 2
        # Result preserves the first (original) clarification so the agent
        # has the most informative context to forward.
        assert result["needs_clarification"] is True
        assert "auto_disambiguated_to" not in result

    def test_retry_returning_rows_replaces_first_response(self, monkeypatch) -> None:
        # Even if the retry returns `needs_clarification: false` AND rows,
        # we use it. Confirmed by checking the rows are from the second call.
        first = _FakeResp(200, {
            "rows": [],
            "needs_clarification": True,
            "clarification": "  1. Infosys Ltd. (NSE: INFY)\n",
            "error": None,
        })
        second = _FakeResp(200, {
            "rows": [{"company_name": "Infosys Ltd.", "period_end": "2025-03-31",
                      "net_profit_margin_pct": 16.48}],
            "needs_clarification": False,
            "error": None,
        })
        result, _ = _invoke(monkeypatch, [first, second])

        assert result["rows"][0]["net_profit_margin_pct"] == 16.48
        assert result["auto_disambiguated_to"] == "Infosys Ltd."

    def test_retry_failure_falls_back_to_original(self, monkeypatch) -> None:
        # If the retry HTTP call errors (5xx), wrapper keeps the original
        # clarification — doesn't propagate the retry error.
        first = _FakeResp(200, {
            "rows": [],
            "needs_clarification": True,
            "clarification": "  1. Tata Consultancy Services Ltd. (NSE: TCS)\n",
            "error": None,
        })
        second = _FakeResp(503, text="upstream down on retry")
        result, calls = _invoke(monkeypatch, [first, second])

        # Two attempts; original clarification surfaces.
        assert len(calls) == 2
        assert result.get("needs_clarification") is True
        assert "auto_disambiguated_to" not in result

    def test_happy_path_no_retry(self, monkeypatch) -> None:
        # Sanity: when the first call returns rows directly, NO retry fires.
        # Steady-state behavior is unchanged for the common case.
        resp = _FakeResp(200, {
            "rows": [{"company_name": "TCS", "period_end": "2025-03-31", "pat": 50000}],
            "needs_clarification": False,
            "error": None,
        })
        result, calls = _invoke(monkeypatch, [resp])

        assert len(calls) == 1  # no retry
        assert "auto_disambiguated_to" not in result
        assert result["rows"][0]["pat"] == 50000
