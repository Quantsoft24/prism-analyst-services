"""Tests for deep-dive "explore further" suggestion synthesis.

Pure unit tests — ``deep_dive.synthesize`` is deterministic and synchronous, so
we just feed it (user_message, tool_trace) and assert which chips it emits, the
seeded context, priority ranking, the cap, dedup, silence-by-default, and
robustness against malformed traces.
"""

from __future__ import annotations

from src.schemas.chat import DeepDiveSuggestion
from src.services import deep_dive

# ── trace builders ──────────────────────────────────────────────────────────


def _resolve(security_id=123, name="Eternal Ltd", symbol="ETERNAL", sector="TECH"):
    return {
        "tool": "resolve_company",
        "args": {"query": name},
        "call_id": "c-resolve",
        "response": {
            "found": True,
            "security_id": security_id,
            "name": name,
            "symbol": symbol,
            "sector": sector,
            "resolved_by": "exact",
        },
    }


def _tool(name, args=None, call_id="c1", response=None):
    return {
        "tool": name,
        "args": args or {},
        "call_id": call_id,
        "response": response if response is not None else {"ok": True},
    }


def _actions(suggestions):
    return [s.action for s in suggestions]


# ── silence by default ────────────────────────────────────────────────────


def test_silent_when_nothing_relevant():
    # A trivial turn (greeting, no tools, no intent) → no chips.
    assert deep_dive.synthesize("hello there", []) == []
    assert deep_dive.synthesize("thanks!", [_resolve()]) == []


def test_silent_on_none_trace():
    assert deep_dive.synthesize("hi", None) == []


# ── BMC ─────────────────────────────────────────────────────────────────────


def test_bmc_fires_when_bmc_tool_ran_with_ticker():
    trace = [_resolve(), _tool("bmc_get", args={"ticker": "ETERNAL"}, call_id="c2")]
    out = deep_dive.synthesize("what is the business model of Eternal", trace)
    bmc = [s for s in out if s.action == "bmc"]
    assert len(bmc) == 1
    assert bmc[0].context == {"ticker": "ETERNAL"}
    assert "Eternal" in bmc[0].label
    assert isinstance(bmc[0], DeepDiveSuggestion)


def test_bmc_fires_on_bizmodel_intent_with_resolved_company_using_symbol():
    # No bmc tool ran, but the user asked about the business model and a company
    # resolved → suggest BMC seeded with the resolved symbol (lower priority).
    trace = [_resolve(symbol="ETERNAL")]
    out = deep_dive.synthesize("how does Eternal's revenue model work?", trace)
    bmc = [s for s in out if s.action == "bmc"]
    assert len(bmc) == 1
    assert bmc[0].context == {"ticker": "ETERNAL"}


def test_bmc_silent_on_bizmodel_intent_without_company():
    # Intent but nothing resolved → no company-specific chip.
    assert deep_dive.synthesize("explain business models in general", []) == []


# ── Stock dashboard ─────────────────────────────────────────────────────────


def test_stock_dashboard_fires_on_financials_with_security_id():
    trace = [_resolve(security_id=456), _tool("financials_query", call_id="c2")]
    out = deep_dive.synthesize("what was Eternal's revenue last year?", trace)
    stock = [s for s in out if s.action == "stock_dashboard"]
    assert len(stock) == 1
    assert stock[0].context == {"security_id": 456}


def test_stock_dashboard_silent_without_resolved_company():
    # A price-intent turn with no resolved company has nowhere to land.
    out = deep_dive.synthesize("what is the share price today?", [])
    assert [s for s in out if s.action == "stock_dashboard"] == []


# ── News ────────────────────────────────────────────────────────────────────


def test_news_fires_when_news_tool_ran_with_company_name():
    trace = [_resolve(name="Eternal Ltd"), _tool("news_sentiment", call_id="c2")]
    out = deep_dive.synthesize("how is Eternal doing in the news?", trace)
    news = [s for s in out if s.action == "news"]
    assert len(news) == 1
    assert news[0].context == {"company": "Eternal Ltd"}


# ── Regulatory ──────────────────────────────────────────────────────────────


def test_regulatory_fires_when_sebi_tool_ran():
    trace = [_tool("sebi_search", call_id="c1")]
    out = deep_dive.synthesize("any recent SEBI circulars on insider trading?", trace)
    assert "regulatory" in _actions(out)
    reg = [s for s in out if s.action == "regulatory"][0]
    assert reg.context == {}


# ── Portfolio (capability-gap conversion) ───────────────────────────────────


def test_portfolio_fires_on_build_intent_without_any_tool():
    # The agent has no portfolio tool; the ask is a dead-end we convert.
    out = deep_dive.synthesize("build a portfolio of large-cap IT stocks", [])
    assert "portfolio" in _actions(out)


def test_portfolio_fires_on_screening_intent():
    out = deep_dive.synthesize("find stocks with P/E below 15 and high ROE", [])
    assert "portfolio" in _actions(out)


# ── cap, dedup, priority ranking ─────────────────────────────────────────────


def test_cap_and_priority_ranking():
    # Trigger five actions at once; default cap is 3, ranked by priority:
    # portfolio(100) > bmc-ran(90) > stock-ran(80) > news-ran(70) > regulatory(60).
    trace = [
        _resolve(security_id=789),
        _tool("bmc_get", args={"ticker": "ETERNAL"}, call_id="c2"),
        _tool("financials_query", call_id="c3"),
        _tool("news_search", call_id="c4"),
        _tool("sebi_search", call_id="c5"),
    ]
    msg = "build a portfolio around Eternal and its business model news and filings"
    out = deep_dive.synthesize(msg, trace)
    assert len(out) == 3
    assert _actions(out) == ["portfolio", "bmc", "stock_dashboard"]


def test_cap_is_configurable(monkeypatch):
    monkeypatch.setattr(deep_dive.settings, "DEEP_DIVE_MAX_SUGGESTIONS", 1)
    trace = [
        _resolve(),
        _tool("bmc_get", args={"ticker": "ETERNAL"}, call_id="c2"),
        _tool("financials_query", call_id="c3"),
    ]
    out = deep_dive.synthesize("business model and financials of Eternal", trace)
    assert len(out) == 1
    assert out[0].action == "bmc"  # highest priority of the matched set


def test_no_duplicate_actions():
    # Two bmc tools in one turn must still yield a single BMC chip.
    trace = [
        _resolve(),
        _tool("bmc_get", args={"ticker": "ETERNAL"}, call_id="c2"),
        _tool("bmc_generate", args={"ticker": "ETERNAL"}, call_id="c3"),
    ]
    out = deep_dive.synthesize("business model of Eternal", trace)
    assert _actions(out).count("bmc") == 1


# ── entity extraction ────────────────────────────────────────────────────────


def test_resolve_companies_multiple_dedup_by_security_id():
    trace = [
        {
            "tool": "resolve_companies",
            "args": {"names": ["HDFC", "ICICI", "HDFC"]},
            "call_id": "c1",
            "response": {
                "resolved": [
                    {"security_id": 1, "name": "HDFC Bank", "symbol": "HDFCBANK", "sector": "BANKING"},
                    {"security_id": 2, "name": "ICICI Bank", "symbol": "ICICIBANK", "sector": "BANKING"},
                    {"security_id": 1, "name": "HDFC Bank", "symbol": "HDFCBANK", "sector": "BANKING"},
                ]
            },
        },
        _tool("financials_query", call_id="c2"),
    ]
    out = deep_dive.synthesize("compare HDFC and ICICI margins", trace)
    stock = [s for s in out if s.action == "stock_dashboard"]
    # Primary (first resolved) drives the seeded context.
    assert stock and stock[0].context == {"security_id": 1}


def test_not_found_company_yields_no_company_context():
    trace = [
        {
            "tool": "resolve_company",
            "args": {"query": "SomePrivateCo"},
            "call_id": "c1",
            "response": {"found": False, "not_found": True, "query": "SomePrivateCo"},
        }
    ]
    # bizmodel intent but no resolved company → silent.
    assert deep_dive.synthesize("what's the business model of SomePrivateCo", trace) == []


# ── robustness ───────────────────────────────────────────────────────────────


def test_malformed_trace_does_not_raise():
    trace = [
        "not a dict",
        {"tool": None},
        {"tool": "bmc_get"},  # no args, no response
        {"tool": "resolve_company", "response": "garbage"},
        {"tool": "resolve_company", "response": {"found": True}},  # no ids at all
    ]
    # Should not raise; bmc_get ran (no ticker) → a bare-context BMC chip is fine.
    out = deep_dive.synthesize("business model", trace)
    assert isinstance(out, list)
    for s in out:
        assert isinstance(s, DeepDiveSuggestion)


def test_security_id_string_digit_coerced():
    trace = [
        {
            "tool": "resolve_company",
            "args": {},
            "call_id": "c1",
            "response": {"found": True, "security_id": "321", "name": "X Ltd", "symbol": "X"},
        },
        _tool("stock_technicals", call_id="c2"),
    ]
    out = deep_dive.synthesize("show me the chart", trace)
    stock = [s for s in out if s.action == "stock_dashboard"]
    assert stock and stock[0].context == {"security_id": 321}


def test_security_id_bool_rejected():
    # bool is an int subclass — must not be accepted as a security_id.
    trace = [
        {
            "tool": "resolve_company",
            "args": {},
            "call_id": "c1",
            "response": {"found": True, "security_id": True, "name": "X Ltd", "symbol": "X"},
        },
        _tool("financials_query", call_id="c2"),
    ]
    out = deep_dive.synthesize("revenue of X", trace)
    # No usable security_id → no stock-dashboard chip.
    assert [s for s in out if s.action == "stock_dashboard"] == []
