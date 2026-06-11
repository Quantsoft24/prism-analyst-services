"""Tests for the agent-facing company tools (thin wrappers over the resolver).
The resolver itself is covered hermetically in ``test_company_resolver.py``;
here we only check the tool's response shapes by stubbing ``cr.resolve``."""

from __future__ import annotations

from src.services import company_resolver as cr
from src.tools import company_tools as ct


def _grp(sid_nse, name, sym="SYM", isin="INE000A01000"):
    return cr.CompanyGroup(
        isin=isin, name=name, symbol=sym, sector="Diversified", security_id_nse=sid_nse,
    )


async def test_resolve_company_resolved(monkeypatch):
    g = _grp(2228, "Reliance Industries Ltd.", "RELIANCE", "INE002A01018")

    async def fake(_q):
        return cr.ResolveOutcome(query=_q, resolved=True, company=g, reason="symbol")

    monkeypatch.setattr(cr, "resolve", fake)
    out = await ct.resolve_company("RELIANCE")
    assert out["found"] is True
    assert out["security_id"] == 2228
    assert out["name"] == "Reliance Industries Ltd."
    assert out["symbol"] == "RELIANCE"
    assert "security_id_nse" not in out  # payload is trimmed for the agent


async def test_resolve_company_clarification(monkeypatch):
    cands = [_grp(2228, "Reliance Industries Ltd."), _grp(2223, "Reliance Capital Ltd.")]

    async def fake(_q):
        return cr.ResolveOutcome(query=_q, resolved=False, company=None, candidates=cands, reason="ambiguous")

    monkeypatch.setattr(cr, "resolve", fake)
    out = await ct.resolve_company("Reliance")
    assert out["found"] is False and out["needs_clarification"] is True
    opts = out["clarification"]["options"]
    assert [o["value"] for o in opts] == [2228, 2223]
    assert out["clarification"]["mode"] == "single_select"


async def test_resolve_company_true_miss_returns_not_found(monkeypatch):
    # No real candidates → a distinct not_found signal (NOT a clarification with
    # empty options), so the agent can recover (resolve a listed parent / search).
    async def fake(_q):
        return cr.ResolveOutcome(query=_q, resolved=False, company=None, candidates=[], reason="not_found")

    monkeypatch.setattr(cr, "resolve", fake)
    out = await ct.resolve_company("blinkit")
    assert out["found"] is False
    assert out["not_found"] is True
    assert "needs_clarification" not in out
    assert "not a listed company" in out["message"].lower()


async def test_resolve_company_empty_input_errors():
    out = await ct.resolve_company("   ")
    assert out["ok"] is False and out["error_code"] == "missing_input"


async def test_resolve_companies_batches_into_one_clarification(monkeypatch):
    # Multi-company: resolved ones returned; each ambiguous name → one question;
    # unlisted → not_found. The runner surfaces all questions in ONE card.
    async def fake(name):
        n = name.lower()
        if n == "tata consultancy services ltd":
            return cr.ResolveOutcome(query=name, resolved=True,
                                     company=_grp(2718, "Tata Consultancy Services Ltd."),
                                     reason="exact_name")
        if n == "blinkit":
            return cr.ResolveOutcome(query=name, resolved=False, company=None,
                                     candidates=[], reason="not_found")
        return cr.ResolveOutcome(query=name, resolved=False, company=None,
                                 candidates=[_grp(1, name + " A"), _grp(2, name + " B")],
                                 reason="ambiguous")

    monkeypatch.setattr(cr, "resolve", fake)
    out = await ct.resolve_companies(["Reliance", "Adani", "Tata Consultancy Services Ltd", "blinkit"])
    assert [r["name"] for r in out["resolved"]] == ["Tata Consultancy Services Ltd."]
    assert out["not_found"] == ["blinkit"]
    assert out["needs_clarification"] is True
    qs = out["clarification"]["questions"]
    assert [q["id"] for q in qs] == ["Reliance", "Adani"]
    assert all(q["mode"] == "single_select" and len(q["options"]) == 2 for q in qs)


async def test_resolve_companies_all_resolved_no_clarification(monkeypatch):
    async def fake(name):
        return cr.ResolveOutcome(query=name, resolved=True, company=_grp(840, name),
                                 reason="exact_name")

    monkeypatch.setattr(cr, "resolve", fake)
    out = await ct.resolve_companies(["Eternal", "Reliance Industries Ltd"])
    assert "needs_clarification" not in out
    assert len(out["resolved"]) == 2
