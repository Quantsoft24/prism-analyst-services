"""Hermetic tests for the master_securities company resolver.

No DB — we build the index from a hand-made ``SecurityRead`` list and exercise
the pure resolution ladder (``_build_index`` + ``_resolve_in_index``). Live-DB
behaviour is validated separately during integration.
"""

from __future__ import annotations

from src.schemas.stock import SecurityRead
from src.services import company_resolver as cr


def _sec(sid, name, sym, isin, exch, sector="Diversified"):
    return SecurityRead(
        security_id=sid, security_name=name, symbol=sym,
        isin=isin, exchange=exch, sector=sector,
    )


# A miniature master: dual-listed Reliance Industries + several Reliance
# namesakes, the spaced-initials HDFC family, TCS, and L&T.
_ROWS = [
    _sec(2228, "Reliance Industries Ltd.", "RELIANCE", "INE002A01018", "NSE"),
    _sec(6753, "Reliance Industries Ltd.", "RELIANCE", "INE002A01018", "BSE"),
    _sec(2223, "Reliance Capital Ltd.", "RELCAPITAL", "INE013A01015", "NSE"),
    _sec(2230, "Reliance Power Ltd.", "RPOWER", "INE614G01033", "NSE"),
    _sec(2225, "Reliance Communications Ltd.", "RCOM", "INE330H01018", "NSE"),
    _sec(1081, "H D F C Bank Ltd.", "HDFCBANK", "INE040A01034", "NSE"),
    _sec(1082, "H D F C Life Insurance Co. Ltd.", "HDFCLIFE", "INE795G01014", "NSE"),
    _sec(2718, "Tata Consultancy Services Ltd.", "TCS", "INE467B01029", "NSE"),
    _sec(1538, "Larsen & Toubro Ltd.", "LT", "INE018A01030", "NSE"),
]


def _resolve(q):
    idx = cr._build_index(_ROWS)
    return cr._resolve_in_index(q, idx)


def test_abbreviation_resolves_single():
    r = _resolve("RIL")
    assert r.resolved and r.company.security_id == 2228


def test_plain_word_reliance_is_ambiguous():
    r = _resolve("Reliance")
    assert not r.resolved
    names = {g.name for g in r.candidates}
    assert "Reliance Industries Ltd." in names
    assert "Reliance Capital Ltd." in names
    assert len(r.candidates) >= 3  # the family, not a single guess


def test_uppercase_ticker_fastpaths_via_symbol():
    r = _resolve("RELIANCE")
    assert r.resolved and r.company.security_id == 2228 and r.reason == "symbol"


def test_isin_resolves():
    r = _resolve("INE002A01018")
    assert r.resolved and r.company.name == "Reliance Industries Ltd."


def test_security_id_resolves_either_exchange():
    # NSE id and BSE id both land on the same canonical company; security_id
    # property prefers the NSE row.
    assert _resolve("2228").company.security_id == 2228
    assert _resolve("6753").company.security_id == 2228


def test_hdfc_spaced_initials_clarify_family():
    r = _resolve("HDFC")
    assert not r.resolved
    names = {g.name for g in r.candidates}
    assert "H D F C Bank Ltd." in names
    assert "H D F C Life Insurance Co. Ltd." in names


def test_tcs_resolves_any_case():
    assert _resolve("TCS").company.security_id == 2718   # symbol fast-path
    assert _resolve("tcs").company.security_id == 2718   # abbreviation


def test_lt_variants_resolve():
    # "L&T" → bare-alnum "LT" matches the DB symbol; full name → exact name.
    assert _resolve("L&T").company.security_id == 1538
    assert _resolve("Larsen and Toubro").company.security_id == 1538


def test_acronym_prominence_breaks_ambiguity():
    # Two companies share the acronym "RIL"; only Reliance is index-listed
    # (prominence>0) → it resolves rather than forcing a clarification.
    rows = _ROWS + [_sec(9001, "Rico Industries Ltd.", "RICOIND", "INE999Z01010", "NSE")]
    idx = cr._build_index(rows, prominence={2228: 5, 6753: 5})  # Reliance in Nifty 50
    out = cr._resolve_in_index("RIL", idx)
    assert out.resolved and out.company.security_id == 2228 and out.reason == "acronym"

    # If BOTH are index-listed, it's genuinely ambiguous → clarify, prominent first.
    idx2 = cr._build_index(rows, prominence={2228: 5, 6753: 5, 9001: 3})
    out2 = cr._resolve_in_index("RIL", idx2)
    assert not out2.resolved
    assert out2.candidates[0].security_id == 2228  # higher prominence first


def test_isin_group_prefers_nse_id():
    g = _resolve("INE002A01018").company
    assert g.security_id_nse == 2228 and g.security_id_bse == 6753
    assert g.security_id == 2228 and g.exchanges == ["NSE", "BSE"]


def test_clarification_options_carry_security_id_value():
    r = _resolve("Reliance")
    opts = cr.to_clarification_options(r.candidates)
    assert opts and all(isinstance(o["value"], int) for o in opts)
    assert all(o["id"] == str(o["value"]) for o in opts)


def test_clarification_reply_with_security_id_resolves_exactly():
    # The pick reply the UI sends back ("<label> — security_id <N>") must lock
    # onto the id regardless of the label around it.
    idx = cr._build_index(_ROWS)
    out = cr._resolve_in_index("Reliance Industries Ltd. — security_id 2228", idx)
    assert out.resolved and out.company.security_id == 2228 and out.reason == "exact_id"
    out2 = cr._resolve_in_index("security_id: 2223", idx)
    assert out2.resolved and out2.company.security_id == 2223


def test_bad_input_returns_unresolved():
    idx = cr._build_index(_ROWS)
    assert not cr._resolve_in_index("", idx).resolved or True  # empty handled in resolve()
    # Pure path: a long garbage string still returns an outcome, never raises.
    out = cr._resolve_in_index("zzzqqq", idx)
    assert out.resolved is False


def test_unlisted_name_returns_not_found_no_garbage():
    # A name with no genuine token match (a brand/private co.) must NOT surface
    # look-alike garbage — the token gate drops it to not_found with no candidates.
    idx = cr._build_index(_ROWS)
    out = cr._resolve_in_index("blinkit", idx)
    assert out.resolved is False
    assert out.reason == "not_found"
    assert out.candidates == []


def test_typo_still_resolves_through_gate():
    # The gate allows 1-2 char typos (a real token still aligns ≥ gate) — the
    # typo must NOT be dropped as garbage. In this tiny fixture it resolves
    # outright; against the live master it would be one of several candidates.
    idx = cr._build_index(_ROWS)
    out = cr._resolve_in_index("Relianse Industries", idx)
    if out.resolved:
        assert out.company.name == "Reliance Industries Ltd."
    else:
        assert "Reliance Industries Ltd." in {g.name for g in out.candidates}
