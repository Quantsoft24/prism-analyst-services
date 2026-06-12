"""Agent-callable company tools — backed by the master_securities resolver
(``src/services/company_resolver.py``) over the investment DB.

Replaces the old catalog-backed ``lookup_company`` / ``search_companies`` /
``list_covered_sectors``. Resolution always lands on a ``master_securities``
``security_id`` (the key downstream tools like stock-chat now take); ambiguous
references return ``needs_clarification`` + structured ``options`` so the agent
(and the chat UI) can ask the user to pick — no guessing, no hardcoded aliases.

Three tools:
  * ``resolve_company(query)``      — name/ticker/ISIN → one security_id, OR a
                                      clarification with ranked candidate options
  * ``search_companies(query, …)``  — list view (name/sector), distinct companies
  * ``list_sectors()``              — the master_securities sector taxonomy
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.integrations.tools._errors import make_error
from src.services import company_resolver as cr

if TYPE_CHECKING:
    from google.adk.tools import FunctionTool

# search_companies caps the visible items even when a larger limit is passed —
# beyond this the LLM picks badly; tell the user "and N more" instead.
_SEARCH_VISIBLE_CAP = 10


def _company_payload(c: cr.CompanyGroup) -> dict:
    """The fields the agent needs to cite a company + call downstream tools.
    ``security_id`` is the identifier stock-chat takes; ``name`` is the canonical
    name the not-yet-migrated tools (financials/bmc/news) still resolve by."""
    return {
        "security_id": c.security_id,
        "name": c.name,
        "symbol": c.symbol,
        "isin": c.isin,
        "sector": c.sector,
        "exchanges": c.exchanges,
    }


async def resolve_company(query: str) -> dict:
    """Resolve a company the user named — by full/short name, ticker symbol, or
    ISIN — to its canonical record and ``security_id``. **Call this FIRST**
    whenever a question targets a specific company, before any company-scoped
    tool (filings, technicals, financials, BMC): those need the ``security_id``.

    Resolution is fully data-driven over the master securities table (8,239
    NSE/BSE names): exact ticker / ISIN / security_id, name-derived acronyms
    (e.g. "RIL" → Reliance Industries, "SBI" → State Bank of India), and
    typo-tolerant name matching, ranked by Nifty-index prominence.

    WHEN IT RESOLVES: returns ``{found: true, security_id, name, symbol, isin,
    sector, exchanges}``. Use ``security_id`` for downstream tools and quote the
    canonical ``name`` so the user can confirm.

    WHEN IT'S AMBIGUOUS (e.g. "Reliance" → 8 companies, "HDFC" → Bank/Life/AMC):
    returns ``{found: false, needs_clarification: true, clarification: {question,
    mode, options:[{id, label, hint, value}]}}``. **Do NOT pick one yourself** —
    present the options to the user and stop; their selection (a ``security_id``)
    comes back on the next turn.

    WHEN IT'S A TRUE MISS: returns ``{found: false, not_found: true, query}`` — the
    name is NOT a listed company in our coverage (likely a PRIVATE company, a
    SUBSIDIARY, a BRAND/PRODUCT, or an unlisted/misspelled name — e.g. "Blinkit",
    "Jio Platforms"). Do NOT show garbage guesses. Instead: (a) if the user's real
    subject is a different LISTED company (e.g. they asked about "Blinkit" in the
    context of its listed parent **Eternal**), resolve THAT listed company and put
    the unlisted name in your tool ``question`` as the topic; else (b) tell the
    user it isn't a listed company and offer ``search_companies`` / a refined name.

    Args:
        query: Company name, ticker, or ISIN exactly as the user wrote it.

    Returns:
        A resolved company dict, a clarification dict, a ``not_found`` dict, or a
        structured error.
    """
    q = (query or "").strip()
    if not q:
        return make_error(
            message="I need a company name, ticker, or ISIN to look up.",
            code="missing_input",
            next_action="ask_user_to_clarify",
        )
    outcome = await cr.resolve(q)
    if outcome.reason == "not_configured":
        return make_error(
            message="The company database is unavailable right now — try again shortly.",
            code="company_db_unavailable",
            next_action="ask_user_to_retry_later",
            retriable=True,
        )
    if outcome.resolved and outcome.company is not None:
        return {
            "found": True,
            **_company_payload(outcome.company),
            "resolved_by": outcome.reason,
        }
    # True miss (no real candidates) → a distinct ``not_found`` signal, NOT a
    # clarification. The agent decides how to recover (resolve a listed parent,
    # or tell the user + offer search) — we never surface garbage guesses.
    options = cr.to_clarification_options(outcome.candidates)
    if not options:
        return {
            "found": False,
            "not_found": True,
            "query": q,
            "message": (
                f'"{q}" is not a listed company in our coverage (it may be private, '
                "a subsidiary, a brand/product, or unlisted)."
            ),
        }
    # Ambiguous → structured clarification (the runner/UI renders the options as a
    # picker; the agent must not choose for the user).
    return {
        "found": False,
        "needs_clarification": True,
        "query": q,
        "clarification": {
            "question": f'Multiple companies match "{q}". Which one did you mean?',
            "mode": "single_select",
            "options": options,
        },
    }


async def resolve_companies(names: list[str]) -> dict:
    """Resolve SEVERAL companies in ONE call — use this for comparisons / any
    multi-company question ("compare Reliance, Adani and Tata") so that all the
    disambiguation questions are asked TOGETHER in one form, instead of one
    company per turn. Pass each name EXACTLY as the user wrote it.

    Returns ``{resolved: [{query, security_id, name, …}], not_found: [str],
    needs_clarification: bool, clarification: {questions: [{id, question, mode,
    options}]}}``. ``resolved`` holds the unambiguous ones; ``clarification.
    questions`` has one question PER ambiguous name (the runner/UI shows them as a
    single multi-question picker). On the next turn the user's combined reply
    carries every chosen ``security_id`` — proceed with all of them.

    Args:
        names: The company names to resolve (as the user wrote them).
    """
    if not names or not isinstance(names, list):
        return make_error(
            message="Give me the list of company names to resolve.",
            code="missing_input",
            next_action="ask_user_to_clarify",
        )
    resolved: list[dict] = []
    questions: list[dict] = []
    not_found: list[str] = []
    seen: set[str] = set()
    for raw in names:
        name = (raw or "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        outcome = await cr.resolve(name)
        if outcome.reason == "not_configured":
            return make_error(
                message="The company database is unavailable right now — try again shortly.",
                code="company_db_unavailable",
                next_action="ask_user_to_retry_later",
                retriable=True,
            )
        if outcome.resolved and outcome.company is not None:
            resolved.append({"query": name, **_company_payload(outcome.company)})
            continue
        options = cr.to_clarification_options(outcome.candidates)
        if options:
            questions.append({
                "id": name,
                "question": f'Which "{name}" did you mean?',
                "mode": "single_select",
                "options": options,
                "allow_search": True,
            })
        else:
            not_found.append(name)

    out: dict = {"resolved": resolved}
    if not_found:
        out["not_found"] = not_found
    if questions:
        out["needs_clarification"] = True
        out["clarification"] = {"questions": questions}
    return out


async def search_companies(
    query: str | None = None, sector: str | None = None, limit: int = _SEARCH_VISIBLE_CAP,
) -> dict:
    """List companies matching a name/ticker fragment and/or a sector — distinct
    companies (NSE+BSE collapsed). Use when the user is browsing ("show me banks",
    "IT names") rather than asking about ONE company (use ``resolve_company`` for
    that). When ``truncated`` is true, say "N matched, here are the top 10 — narrow
    by sector or a more specific name".

    Args:
        query: Free-text name/ticker fragment (optional — omit to list a sector).
        sector: One of the master_securities sectors (see ``list_sectors``).
        limit: Max visible rows (internally capped at 10).

    Returns:
        ``{total_matched, truncated, items: [{security_id, name, symbol, isin,
        sector, exchanges}]}``.
    """
    if query and len(query) > 200:
        return make_error(
            message="Search query is too long — trim to the company name or ticker.",
            code="input_too_long",
            next_action="ask_user_to_clarify",
        )
    rows, total = await cr.search(query, sector, limit)
    return {
        "total_matched": total,
        "truncated": total > len(rows),
        "items": [_company_payload(r) for r in rows],
    }


async def list_sectors() -> dict:
    """The sector taxonomy available in the securities master (e.g. "Financial
    Services", "Information Technology", "Healthcare"). Use before filtering
    ``search_companies`` by sector, or when the user asks what sectors we cover.

    Returns:
        ``{sectors: [str, ...]}`` (alphabetical).
    """
    return {"sectors": await cr.list_sectors()}


# ── ADK FunctionTool wrappers (lazy — built on first use, same as before) ─────


def _build_tools() -> list["FunctionTool"]:
    from google.adk.tools import FunctionTool

    return [
        FunctionTool(func=resolve_company),
        FunctionTool(func=resolve_companies),
        FunctionTool(func=search_companies),
        FunctionTool(func=list_sectors),
    ]


class _LazyToolList:
    def __init__(self) -> None:
        self._tools: list[FunctionTool] | None = None

    def __iter__(self):
        if self._tools is None:
            self._tools = _build_tools()
        return iter(self._tools)

    def __len__(self) -> int:
        if self._tools is None:
            self._tools = _build_tools()
        return len(self._tools)

    def to_list(self) -> list:
        if self._tools is None:
            self._tools = _build_tools()
        return list(self._tools)


COMPANY_TOOLS = _LazyToolList()
