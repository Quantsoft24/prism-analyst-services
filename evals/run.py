"""PRISM agent eval harness — the root fix for scenario-by-scenario whack-a-mole.

Drives the REAL `/chat/run` pipeline (so it exercises the custom AgentRunner:
clarification events, two-tier composer, citation merge — the logic ADK's own
AgentEvaluator would bypass) with a graded suite of analyst queries spanning the
intent taxonomy AND every bug we've hit. Each case asserts *behaviors* (right
entity resolved, right tools called, clarification shape, citations present,
format honored, no garbage) — not exact strings — so it catches regression
*classes*, not one screenshot.

Run against the live backend (opt-in; consumes message + Gemini quota):

    python evals/run.py                      # all cases
    python evals/run.py --only blinkit-not-found,compare-table
    python evals/run.py --base http://localhost:8000 --firm EVAL_RUN

Add a case = append to CASES. New analyst intents become new cases here, not
prompt surgery. Exit code is non-zero if any case fails (CI-friendly).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

# ── A captured turn: the events the agent emitted for one message ──────────────


@dataclass
class TurnResult:
    tools: list[str] = field(default_factory=list)
    tool_args: list[dict] = field(default_factory=list)
    clarification: dict | None = None
    final: dict | None = None
    error: dict | None = None

    @property
    def answer(self) -> str:
        return (self.final or {}).get("answer") or ""

    @property
    def structured(self) -> dict:
        return (self.final or {}).get("structured") or {}

    @property
    def filing_citations(self) -> list[dict]:
        return [c for c in (self.structured.get("citations") or [])
                if c.get("source_kind") == "filing"]


# ── Assertions (behavioral predicates over a TurnResult) ───────────────────────
# Each returns "" on pass or a failure reason string.

def tools_any(*names: str) -> Callable[[TurnResult], str]:
    def chk(t: TurnResult) -> str:
        return "" if any(n in t.tools for n in names) else f"expected any tool of {names}, got {t.tools}"
    return chk


def no_clarification() -> Callable[[TurnResult], str]:
    def chk(t: TurnResult) -> str:
        if t.clarification is None:
            return ""
        ids = [q.get("id") for q in t.clarification.get("questions", [])]
        return f"unexpected clarification: {ids}"
    return chk


def clarification_questions(n: int) -> Callable[[TurnResult], str]:
    def chk(t: TurnResult) -> str:
        if t.clarification is None:
            return "expected a clarification, got none"
        got = len(t.clarification.get("questions") or [])
        return "" if got == n else f"expected {n} clarification questions, got {got}"
    return chk


def has_final() -> Callable[[TurnResult], str]:
    return lambda t: "" if t.final is not None and t.error is None else f"no final answer (error={t.error})"


def answer_matches(pattern: str) -> Callable[[TurnResult], str]:
    rx = re.compile(pattern, re.IGNORECASE)
    return lambda t: "" if rx.search(t.answer) else f"answer didn't match /{pattern}/i"


def answer_excludes(pattern: str) -> Callable[[TurnResult], str]:
    rx = re.compile(pattern, re.IGNORECASE)
    return lambda t: f"answer should NOT contain /{pattern}/i" if rx.search(t.answer) else ""


def has_table() -> Callable[[TurnResult], str]:
    return lambda t: "" if ("|" in t.answer and "---" in t.answer) else "expected a markdown table"


def filing_citations_present() -> Callable[[TurnResult], str]:
    def chk(t: TurnResult) -> str:
        fil = t.filing_citations
        if not fil:
            return "expected >=1 filing citation"
        bad = [c for c in fil if not (c.get("url") and c.get("page"))]
        return f"{len(bad)} filing citations missing url/page" if bad else ""
    return chk


def tool_arg_contains(tool: str, key: str, pattern: str) -> Callable[[TurnResult], str]:
    rx = re.compile(pattern, re.IGNORECASE)
    def chk(t: TurnResult) -> str:
        for name, args in zip(t.tools, t.tool_args):
            if name == tool and rx.search(str(args.get(key, ""))):
                return ""
        return f"{tool} arg '{key}' never matched /{pattern}/i"
    return chk


# ── Cases ──────────────────────────────────────────────────────────────────────
# A case = id + a list of turns. A turn is {message, expect:[assertions]} OR
# {answer_clarification: True} which auto-picks option[0] of every clarification
# question from the previous turn and sends the combined reply.

@dataclass
class Case:
    id: str
    turns: list[dict]
    intent: str = ""


CASES: list[Case] = [
    Case("financials-single", intent="fundamentals", turns=[
        {"message": "What was Infosys revenue and net profit in FY25?",
         "expect": [no_clarification(), tools_any("financials_query"), has_final(),
                    answer_matches(r"revenue"), answer_matches(r"profit"),
                    answer_excludes(r"\|\s*p\.\s*\d")]},  # no fabricated page-cite on DB data
    ]),
    Case("compare-table", intent="comparison", turns=[
        {"message": "Compare FY24 revenue and net profit of Reliance Industries Ltd, "
                    "Tata Consultancy Services Ltd and Adani Enterprises Ltd",
         "expect": [no_clarification(), tools_any("financials_query"), has_final(), has_table()]},
    ]),
    Case("compare-multi-clarify", intent="comparison+clarify", turns=[
        {"message": "compare KPIs of reliance, adani and tata",
         "expect": [clarification_questions(3), tools_any("resolve_companies")]},
        {"answer_clarification": True,
         "expect": [has_final(), tools_any("financials_query"), has_table()]},
    ]),
    Case("compare-multi-filings", intent="comparison+filings+clarify", turns=[
        {"message": "summary of board meeting of relance, adani and eternal of year 2025",
         "expect": [clarification_questions(2)]},  # eternal auto-resolves; relance+adani ambiguous
        {"answer_clarification": True,
         "expect": [has_final(), tools_any("stock_filings_read"),
                    answer_excludes(r"couldn't put together|Try a more specific|didn't have enough")]},
    ]),
    Case("clarify-single-qfidelity", intent="clarify+filings", turns=[
        {"message": "summary of Reliance board meetings 2025",
         "expect": [clarification_questions(1)]},
        {"answer_clarification": True,
         "expect": [has_final(), tools_any("stock_filings_read"),
                    tool_arg_contains("stock_filings_read", "question", r"2025"),
                    filing_citations_present()]},
    ]),
    Case("filings-citations", intent="filings", turns=[
        {"message": "What did ITC say about sustainability in its latest annual report?",
         "expect": [no_clarification(), tools_any("stock_filings_read"), has_final(),
                    filing_citations_present()]},
    ]),
    Case("entity-selection", intent="entity", turns=[
        {"message": "what are the implications of buying blinkit for Eternal?",
         "expect": [no_clarification(), tool_arg_contains("resolve_company", "query", r"eternal"),
                    has_final()]},
    ]),
    Case("blinkit-not-found", intent="not_found", turns=[
        {"message": "tell me about blinkit",
         "expect": [has_final(), answer_excludes(r"\bITC Ltd\b|\bITI Ltd\b")]},  # no garbage
    ]),
    Case("technicals", intent="market", turns=[
        {"message": "What is TCS trading at right now?",
         "expect": [no_clarification(), tools_any("stock_technicals"), has_final()]},
    ]),
    Case("reformat-table", intent="format", turns=[
        {"message": "What were Reliance Industries Ltd FY24 revenue, net profit and total expenses?",
         "expect": [has_final()]},
        {"message": "show that in a table", "expect": [has_final(), has_table()]},
    ]),
    Case("off-topic-refusal", intent="guardrail", turns=[
        {"message": "what's the weather in Mumbai tomorrow?",
         "expect": [has_final(), answer_excludes(r"\bsunny\b|\brain\b|\btemperature\b")]},
    ]),
]


# ── Runner ─────────────────────────────────────────────────────────────────────

async def _run_turn(client: httpx.AsyncClient, base: str, headers: dict,
                    message: str, session_id: str | None) -> tuple[TurnResult, str | None]:
    body: dict[str, Any] = {"message": message}
    if session_id:
        body["session_id"] = session_id
    res = TurnResult()
    sid = session_id
    async with client.stream("POST", f"{base}/api/v1/chat/run", headers=headers,
                             json=body, timeout=220) as r:
        if r.status_code != 200:
            res.error = {"http": r.status_code, "body": (await r.aread()).decode()[:200]}
            return res, sid
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            t = ev.get("type")
            if t == "meta":
                sid = ev.get("session_id")
            elif t == "tool_call":
                res.tools.append(ev.get("tool"))
                res.tool_args.append(ev.get("args") or {})
            elif t == "clarification":
                res.clarification = ev
            elif t == "final":
                res.final = ev
            elif t == "error":
                res.error = ev
    return res, sid


def _combined_reply(clar: dict) -> str:
    """Auto-answer: pick option[0] of each question; build the combined reply."""
    qs = clar.get("questions") or []
    parts = []
    for q in qs:
        opts = q.get("options") or []
        if not opts:
            continue
        o = opts[0]
        reply = (f"{o['label']} — security_id {o['value']}"
                 if isinstance(o.get("value"), int) else o["label"])
        parts.append(f"{q['id']}: {reply}" if len(qs) > 1 else reply)
    return "; ".join(parts)


async def run_case(client: httpx.AsyncClient, base: str, firm: str, case: Case) -> list[str]:
    """Run all turns; return a list of failure strings ([] = pass)."""
    headers = {"Content-Type": "application/json", "X-Dev-Firm": firm,
               "Accept": "text/event-stream"}
    fails: list[str] = []
    sid: str | None = None
    prev: TurnResult | None = None
    for i, turn in enumerate(case.turns):
        if turn.get("answer_clarification"):
            if not prev or not prev.clarification:
                fails.append(f"turn{i}: expected a prior clarification to answer")
                break
            message = _combined_reply(prev.clarification)
        else:
            message = turn["message"]
        res, sid = await _run_turn(client, base, headers, message, sid)
        for chk in turn.get("expect", []):
            reason = chk(res)
            if reason:
                fails.append(f"turn{i}: {reason}")
        prev = res
    return fails


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--firm", default="EVAL_RUN")
    ap.add_argument("--only", default="", help="comma-separated case ids")
    ap.add_argument("--runs", type=int, default=1, help="run each case N times (flakiness)")
    args = ap.parse_args()

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    cases = [c for c in CASES if not only or c.id in only]

    runs = max(1, args.runs)
    label = f" ({runs}x for flakiness)" if runs > 1 else ""
    print(f"PRISM evals — {len(cases)} case(s){label} vs {args.base}\n" + "=" * 60)
    total = ok = 0
    async with httpx.AsyncClient() as client:
        for c in cases:
            c_pass = 0
            for _ in range(runs):
                try:
                    fails = await run_case(client, args.base, args.firm, c)
                except Exception as exc:  # noqa: BLE001 — a crashed case is a fail
                    fails = [f"harness error: {type(exc).__name__}: {exc}"]
                total += 1
                ok += not fails
                c_pass += not fails
                if fails:
                    for f in fails:
                        print(f"   FAIL {c.id}: {f}")
            mark = "PASS" if c_pass == runs else ("FLAKY" if c_pass else "FAIL")
            print(f"[{mark:5}] {c.id:26} {c_pass}/{runs} ({c.intent})")
    print("=" * 60)
    print(f"{ok}/{total} runs passed")
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
