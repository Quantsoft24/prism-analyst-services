# PRISM agent evals

A graded suite that drives the **real `/chat/run` pipeline** (custom AgentRunner:
clarification events, two-tier composer, citation merge) and asserts *behaviors*
across the analyst intent taxonomy + every bug we've hit. This is the root fix for
fixing things one screenshot at a time — regressions get caught here.

## Run (opt-in; needs the backend live + Gemini keys; consumes message quota)

```bash
python evals/run.py                         # all cases vs http://localhost:8000
python evals/run.py --only blinkit-not-found,compare-table
python evals/run.py --firm EVAL_RUN         # use a dedicated dev firm for quota
```

Exit code is non-zero if any case fails (CI-friendly).

## Add a case

Append a `Case(id, intent, turns=[...])` to `CASES` in `run.py`. A turn is
`{"message": "...", "expect": [assertions]}`, or `{"answer_clarification": True}`
which auto-picks option[0] of every clarification question and sends the combined
reply. Assertions (in `run.py`) check *behaviors*, not exact strings:
`no_clarification()`, `clarification_questions(n)`, `tools_any(...)`,
`tool_arg_contains(tool, key, regex)`, `has_final()`, `has_table()`,
`answer_matches(rx)`, `answer_excludes(rx)`, `filing_citations_present()`.

**New analyst intents become new cases here — not prompt surgery.**

## Why not ADK `AgentEvaluator`?

ADK's evaluator runs the agent through ADK's *standard* runner, which bypasses
PRISM's custom `AgentRunner` where the clarification flow, two-tier composition,
and citation enrichment live — i.e. exactly what we keep fixing. This harness
drives the actual pipeline. (We still adopt other ADK primitives — planner,
sessions/state, callbacks — per the roadmap.)
