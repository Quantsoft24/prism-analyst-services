r"""Regression guards on the agent system prompts.

ADK's ``LlmAgent`` runs every instruction through
``google.adk.utils.instructions_utils.inject_session_state`` BEFORE sending
it to the model. That helper matches ``{identifier}`` patterns and treats
them as session-state lookups — raising ``KeyError('Context variable not
found: `X`.')`` when the name isn't a key on the session state.

We learned this the hard way: a literal ``FY{a} → FY{b}`` in the prompt
(a documentation example for the agent's tool catalogue) caused EVERY chat
turn to crash before reaching the model. This test scans all agent prompts
for the same hazard so it can't regress.

ADK matcher (paraphrased — see ``google/adk/utils/instructions_utils.py``):
    re.finditer(r'{+[^{}]*}+', template)
    name = match.group().lstrip('{').rstrip('}').strip().rstrip('?')
    if name.isidentifier():
        # treated as a session-state variable; KeyError when not found.

Double-brace escaping (``{{x}}``) does NOT save you — the .lstrip/.rstrip
peels every leading/trailing brace off.
"""

from __future__ import annotations

import re

import pytest

from src.agents.base import FINANCE_DOMAIN_RULES
from src.agents.company_intel import COMPANY_INTEL_INSTRUCTION


def _adk_template_hits(template: str) -> list[tuple[int, str, str]]:
    """Return every (offset, raw_match, var_name) that ADK would treat as a
    session-state variable in ``template``."""
    hits: list[tuple[int, str, str]] = []
    for match in re.finditer(r"{+[^{}]*}+", template):
        raw = match.group()
        name = raw.lstrip("{").rstrip("}").strip()
        if name.endswith("?"):
            name = name[:-1]
        if name.isidentifier():
            hits.append((match.start(), raw, name))
    return hits


@pytest.mark.parametrize(
    "name,template",
    [
        ("FINANCE_DOMAIN_RULES", FINANCE_DOMAIN_RULES),
        ("COMPANY_INTEL_INSTRUCTION", COMPANY_INTEL_INSTRUCTION),
    ],
    # Explicit IDs — pytest's default ID-from-value embeds the full prompt
    # string into the test name. On Windows that hits the 32767-char env
    # var limit during test collection. Stable short IDs avoid it.
    ids=["finance_rules", "company_intel"],
)
def test_prompt_has_no_adk_template_hazards(name: str, template: str) -> None:
    """Fail if the prompt contains a ``{identifier}`` literal that ADK would
    misinterpret as a session-state variable.

    Fix when this fails: replace the literal with something whose stripped
    content is NOT a valid Python identifier — angle brackets, an actual
    example value, or a backticked placeholder all work. See git blame on
    src/agents/company_intel.py for the canonical fix pattern.
    """
    hits = _adk_template_hits(template)
    if hits:
        diag = "\n".join(
            f"  at offset {off}: {raw!r} would resolve as var {var!r}"
            for off, raw, var in hits
        )
        pytest.fail(
            f"{name} contains {len(hits)} ADK template hazard(s):\n{diag}\n"
            "ADK treats these as session-state lookups → KeyError at runtime. "
            "Replace with non-identifier text (e.g. <a>, FY24, or `placeholder`)."
        )


def test_web_search_prompt_has_no_adk_template_hazards() -> None:
    """Same guard for the web_search subagent's instruction. Imports ADK
    lazily so the test file is importable without the SDK installed."""
    pytest.importorskip("google.adk")
    from src.agents.web_search import build_web_search_agent

    agent = build_web_search_agent()
    hits = _adk_template_hits(agent.instruction)
    if hits:
        diag = "\n".join(
            f"  at offset {off}: {raw!r} would resolve as var {var!r}"
            for off, raw, var in hits
        )
        pytest.fail(f"web_search.instruction has hazards:\n{diag}")


def test_adk_template_hits_finds_known_hazards() -> None:
    """Smoke test the detector against the original bug — guards the test
    against silently passing if someone breaks the regex."""
    bad = "Example: FY{a} → FY{b} and another {count} marker."
    hits = _adk_template_hits(bad)
    names = {var for _, _, var in hits}
    assert names == {"a", "b", "count"}, names


def test_adk_template_hits_ignores_json_literals() -> None:
    """JSON-shaped braces (with colons, quotes, commas) aren't hazards
    because the stripped content isn't a valid identifier."""
    ok = '<answer_meta>{"confidence":"high","data_freshness":"2026-Q4"}</answer_meta>'
    assert _adk_template_hits(ok) == []
