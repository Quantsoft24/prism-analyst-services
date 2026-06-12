"""Agent-callable clarification tool.

Lets the agent PAUSE and ask the user a structured question — a single-select
MCQ, a multi-select, or an open-text prompt — when it genuinely cannot proceed
without input (which of several companies they mean, which metric/period, …).
The agent composes the question + options; the runner turns the return value
into a terminal ``ClarificationEvent`` (see ``src/services/agent_runner.py``)
and the user's selection arrives as their next message.

Safety: for company picks each option's ``value`` should be a master_securities
``security_id``; we validate int values against the resolver and drop unknown
ones, so a mistyped id can't mis-route a downstream tool.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.services import company_resolver as cr

if TYPE_CHECKING:
    from google.adk.tools import FunctionTool

logger = logging.getLogger(__name__)

# Marker key the runner looks for to convert this into a ClarificationEvent.
CLARIFY_MARKER = "_clarification"
_VALID_MODES = {"single_select", "multi_select", "open_text"}


async def request_clarification(
    question: str,
    options: list[dict[str, Any]] | None = None,
    mode: str = "single_select",
) -> dict:
    """Ask the USER a clarifying question and PAUSE the turn. Use this whenever
    you cannot answer accurately without the user's input — most importantly to
    disambiguate a company when ``resolve_company`` returns ``needs_clarification``
    (e.g. "Reliance" → 8 companies), but also for an unclear metric, period, or
    scope. The turn ends here; the user's choice comes back as their next message.

    Choose the format that fits the situation:
      • ``single_select`` — one pick from ``options`` (default; e.g. which Reliance).
      • ``multi_select``  — several picks (e.g. which companies to compare).
      • ``open_text``     — a free-text answer, no options (e.g. "which fiscal year?").

    For a company pick, build ``options`` from ``resolve_company``'s
    ``clarification.options`` — each is ``{id, label, hint, value}`` where
    ``value`` is the company's ``security_id``. Keep the labels/hints so the user
    can tell them apart; do NOT invent companies or ids.

    Args:
        question: The single, clear question to show the user.
        options: For select modes, a list of ``{label, value, id?, hint?}``.
            ``value`` is returned when chosen (a security_id for company picks).
            Omit or leave empty for ``open_text``.
        mode: ``single_select`` | ``multi_select`` | ``open_text``.

    Returns:
        A control payload the UI renders. NOTE: you do NOT receive the user's
        answer from this call — it arrives as their next message, so STOP after
        calling this.
    """
    m = mode if mode in _VALID_MODES else "single_select"
    clean: list[dict[str, Any]] = []
    for i, o in enumerate(options or []):
        if not isinstance(o, dict):
            continue
        label = str(o.get("label") or o.get("name") or "").strip()
        if not label:
            continue
        value = o.get("value", o.get("id"))
        # Validate security_id-shaped values against the resolver.
        if isinstance(value, int):
            if await cr.get_by_security_id(value) is None:
                logger.warning("request_clarification: dropping unknown security_id=%s", value)
                continue
        clean.append({
            "id": str(o.get("id") or value or i),
            "label": label,
            "hint": o.get("hint"),
            "value": value,
        })
    if m != "open_text" and not clean:
        m = "open_text"  # no valid options → still ask, as free text
    return {
        CLARIFY_MARKER: {
            "question": (question or "Could you clarify what you mean?").strip(),
            "mode": m,
            "options": clean,
            "allow_search": True,
        }
    }


def _build_tools() -> list["FunctionTool"]:
    from google.adk.tools import FunctionTool

    return [FunctionTool(func=request_clarification)]


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


CLARIFY_TOOLS = _LazyToolList()
