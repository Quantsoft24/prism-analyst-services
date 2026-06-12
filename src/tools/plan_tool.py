"""``update_plan`` — the agent's visible task checklist (Claude-Code-style).

The agent calls this to DECLARE its plan for a multi-step task and to UPDATE step
statuses as it works. Each call returns a ``_plan`` marker the runner intercepts
to emit a ``PlanEvent`` (no tool card, no evidence) — the UI renders the latest
steps as checkboxes that tick off. Makes the agent's reasoning visible = feels
agentic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.adk.tools import FunctionTool

# Marker the runner detects on this tool's response.
PLAN_MARKER = "_plan"

_VALID_STATUS = {"pending", "in_progress", "done"}


async def update_plan(tasks: list[dict]) -> dict:
    """Declare the user-visible task checklist for a MULTI-STEP request.

    Call this ONCE near the start when a question needs several steps (e.g. resolve
    companies → pull financials → compose a comparison), passing the whole ordered
    list with the first task ``in_progress`` and the rest ``pending``. You do NOT
    need to call it again or update statuses — the runner ticks each task off
    automatically as that step's tool work actually completes, so the checklist
    stays in sync with the real work. Skip it for trivial one-step answers. Keep
    titles short and user-facing ("Compare FY24 margins").

    Args:
        tasks: ordered list of ``{"title": str, "status": "pending"|"in_progress"
            |"done"}``. ``status`` defaults to ``pending``.

    Returns:
        A ``_plan`` control payload (the runner turns it into the checklist).
    """
    steps: list[dict] = []
    for i, t in enumerate(tasks or []):
        if not isinstance(t, dict):
            continue
        title = str(t.get("title") or "").strip()
        if not title:
            continue
        status = t.get("status") if t.get("status") in _VALID_STATUS else "pending"
        steps.append({"id": str(t.get("id") or f"s{i}"), "title": title, "status": status})
    return {PLAN_MARKER: {"steps": steps}}


def _build_tools() -> list["FunctionTool"]:
    from google.adk.tools import FunctionTool

    return [FunctionTool(func=update_plan)]


class _LazyToolList:
    def __init__(self) -> None:
        self._tools: list[FunctionTool] | None = None

    def to_list(self) -> list:
        if self._tools is None:
            self._tools = _build_tools()
        return list(self._tools)


PLAN_TOOLS = _LazyToolList()
