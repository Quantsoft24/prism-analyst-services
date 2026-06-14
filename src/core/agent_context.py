"""Per-run agent context.

The ``AgentRunner`` stashes the request's ``firm_id`` here so tools that persist
PER-TENANT data — notably the BMC service, keyed by ``(firm_id, ticker)`` — write
and read under the SAME firm the frontend uses. Without this, the agent's BMC
tools fall back to the service's ``DEFAULT_FIRM_ID`` ("default") while the
frontend proxy injects the logged-in firm, so a canvas the agent generates is
invisible on ``/bmc`` (and never cache-hits → repeated cold regenerations).

A ``ContextVar`` is task-local; the async tools are awaited inline within the
same run task, so they inherit whatever the runner set. The LLM never sees this
— it's pure server-side context, not a tool argument.
"""

from __future__ import annotations

from contextvars import ContextVar

current_firm_id: ContextVar[str | None] = ContextVar("prism_current_firm_id", default=None)
