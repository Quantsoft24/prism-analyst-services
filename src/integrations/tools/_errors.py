"""Standard error shape for agent-callable tools.

Every tool — built-in (``src/tools/``) or integration (``src/integrations/tools/``)
— returns one of two dict shapes:

  • SUCCESS — tool-specific keys (no ``ok`` field needed; absence of error
              is the success signal). For symmetry, success paths MAY also
              return ``{"ok": True, ...}`` but it's not required.

  • FAILURE — exactly this shape:
        {
          "ok": False,
          "error": "<short user-facing sentence>",
          "error_code": "<machine_readable_token>",
          "next_action": "<one of NextAction>",
          "retriable": bool,
          "detail": "<optional debug detail>",  # may be omitted
        }

The agent's system prompt (see ``src/agents/company_intel.py``) is taught how
to react to each ``next_action``:

  • ``ask_user_to_retry_later``  — upstream is down; tell the user and stop.
  • ``try_alternate_tool``       — this tool can't help; reach for another.
  • ``ask_user_to_clarify``      — input was ambiguous; ask a follow-up.
  • ``give_up_gracefully``       — non-recoverable but not surprising;
                                    surface a clean apology to the user.

The runner (``src/services/agent_runner.py``) inspects the response and emits
``ToolResultEvent(ok=False, error=...)`` when ``ok`` is False OR ``error`` is
present (the latter for legacy callers we haven't migrated yet). The retry
middleware re-invokes the tool exactly once when ``retriable=True`` AND the
``next_action`` is ``ask_user_to_retry_later``.
"""

from __future__ import annotations

from typing import Literal

NextAction = Literal[
    "ask_user_to_retry_later",
    "try_alternate_tool",
    "ask_user_to_clarify",
    "give_up_gracefully",
]


def make_error(
    *,
    message: str,
    code: str,
    next_action: NextAction,
    retriable: bool = False,
    detail: str | None = None,
) -> dict:
    """Build the standard error dict.

    Args:
        message: Short user-facing sentence (the LLM may quote it).
        code: snake_case machine token (used by frontend for icons / styling).
        next_action: see module docstring.
        retriable: True iff a single retry has a real chance of succeeding
            (typically network blips). The runner uses this to decide
            whether to auto-retry.
        detail: optional debug detail (HTTP status, exception text, etc.) —
            kept short by callers; trimmed to 500 chars defensively.
    """
    payload: dict = {
        "ok": False,
        "error": message,
        "error_code": code,
        "next_action": next_action,
        "retriable": retriable,
    }
    if detail:
        payload["detail"] = detail[:500]
    return payload


def is_error(response: object) -> bool:
    """True when a tool response represents a failure under either:
      (a) the new shape (``ok=False``), or
      (b) the legacy bare ``{"error": ...}`` shape that older tools still emit.

    Used by the runner to map ADK tool results onto ``ToolResultEvent.ok``.
    """
    if not isinstance(response, dict):
        return False
    if response.get("ok") is False:
        return True
    if "error" in response and response["error"]:
        return True
    return False


def is_retriable(response: object) -> bool:
    """True iff the runner's single-retry middleware should re-invoke the tool."""
    if not isinstance(response, dict):
        return False
    return bool(response.get("retriable"))


def extract_error_message(response: object) -> str | None:
    """Pull a user-facing error string out of either shape, or None if not an error."""
    if not isinstance(response, dict):
        return None
    if response.get("ok") is False or "error" in response:
        return response.get("error") or "tool error"
    return None
