"""Agent-callable Numerical Reasoning Engine tools.

These are how an agent does math WITHOUT doing math: it passes numbers it read
from filings, the deterministic engine computes, the agent verbalizes the
result. Each tool returns a JSON dict (errors as ``{"error": ...}`` rather than
raising) so the agent can recover gracefully — e.g. ask for a different number.

Stateless + no I/O, so no ``session_scope`` needed (unlike the DB-backed tools).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.integrations.tools._errors import make_error
from src.services.nre import engine

if TYPE_CHECKING:
    from google.adk.tools import FunctionTool


def _safe(fn, **kwargs) -> dict:
    """Run a deterministic NRE engine function and return the standard
    result/error shape. Computation errors (divide-by-zero, negative log
    inputs) are surfaced as ``ask_user_to_clarify`` — the agent should ask
    the user to recheck the numbers rather than retrying.
    """
    try:
        return fn(**kwargs).to_dict()
    except engine.NREError as exc:
        return make_error(
            message=str(exc),
            code="nre_invalid_inputs",
            next_action="ask_user_to_clarify",
            retriable=False,
        )


def compute_growth(start: float, end: float) -> dict:
    """Compute percentage growth from ``start`` to ``end`` (e.g. YoY/QoQ revenue
    growth). ALWAYS use this instead of computing percentages yourself.

    Args:
        start: The earlier-period value (e.g. prior-year revenue).
        end: The later-period value (e.g. current-year revenue).

    Returns:
        ``{operation, result, unit:"%", inputs}`` or ``{error}`` if start is 0.
    """
    return _safe(engine.growth_pct, start=start, end=end)


def compute_cagr(start: float, end: float, periods: float) -> dict:
    """Compute compound annual growth rate (CAGR) over a number of periods.

    Args:
        start: Value at the start of the window (must be > 0).
        end: Value at the end of the window.
        periods: Number of periods (years) between them (must be > 0).

    Returns:
        ``{operation, result, unit:"%", inputs}`` or ``{error}``.
    """
    return _safe(engine.cagr_pct, start=start, end=end, periods=periods)


def compute_margin(numerator: float, denominator: float) -> dict:
    """Compute a margin or share as a percentage (numerator / denominator * 100),
    e.g. operating margin = operating income / revenue.

    Returns ``{operation, result, unit:"%", inputs}`` or ``{error}`` if
    denominator is 0.
    """
    return _safe(engine.margin_pct, numerator=numerator, denominator=denominator)


def compute_ratio(a: float, b: float) -> dict:
    """Compute the ratio a / b (e.g. debt-to-equity). Returns unit 'x'."""
    return _safe(engine.ratio, a=a, b=b)


def compute_percent_of(part: float, whole: float) -> dict:
    """Compute what percent ``part`` is of ``whole`` (part / whole * 100)."""
    return _safe(engine.pct_of, part=part, whole=whole)


# ── ADK FunctionTool wrappers (lazy; mirrors other tool modules) ────────────


def _build_tools() -> list["FunctionTool"]:
    from google.adk.tools import FunctionTool

    return [
        FunctionTool(func=compute_growth),
        FunctionTool(func=compute_cagr),
        FunctionTool(func=compute_margin),
        FunctionTool(func=compute_ratio),
        FunctionTool(func=compute_percent_of),
    ]


class _LazyToolList:
    def __init__(self) -> None:
        self._tools: list[FunctionTool] | None = None

    def _ensure(self) -> list:
        if self._tools is None:
            self._tools = _build_tools()
        return self._tools

    def __iter__(self):
        return iter(self._ensure())

    def __len__(self) -> int:
        return len(self._ensure())

    def to_list(self) -> list:
        return list(self._ensure())


NRE_TOOLS = _LazyToolList()
