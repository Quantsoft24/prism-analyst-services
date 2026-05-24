"""Numerical Reasoning Engine (NRE).

Architectural commitment from ``final_docs/02_ARCHITECTURE_AND_STACK.md``:
**the LLM never does arithmetic.** Any derived financial figure — growth %,
CAGR, margin, ratio, delta — is computed by deterministic Python here, not by
the model. The agent passes raw numbers it READ from filings; the NRE returns
the computed answer with full provenance (inputs echoed, operation named,
unit attached). This is what lets PRISM put a number in front of an analyst
without the hallucination risk that kills every ChatGPT-grade finance tool.

Two layers:
  * ``engine.py`` — pure, typed functions. No I/O. Fully unit-tested.
  * ``../tools/nre_tools.py`` — ADK FunctionTool wrappers agents call.
"""

from src.services.nre.engine import (
    NREError,
    NREResult,
    average,
    cagr_pct,
    delta,
    growth_pct,
    margin_pct,
    pct_of,
    ratio,
    sum_values,
)

__all__ = [
    "NREError",
    "NREResult",
    "growth_pct",
    "cagr_pct",
    "margin_pct",
    "ratio",
    "delta",
    "pct_of",
    "sum_values",
    "average",
]
