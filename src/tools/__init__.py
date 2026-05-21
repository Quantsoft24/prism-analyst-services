"""PRISM Tools — agent-callable tools.

Every tool here is a Python function decorated with an ADK ``FunctionTool``
wrapper. The function's docstring + type hints become the tool's schema that
the LLM sees, so **docstrings here are not optional** — they're how the agent
learns when to call which tool.

Adding a tool checklist:
  1. Define a typed Python function with a clear NumPy-style docstring.
  2. Open any DB session needed *inside* the function via ``session_scope()``
     — never assume an outer transaction.
  3. Return JSON-serializable data only (dict / list / primitives).
  4. Wrap with ``FunctionTool`` in the module's tool list.
  5. Pass the tool into the relevant ``PrismAgent``'s ``tools`` list.

Tools are shared across agents — BMC, Writer, Modelling will all reuse these
company lookup tools.
"""

from src.tools.bmc_tools import BMC_TOOLS
from src.tools.company_tools import COMPANY_TOOLS
from src.tools.filing_tools import FILING_TOOLS
from src.tools.nre_tools import NRE_TOOLS

__all__ = ["COMPANY_TOOLS", "FILING_TOOLS", "BMC_TOOLS", "NRE_TOOLS"]
