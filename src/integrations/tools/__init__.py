"""In-process tool wrappers for `source: python` integrations.

Each module here exposes a typed async function (or a list of them) that the
integration registry wraps in an ADK ``FunctionTool``. The function's
**docstring + type hints are the contract the LLM sees** — keep them precise.
"""
