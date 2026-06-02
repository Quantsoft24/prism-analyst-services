"""Safe arithmetic expression engine for custom factors.

A custom factor composes existing factors, e.g. ``(roe + earnings_yield) / pb``.
Expressions are parsed with ``ast`` and evaluated by an explicit recursive walker
(never ``eval``) over a per-security dict of base-factor values — only names,
numeric literals, ``+ - * /``, unary minus, and parentheses are allowed. Any
referenced factor that is ``None`` (or a divide-by-zero) makes the result
``None`` (counted in coverage, never zero-filled).
"""

from __future__ import annotations

import ast

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)


class ExpressionError(ValueError):
    """Raised when an expression is malformed or references unknown factors."""


def _check_node(node: ast.AST, refs: set[str]) -> None:
    if isinstance(node, ast.Expression):
        _check_node(node.body, refs)
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise ExpressionError("Only + - * / are allowed.")
        _check_node(node.left, refs)
        _check_node(node.right, refs)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ExpressionError("Only unary +/- is allowed.")
        _check_node(node.operand, refs)
    elif isinstance(node, ast.Name):
        refs.add(node.id)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise ExpressionError("Only numeric literals are allowed.")
    else:
        raise ExpressionError(f"Disallowed expression element: {type(node).__name__}.")


def parse_refs(expression: str) -> set[str]:
    """Parse + validate structure; return the set of factor names referenced."""
    if not expression or not expression.strip():
        raise ExpressionError("Expression is empty.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Syntax error: {exc.msg}") from exc
    refs: set[str] = set()
    _check_node(tree, refs)
    if not refs:
        raise ExpressionError("Expression must reference at least one factor.")
    return refs


def validate(expression: str, allowed_ids: set[str]) -> set[str]:
    """Validate structure + that every referenced name is a known factor id."""
    refs = parse_refs(expression)
    unknown = refs - allowed_ids
    if unknown:
        raise ExpressionError(f"Unknown factor(s): {', '.join(sorted(unknown))}.")
    return refs


def _eval(node: ast.AST, values: dict[str, float | None]) -> float | None:
    if isinstance(node, ast.Expression):
        return _eval(node.body, values)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return values.get(node.id)
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, values)
        if v is None:
            return None
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, values), _eval(node.right, values)
        if a is None or b is None:
            return None
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            return None if b == 0 else a / b
    raise ExpressionError("Unevaluable expression.")


def evaluate(expression: str, values: dict[str, float | None]) -> float | None:
    """Evaluate ``expression`` against per-security base-factor ``values``."""
    tree = ast.parse(expression, mode="eval")
    return _eval(tree, values)
