"""
Safe mathematical expression evaluator using the ast module.

Evaluates arithmetic expressions without using Python's built-in eval(),
which would be a security risk. Only pure math operations are permitted:
addition, subtraction, multiplication, division, exponentiation, modulo,
and unary plus/minus.
"""

import ast
import logging
import operator
from typing import Union

logger = logging.getLogger(__name__)

# ── Allowed binary operators ───────────────────────────────────────────────
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

# ── Allowed unary operators ────────────────────────────────────────────────
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> Union[int, float]:
    """Recursively evaluate an AST node, allowing only safe math operations.

    Args:
        node: An AST node from a parsed expression.

    Returns:
        The numeric result of the expression.

    Raises:
        ValueError: If the node type is not in the allowed set.
    """
    # Numeric literal (Python 3.8+: ast.Constant replaces ast.Num)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(
            f"Unsafe expression: unsupported constant type {type(node.value).__name__}"
        )

    # Legacy ast.Num support (Python < 3.8 compatibility)
    if isinstance(node, ast.Num):
        return node.n

    # Binary operation: left op right
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise ValueError(f"Unsafe expression: unsupported operator {op_type.__name__}")

        left = _safe_eval(node.left)
        right = _safe_eval(node.right)

        # Guard against excessively large exponents
        if op_type is ast.Pow and isinstance(right, (int, float)) and abs(right) > 1000:
            raise ValueError(
                f"Unsafe expression: exponent too large ({right}). Max allowed: 1000"
            )

        return _BINARY_OPS[op_type](left, right)

    # Unary operation: +x or -x
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unsafe expression: unsupported unary operator {op_type.__name__}")
        return _UNARY_OPS[op_type](_safe_eval(node.operand))

    # Top-level Expression wrapper
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    # Anything else is unsafe
    raise ValueError(
        f"Unsafe expression: disallowed AST node type {type(node).__name__}"
    )


# ── Public API ─────────────────────────────────────────────────────────────

def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Supports: +, -, *, /, ** (power), % (modulo), unary +/-.
    Does NOT support: function calls, variable access, imports,
    attribute access, or any non-arithmetic operations.

    Args:
        expression: A string containing a mathematical expression,
                     e.g. "2 + 3 * (4 - 1)" or "2 ** 8".

    Returns:
        String representation of the result.
        Integers are returned without decimal point.

    Raises:
        ValueError: On empty input, syntax errors, unsafe expressions,
                     division by zero, or numeric overflow.
    """
    if not expression or not expression.strip():
        raise ValueError("Expression cannot be empty")

    expr = expression.strip()

    # Normalise common alternative notations
    expr = expr.replace("^", "**")  # caret → exponent

    logger.debug("Evaluating expression: %s", expr)

    # Parse the expression into an AST
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    # Evaluate the AST safely
    try:
        result = _safe_eval(tree)
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except OverflowError:
        raise ValueError("Numeric overflow — result too large to represent")

    # Format output: drop .0 for integer results
    if isinstance(result, float) and result.is_integer():
        return str(int(result))

    return str(result)
