"""
Tool agent — routes to calculator or web search based on query analysis.

When the knowledge base doesn't have relevant context (similarity below
threshold), the tool agent determines the appropriate tool, executes it,
and passes the tool output to the LLM for final answer generation.
"""

import logging
import re
from typing import Dict

from backend.tools.calculator import calculate
from backend.tools.web_search import search
from backend.agent.fallback import call_with_fallback

logger = logging.getLogger(__name__)

# ── Calculator detection patterns ──────────────────────────────────────────
_CALC_KEYWORDS = {"calculate", "compute", "evaluate", "solve", "math"}
_CALC_OPERATORS = {"+", "-", "*", "/", "^", "%", "**"}
_CALC_PATTERN = re.compile(r"\d+\s*[\+\-\*\/\%\^]\s*\d+")


def _is_calculator_query(query: str) -> bool:
    """Determine if a query should be routed to the calculator tool.

    Returns True if the query contains calculator keywords, arithmetic
    operators alongside digits, or matches digit-operator-digit patterns.
    """
    lower = query.lower()

    # Check for calculator keywords
    for keyword in _CALC_KEYWORDS:
        if keyword in lower:
            return True

    # Check for arithmetic operator patterns with digits
    if _CALC_PATTERN.search(query):
        return True

    # Check if query has operators and digits together
    has_digit = any(c.isdigit() for c in query)
    has_op = any(op in query for op in _CALC_OPERATORS)
    if has_digit and has_op:
        return True

    return False


def _extract_expression(query: str) -> str:
    """Extract the mathematical expression from a query string.

    Attempts to pull out just the math portion. Falls back to
    stripping common words if no clean expression is found.
    """
    # Try to find a clear mathematical expression
    match = re.search(r"[\d\.\s\+\-\*\/\%\^\(\)]+", query)
    if match:
        expr = match.group().strip()
        # Only use if it contains at least one digit and one operator
        if any(c.isdigit() for c in expr) and any(c in expr for c in "+-*/%^"):
            return expr

    # Fall back: remove common words and keep the math part
    cleaned = query.lower()
    for word in ["calculate", "compute", "evaluate", "solve", "what is", "what's", "how much is"]:
        cleaned = cleaned.replace(word, "")
    return cleaned.strip()


# ── LLM prompt template ───────────────────────────────────────────────────
_TOOL_PROMPT = """Based on the following information, answer the user's question.

TOOL OUTPUT:
{tool_output}

USER QUESTION:
{query}

Provide a clear, helpful answer based on the tool output above. If the tool output is a calculation result, present it clearly. If it contains search results, synthesise the information into a coherent answer."""


# ── Public API ─────────────────────────────────────────────────────────────

def run_tool(query: str) -> Dict:
    """Execute the appropriate tool and generate an LLM answer.

    Tool routing logic:
    - Calculator: if query contains math keywords or digit-operator patterns
    - Web search: for all other queries

    Args:
        query: The user's question.

    Returns:
        Dict with keys:
            answer (str): The generated answer.
            tool_used (str): "calculator" or "web_search".
            tool_output (str): Raw output from the tool.
            model_used (str): Which LLM produced the answer.
            fallback_triggered (bool): Whether the fallback LLM was used.
    """
    # ── Route to appropriate tool ──────────────────────────────────────
    if _is_calculator_query(query):
        tool_used = "calculator"
        logger.info("Tool agent: routing to calculator")

        expression = _extract_expression(query)
        try:
            tool_output = calculate(expression)
            logger.info("Calculator result: %s = %s", expression, tool_output)
        except ValueError as exc:
            tool_output = f"Calculator error: {exc}"
            logger.warning("Calculator failed for '%s': %s", expression, exc)
    else:
        tool_used = "web_search"
        logger.info("Tool agent: routing to web search for '%s'", query[:80])

        tool_output = search(query, max_results=3)
        logger.info("Web search returned %d characters", len(tool_output))

    # ── Generate final answer via LLM ──────────────────────────────────
    prompt = _TOOL_PROMPT.format(tool_output=tool_output, query=query)
    llm_result = call_with_fallback(prompt)

    return {
        "answer": llm_result["response"],
        "tool_used": tool_used,
        "tool_output": tool_output,
        "model_used": llm_result["model_used"],
        "fallback_triggered": llm_result["fallback_triggered"],
    }
