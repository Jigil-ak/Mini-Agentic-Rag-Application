"""
Tool Agent — routes the query to a tool (calculator or web search) and synthesises an answer.
"""

import re
import logging
from typing import Dict

from backend.tools.calculator import CalculatorTool
from backend.tools.web_search import WebSearchTool
from backend.agent.fallback import call_with_fallback

logger = logging.getLogger(__name__)


def _extract_math_expression(query: str) -> str:
    """Extract a mathematical expression from the user query."""
    cleaned = query.lower()
    for kw in ['calculate', 'compute', 'evaluate', 'solve', 'what is']:
        cleaned = cleaned.replace(kw, '')
    cleaned = cleaned.replace('?', '').strip()
    allowed_chars = set("0123456789+-*/^%() .")
    return "".join([c for c in cleaned if c in allowed_chars]).strip()


def run_tool(query: str) -> Dict:
    """Choose between calculator and web search, run it, and synthesise the final response.

    Args:
        query: User question.

    Returns:
        Dict with keys: answer, tool_used, tool_output, model_used, fallback_triggered.
    """
    try:
        # Tool routing logic
        math_keywords = ['calculate', 'compute', 'evaluate', 'solve']
        math_pattern = re.compile(r'[\d\+\-\*\/\^\%\(\)]{2,}')

        if any(kw in query.lower() for kw in math_keywords) or math_pattern.search(query):
            tool_name = "calculator"
            calculator = CalculatorTool()
            expr = _extract_math_expression(query)
            try:
                tool_output = calculator.calculate(expr)
            except Exception as e:
                tool_output = f"Calculator error: {e}"
        else:
            tool_name = "web_search"
            searcher = WebSearchTool()
            tool_output = searcher.search(query)

        print(f"[ToolAgent] Tool selected: {tool_name}")
        print(f"[ToolAgent] Tool output preview: {tool_output[:200]}")

        # Build synthesis prompt
        prompt = f"Based on this information:\n{tool_output}\n\nAnswer this question: {query}\n\nAnswer:"
        result = call_with_fallback(prompt)

        return {
            "answer": result["response"],
            "tool_used": tool_name,
            "tool_output": tool_output,
            "model_used": result["model_used"],
            "fallback_triggered": result["fallback_triggered"],
        }
    except Exception as exc:
        logger.error("Tool agent pipeline failed: %s", exc, exc_info=True)
        return {
            "answer": f"Tool execution failed: {exc}",
            "tool_used": "none",
            "tool_output": f"Error: {exc}",
            "model_used": "None",
            "fallback_triggered": False,
        }
