"""
Web search tool using DuckDuckGo via the duckduckgo-search package.

Performs text searches with no API key required. Results are formatted
as a numbered list suitable for inclusion in LLM prompts.
"""

import logging
from typing import List

from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


def search(query: str, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo and return formatted results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default: 3).

    Returns:
        A formatted string of numbered results, each with title, snippet,
        and URL. Returns "No search results found." if nothing is found.

    Raises:
        ValueError: If the query is empty or whitespace-only.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")

    query = query.strip()
    logger.info("Web search: '%s' (max_results=%d)", query, max_results)

    try:
        results: List[dict] = []
        with DDGS() as ddgs:
            for result in ddgs.text(query, max_results=max_results):
                results.append(result)

    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
        return f"Web search failed: {exc}"

    if not results:
        logger.info("No results found for query: '%s'", query)
        return "No search results found."

    # Format as numbered list
    formatted_lines: List[str] = []
    for idx, result in enumerate(results, start=1):
        title = result.get("title", "No title")
        body = result.get("body", "No description")
        href = result.get("href", "No URL")
        formatted_lines.append(f"{idx}. [{title}]: {body}\n   URL: {href}")

    output = "\n\n".join(formatted_lines)
    logger.info("Found %d search results", len(results))
    return output
