"""
Web search tool using DuckDuckGo via the duckduckgo-search package.
"""

import logging
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)


def _strip_non_ascii(text: str) -> str:
    """Helper to remove non-ASCII characters from search result fields."""
    if not text:
        return ""
    return "".join(c for c in text if ord(c) < 128)


class WebSearchTool:
    """Web search tool class using the DDGS client directly."""

    def search(self, query: str, max_results: int = 3) -> str:
        """Perform a web search and format results as a numbered list.

        Args:
            query: The search query.
            max_results: Max results to fetch.

        Returns:
            A formatted string list of results or a fallback message if failed.
        """
        print(f"[WebSearch] Searching for: {query}")
        try:
            results = []
            with DDGS() as ddgs:
                ddgs_generator = ddgs.text(query, max_results=max_results)
                if ddgs_generator:
                    for r in ddgs_generator:
                        results.append(r)

            n = len(results)
            print(f"[WebSearch] Found {n} results")

            if not results:
                return "No search results found."

            formatted_results = []
            for idx, r in enumerate(results, start=1):
                title = _strip_non_ascii(r.get("title", ""))
                snippet = _strip_non_ascii(r.get("body", ""))
                url = _strip_non_ascii(r.get("href", ""))
                formatted_results.append(f"{idx}. [{title}]: {snippet}\nURL: {url}\n")

            return "\n".join(formatted_results)

        except Exception as exc:
            print(f"[WebSearch] Search failed: {exc}")
            logger.error("DuckDuckGo web search encountered an exception: %s", exc)
            return "No search results found."
