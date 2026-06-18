"""
Web search tool using DuckDuckGo via the ddgs package.
"""

import logging
import time
from ddgs import DDGS

logger = logging.getLogger(__name__)


def _strip_non_ascii(text: str) -> str:
    """Helper to remove non-ASCII characters from search result fields."""
    if not text:
        return ""
    return "".join(c for c in text if ord(c) < 128)


class WebSearchTool:
    """Web search tool class using the DDGS client directly."""

    def __init__(self):
        self.last_results_count = 0

    def search(self, query: str, max_results: int = 3) -> str:
        """Perform a web search and format results as a numbered list.

        Args:
            query: The search query.
            max_results: Max results to fetch.

        Returns:
            A formatted string list of results or a fallback message if failed.
        """
        print(f"[WebSearch] Searching for: {query}")
        self.last_results_count = 0

        for attempt in range(2):
            try:
                results = []
                with DDGS() as ddgs:
                    ddgs_generator = ddgs.text(query, max_results=max_results)
                    if ddgs_generator:
                        for r in ddgs_generator:
                            results.append(r)

                n = len(results)
                self.last_results_count = n
                print(f"[WebSearch] Found {n} results")

                first_title = results[0].get("title", "") if n > 0 else "N/A"
                first_url = results[0].get("href", "") if n > 0 else "N/A"
                first_snippet = results[0].get("body", "")[:100] if n > 0 else "N/A"

                logger.info(
                    "Web Search Stage - query='%s', results_count=%d, first_result_title='%s', first_result_url='%s', first_result_preview='%s'",
                    query,
                    n,
                    first_title,
                    first_url,
                    first_snippet,
                )

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
                print(f"[WebSearch] Search attempt {attempt + 1} failed: {exc}")
                logger.warning(
                    "DuckDuckGo search attempt %d failed for query '%s': %s",
                    attempt + 1,
                    query,
                    exc,
                    exc_info=True,
                )
                if attempt == 0:
                    time.sleep(1.0)
                else:
                    self.last_results_count = 0
                    logger.error(
                        "Web Search Stage - query='%s', results_count=0, first_result_title='N/A', first_result_url='N/A', first_result_preview='N/A', error='%s'",
                        query,
                        exc,
                    )
                    return "No search results found due to search engine failure."

