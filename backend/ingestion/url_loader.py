"""
URL content loader using requests + BeautifulSoup.

Fetches a web page, strips boilerplate elements (nav, footer, scripts),
and returns the cleaned body text.
"""

import logging
from typing import List

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Tags to strip before text extraction (boilerplate / non-content)
_STRIP_TAGS = ["script", "style", "nav", "footer", "header"]

# Request timeout in seconds
_TIMEOUT = 10


def load_url(url: str) -> List[str]:
    """Fetch and extract clean text from a web page.

    Args:
        url: The URL to fetch. Must start with http:// or https://.

    Returns:
        A list containing a single string of the cleaned page text.

    Raises:
        ValueError: On invalid URL, connection failure, non-200 status,
                     non-HTML content type, or empty extracted text.
    """
    # ── Validate URL ───────────────────────────────────────────────────
    if not url or not url.strip():
        raise ValueError("URL cannot be empty")

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError(
            f"Invalid URL scheme. Must start with http:// or https://: {url}"
        )

    # ── Fetch page ─────────────────────────────────────────────────────
    try:
        response = requests.get(
            url,
            timeout=_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (AgenticRAG/1.0)"},
        )
    except requests.exceptions.Timeout:
        raise ValueError(f"Request timed out after {_TIMEOUT}s: {url}")
    except requests.exceptions.ConnectionError as exc:
        raise ValueError(f"Connection failed for URL: {url} — {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise ValueError(f"Failed to fetch URL: {url} — {exc}") from exc

    # ── Check status ───────────────────────────────────────────────────
    if response.status_code != 200:
        raise ValueError(
            f"Non-200 status code ({response.status_code}) for URL: {url}"
        )

    # ── Validate content type ──────────────────────────────────────────
    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        raise ValueError(
            f"Non-HTML content type ({content_type}) for URL: {url}"
        )

    # ── Parse and clean HTML ───────────────────────────────────────────
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove boilerplate elements
    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Extract visible text
    text = soup.get_text(separator="\n", strip=True)

    if not text or not text.strip():
        raise ValueError(f"No text content extracted from URL: {url}")

    cleaned = text.strip()
    logger.info("Extracted %d characters from %s", len(cleaned), url)

    return [cleaned]
