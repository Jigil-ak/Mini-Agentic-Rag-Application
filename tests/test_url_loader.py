"""
Tests for the URL loader.

Uses unittest.mock.patch on requests.get to test HTML extraction,
error handling, and tag stripping without making real HTTP requests.
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.ingestion.url_loader import load_url


def _mock_response(
    status_code: int = 200,
    text: str = "",
    content_type: str = "text/html; charset=utf-8",
):
    """Create a mock requests.Response object."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.text = text
    mock.headers = {"Content-Type": content_type}
    return mock


class TestSuccessfulExtraction:
    """Test successful text extraction from HTML."""

    @patch("backend.ingestion.url_loader.requests.get")
    def test_valid_html_returns_text(self, mock_get):
        html = """
        <html>
        <body>
            <article>
                <h1>Test Article</h1>
                <p>This is a test paragraph with important content.</p>
                <p>Second paragraph with more details.</p>
            </article>
        </body>
        </html>
        """
        mock_get.return_value = _mock_response(text=html)

        result = load_url("https://example.com/article")

        assert isinstance(result, list)
        assert len(result) == 1
        assert "Test Article" in result[0]
        assert "important content" in result[0]

    @patch("backend.ingestion.url_loader.requests.get")
    def test_plain_text_content_type(self, mock_get):
        mock_get.return_value = _mock_response(
            text="Plain text content here",
            content_type="text/plain",
        )

        result = load_url("https://example.com/data.txt")

        assert len(result) == 1
        assert "Plain text content" in result[0]


class TestTagStripping:
    """Test that boilerplate tags are stripped from output."""

    @patch("backend.ingestion.url_loader.requests.get")
    def test_script_tags_stripped(self, mock_get):
        html = """
        <html>
        <body>
            <p>Visible content</p>
            <script>var x = 'should not appear';</script>
        </body>
        </html>
        """
        mock_get.return_value = _mock_response(text=html)

        result = load_url("https://example.com")

        assert "Visible content" in result[0]
        assert "should not appear" not in result[0]

    @patch("backend.ingestion.url_loader.requests.get")
    def test_style_tags_stripped(self, mock_get):
        html = """
        <html>
        <head><style>.hidden { display: none; }</style></head>
        <body><p>Real content</p></body>
        </html>
        """
        mock_get.return_value = _mock_response(text=html)

        result = load_url("https://example.com")

        assert "Real content" in result[0]
        assert "display: none" not in result[0]

    @patch("backend.ingestion.url_loader.requests.get")
    def test_nav_footer_header_stripped(self, mock_get):
        html = """
        <html>
        <body>
            <header>Site Header Navigation</header>
            <nav>Menu Item 1 | Menu Item 2</nav>
            <main><p>Main article content here</p></main>
            <footer>Copyright 2024</footer>
        </body>
        </html>
        """
        mock_get.return_value = _mock_response(text=html)

        result = load_url("https://example.com")

        assert "Main article content here" in result[0]
        assert "Site Header Navigation" not in result[0]
        assert "Menu Item 1" not in result[0]
        assert "Copyright 2024" not in result[0]


class TestErrorHandling:
    """Test error cases raise ValueError."""

    @patch("backend.ingestion.url_loader.requests.get")
    def test_non_200_status_raises(self, mock_get):
        mock_get.return_value = _mock_response(status_code=404, text="Not Found")

        with pytest.raises(ValueError, match="Non-200 status code"):
            load_url("https://example.com/missing")

    @patch("backend.ingestion.url_loader.requests.get")
    def test_timeout_raises(self, mock_get):
        import requests as real_requests
        mock_get.side_effect = real_requests.exceptions.Timeout("Connection timed out")

        with pytest.raises(ValueError, match="timed out"):
            load_url("https://example.com/slow")

    @patch("backend.ingestion.url_loader.requests.get")
    def test_empty_body_raises(self, mock_get):
        mock_get.return_value = _mock_response(text="<html><body></body></html>")

        with pytest.raises(ValueError, match="No text content"):
            load_url("https://example.com/empty")

    def test_empty_url_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            load_url("")

    def test_invalid_scheme_raises(self):
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            load_url("ftp://example.com")

    @patch("backend.ingestion.url_loader.requests.get")
    def test_non_html_content_type_raises(self, mock_get):
        mock_get.return_value = _mock_response(
            text="binary data",
            content_type="application/pdf",
        )

        with pytest.raises(ValueError, match="Non-HTML content type"):
            load_url("https://example.com/file.pdf")

    @patch("backend.ingestion.url_loader.requests.get")
    def test_connection_error_raises(self, mock_get):
        import requests as real_requests
        mock_get.side_effect = real_requests.exceptions.ConnectionError("DNS failed")

        with pytest.raises(ValueError, match="Connection failed"):
            load_url("https://nonexistent.example.com")
