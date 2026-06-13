"""
Tests for the RouterAgent.

All external dependencies (Milvus, embeddings) are mocked to test
pure routing logic: similarity thresholds, empty collections, and
edge cases.
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.agent.router import RouterAgent


@pytest.fixture
def router():
    """Create a fresh RouterAgent instance for each test."""
    return RouterAgent()


class TestRAGRouting:
    """Test that high-similarity queries route to RAG."""

    @patch("backend.agent.router.get_document_count", return_value=10)
    @patch("backend.agent.router.collection_exists", return_value=True)
    @patch("backend.agent.router.search_documents")
    @patch("backend.agent.router.embed_query")
    def test_high_similarity_routes_to_rag(
        self, mock_embed, mock_search, mock_exists, mock_count, router
    ):
        mock_embed.return_value = [0.1] * 384
        mock_search.return_value = [
            {"text": "Relevant document text here", "source": "test.pdf",
             "distance": 0.15, "similarity": 0.85}
        ]

        result = router.route("What is RAG?")

        assert result["path"] == "rag"
        assert result["similarity_score"] == 0.85
        assert "relevant context" in result["reason"].lower() or "similarity" in result["reason"].lower()
        assert len(result["top_chunk_preview"]) > 0

    @patch("backend.agent.router.get_document_count", return_value=10)
    @patch("backend.agent.router.collection_exists", return_value=True)
    @patch("backend.agent.router.search_documents")
    @patch("backend.agent.router.embed_query")
    def test_exact_threshold_routes_to_rag(
        self, mock_embed, mock_search, mock_exists, mock_count, router
    ):
        mock_embed.return_value = [0.1] * 384
        mock_search.return_value = [
            {"text": "Threshold test", "source": "test.pdf",
             "distance": 0.40, "similarity": 0.60}
        ]

        result = router.route("test query")

        assert result["path"] == "rag"
        assert result["similarity_score"] == 0.60


class TestToolRouting:
    """Test that low-similarity queries route to tools."""

    @patch("backend.agent.router.get_document_count", return_value=10)
    @patch("backend.agent.router.collection_exists", return_value=True)
    @patch("backend.agent.router.search_documents")
    @patch("backend.agent.router.embed_query")
    def test_low_similarity_routes_to_tool(
        self, mock_embed, mock_search, mock_exists, mock_count, router
    ):
        mock_embed.return_value = [0.1] * 384
        mock_search.return_value = [
            {"text": "Irrelevant content", "source": "test.pdf",
             "distance": 0.70, "similarity": 0.30}
        ]

        result = router.route("What is the weather today?")

        assert result["path"] == "tool"
        assert result["similarity_score"] == 0.30
        assert "no relevant context" in result["reason"].lower() or "similarity" in result["reason"].lower()

    @patch("backend.agent.router.get_document_count", return_value=10)
    @patch("backend.agent.router.collection_exists", return_value=True)
    @patch("backend.agent.router.search_documents")
    @patch("backend.agent.router.embed_query")
    def test_empty_search_results_routes_to_tool(
        self, mock_embed, mock_search, mock_exists, mock_count, router
    ):
        mock_embed.return_value = [0.1] * 384
        mock_search.return_value = []

        result = router.route("something random")

        assert result["path"] == "tool"
        assert result["similarity_score"] == 0.0


class TestEmptyKnowledgeBase:
    """Test behaviour when the knowledge base has no documents."""

    @patch("backend.agent.router.get_document_count", return_value=0)
    @patch("backend.agent.router.collection_exists", return_value=True)
    def test_empty_collection_routes_to_tool(self, mock_exists, mock_count, router):
        result = router.route("any question")

        assert result["path"] == "tool"
        assert result["similarity_score"] == 0.0
        assert "empty" in result["reason"].lower()
        assert result["top_chunk_preview"] == ""

    @patch("backend.agent.router.collection_exists", return_value=False)
    def test_no_collection_routes_to_tool(self, mock_exists, router):
        result = router.route("any question")

        assert result["path"] == "tool"
        assert result["similarity_score"] == 0.0
        assert "empty" in result["reason"].lower()


class TestRouterOutputValidation:
    """Test that router output always has valid structure."""

    @patch("backend.agent.router.get_document_count", return_value=10)
    @patch("backend.agent.router.collection_exists", return_value=True)
    @patch("backend.agent.router.search_documents")
    @patch("backend.agent.router.embed_query")
    def test_routing_reason_is_nonempty(
        self, mock_embed, mock_search, mock_exists, mock_count, router
    ):
        mock_embed.return_value = [0.1] * 384
        mock_search.return_value = [
            {"text": "test", "source": "s", "distance": 0.5, "similarity": 0.5}
        ]

        result = router.route("test")
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0

    @patch("backend.agent.router.get_document_count", return_value=10)
    @patch("backend.agent.router.collection_exists", return_value=True)
    @patch("backend.agent.router.search_documents")
    @patch("backend.agent.router.embed_query")
    def test_similarity_score_in_valid_range(
        self, mock_embed, mock_search, mock_exists, mock_count, router
    ):
        mock_embed.return_value = [0.1] * 384
        mock_search.return_value = [
            {"text": "test", "source": "s", "distance": 0.25, "similarity": 0.75}
        ]

        result = router.route("test")
        assert 0.0 <= result["similarity_score"] <= 1.0

    @patch("backend.agent.router.get_document_count", return_value=0)
    @patch("backend.agent.router.collection_exists", return_value=True)
    def test_output_has_all_required_keys(self, mock_exists, mock_count, router):
        result = router.route("test")
        assert "path" in result
        assert "similarity_score" in result
        assert "reason" in result
        assert "top_chunk_preview" in result
