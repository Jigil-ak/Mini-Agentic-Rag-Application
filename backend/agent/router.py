"""
Router agent — decides whether to use RAG or tools for each query.

Embeds the query, searches Milvus for similar documents, and compares
the top similarity score against the configured threshold to determine
the execution path.

Routing logic:
    similarity >= SIMILARITY_THRESHOLD → RAG path (use knowledge base)
    similarity <  SIMILARITY_THRESHOLD → Tool path (web search / calculator)
    empty collection                   → Tool path (no knowledge to search)
"""

import logging
from typing import Dict

from backend.config import SIMILARITY_THRESHOLD, TOP_K
from backend.ingestion.embedding import embed_query
from backend.vectordb.milvus_client import search_documents, collection_exists, get_document_count

logger = logging.getLogger(__name__)


class RouterAgent:
    """Routes user queries to the appropriate agent based on retrieval similarity.

    The router is stateless — all configuration comes from backend.config.
    """

    def __init__(self) -> None:
        self._threshold = SIMILARITY_THRESHOLD

    def route(self, query: str) -> Dict:
        """Analyse the query against the knowledge base and decide the path.

        Args:
            query: The user's question.

        Returns:
            Dict with keys:
                path (str): "rag" or "tool"
                similarity_score (float): Normalised 0–1 score (1.0 = identical)
                reason (str): Human-readable explanation of routing decision
                top_chunk_preview (str): First 100 chars of top chunk, or ""
        """
        # ── Handle empty knowledge base ────────────────────────────────
        if not collection_exists() or get_document_count() == 0:
            logger.info("Router: knowledge base is empty — routing to tool")
            return {
                "path": "tool",
                "similarity_score": 0.0,
                "reason": "Knowledge base is empty",
                "top_chunk_preview": "",
            }

        # ── Embed query and search ─────────────────────────────────────
        query_embedding = embed_query(query)
        results = search_documents(query_embedding, top_k=1)

        # No results from search
        if not results:
            logger.info("Router: search returned no results — routing to tool")
            return {
                "path": "tool",
                "similarity_score": 0.0,
                "reason": f"No relevant context found (no search results, threshold: {self._threshold})",
                "top_chunk_preview": "",
            }

        # ── Extract similarity score ───────────────────────────────────
        top_result = results[0]
        similarity = top_result.get("similarity", 0.0)
        top_text = top_result.get("text", "")
        preview = top_text[:100] if top_text else ""

        # ── Make routing decision ──────────────────────────────────────
        if similarity >= self._threshold:
            path = "rag"
            reason = (
                f"Found relevant context "
                f"(similarity: {similarity:.3f} >= threshold: {self._threshold})"
            )
            logger.info("Router: %s — routing to RAG", reason)
        else:
            path = "tool"
            reason = (
                f"No relevant context found "
                f"(similarity: {similarity:.3f} < threshold: {self._threshold})"
            )
            logger.info("Router: %s — routing to tool", reason)
            preview = ""  # Don't expose irrelevant chunks

        return {
            "path": path,
            "similarity_score": similarity,
            "reason": reason,
            "top_chunk_preview": preview,
        }
