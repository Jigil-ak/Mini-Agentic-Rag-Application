"""
Router agent — decides whether to use RAG or tools for each query.

Embeds the query, searches Milvus for similar documents, and compares
the top similarity score against the configured threshold to determine
the execution path.

Routing logic:
    similarity >= SIMILARITY_THRESHOLD → RAG path (use knowledge base)
    similarity <  SIMILARITY_THRESHOLD → Tool path (web search / calculator)
    empty collection                   → Tool path (no knowledge to search)
    Milvus unavailable                 → Tool path (graceful degradation)
"""

import logging
from typing import Dict

from backend.config import SIMILARITY_THRESHOLD, TOP_K
from backend.ingestion.embedding import embed_query
from backend.vectordb.milvus_client import (
    search_documents,
    collection_exists,
    get_document_count,
    is_available,
)

logger = logging.getLogger(__name__)


class RouterAgent:
    """Routes user queries to the appropriate agent based on retrieval similarity.

    The router is stateless — all configuration comes from backend.config.
    This class never raises exceptions; it always returns a valid routing dict.
    """

    def __init__(self) -> None:
        self._threshold = SIMILARITY_THRESHOLD

    def route(self, query: str) -> Dict:
        """Analyse the query against the knowledge base and decide the path.

        This method is designed to NEVER raise an exception. If Milvus is
        unavailable, the collection is empty, or any error occurs during
        retrieval, it gracefully falls back to the tool path.

        Args:
            query: The user's question.

        Returns:
            Dict with keys:
                path (str): "rag" or "tool"
                similarity_score (float): Normalised 0–1 score (1.0 = identical)
                reason (str): Human-readable explanation of routing decision
                top_chunk_preview (str): First 100 chars of top chunk, or ""
        """
        # ── Handle Milvus unavailable ──────────────────────────────────
        try:
            milvus_ready = is_available()
        except Exception:
            milvus_ready = False

        if not milvus_ready:
            logger.info("Router: Milvus is not available — routing to tool")
            return {
                "path": "tool",
                "similarity_score": 0.0,
                "reason": "Knowledge base is empty, routing to external tools.",
                "top_chunk_preview": "",
            }

        # ── Handle empty knowledge base ────────────────────────────────
        try:
            has_collection = collection_exists()
            doc_count = get_document_count() if has_collection else 0
        except Exception as exc:
            logger.warning("Router: failed to check collection status: %s", exc)
            return {
                "path": "tool",
                "similarity_score": 0.0,
                "reason": "Knowledge base is empty, routing to external tools.",
                "top_chunk_preview": "",
            }

        if not has_collection or doc_count == 0:
            logger.info("Router: knowledge base is empty — routing to tool")
            return {
                "path": "tool",
                "similarity_score": 0.0,
                "reason": "Knowledge base is empty, routing to external tools.",
                "top_chunk_preview": "",
            }

        # ── Embed query and search ─────────────────────────────────────
        try:
            query_embedding = embed_query(query)
            results = search_documents(query_embedding, top_k=1)
        except Exception as exc:
            logger.warning("Router: search failed: %s — routing to tool", exc)
            return {
                "path": "tool",
                "similarity_score": 0.0,
                "reason": f"Search failed ({exc}), routing to external tools.",
                "top_chunk_preview": "",
            }

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
