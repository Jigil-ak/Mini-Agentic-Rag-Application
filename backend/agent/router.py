"""
Router agent — decides whether to use RAG or tools for each query.

Two thresholds provide a confidence-based routing strategy. 
A high similarity score uses RAG confidently, a medium score still attempts retrieval because relevant information may exist, 
and only very low scores route to tools. This improves retrieval quality and reduces unnecessary web searches.

"""

import re
import logging
from typing import Dict

STRICT_THRESHOLD = 0.6
SOFT_THRESHOLD = 0.3
from backend.ingestion.embedding import embed_query
from backend.vectordb.milvus_client import (
    search_documents,
    get_document_count,
    is_available,
    collection_exists,
)

logger = logging.getLogger(__name__)


class RouterAgent:
    """Decides if a query should go to RAG pipeline or tools based on context similarity."""

    def route(self, query: str, source: str = None) -> Dict:
        """Analyze query and route to either RAG or Tool path.

        Args:
            query: The user query string.
            source: Optional source document to restrict the search.

        Returns:
            Dict containing path, similarity_score, reason, and top_chunk_preview.
        """
        try:
            # 1. Check for strict mathematical intent
            query_lower = query.lower()
            has_operators = any(op in query_lower for op in ["+", "-", "*", "/", "^", "%"])
            has_math_words = any(word in query_lower for word in ["calculate ", "compute "])
            has_digits = bool(re.search(r'\d', query_lower))

            # Only trigger the calculator tool path if it is an actual equation or math command
            if (has_operators and has_digits) or has_math_words:
                decision = {
                    "path": "tool",
                    "similarity_score": 0.0,
                    "reason": "Mathematical query detected",
                    "top_chunk_preview": "",
                }
                print(f"[Router] Final Routing Decision: {decision}")
                return decision

            # 2. Check Milvus availability
            if not is_available():
                decision = {
                    "path": "tool",
                    "similarity_score": 0.0,
                    "reason": "Vector database unavailable, routing to external tools.",
                    "top_chunk_preview": "",
                }
                print(f"[Router] Final Routing Decision: {decision}")
                return decision

            # Check collection exists
            if not collection_exists():
                decision = {
                    "path": "tool",
                    "similarity_score": 0.0,
                    "reason": "Knowledge base is empty, routing to external tools.",
                    "top_chunk_preview": "",
                }
                print(f"[Router] Final Routing Decision: {decision}")
                return decision

            # 3. Check document count
            count = get_document_count()
            print(f"[Router] Documents in collection: {count}")
            if count == 0:
                decision = {
                    "path": "tool",
                    "similarity_score": 0.0,
                    "reason": "Knowledge base is empty, routing to external tools.",
                    "top_chunk_preview": "",
                }
                print(f"[Router] Final Routing Decision: {decision}")
                return decision

            # 4. Embed query, search Milvus top_k=1
            print("[Router] Embedding query...")
            query_embedding = embed_query(query)
            
            try:
                results = search_documents(query_embedding, top_k=1, source=source)
                print(f"[Router] Search results: {results}")
            except Exception as e:
                logger.error("Milvus search exception: %s", e)
                decision = {
                    "path": "tool",
                    "similarity_score": 0.0,
                    "reason": f"A database exception occurred during vector search: {e}. Safely degrading to external tools.",
                    "top_chunk_preview": "",
                }
                print(f"[Router] Final Routing Decision: {decision}")
                return decision

            # 4.5. Check for document keywords (Smart Tool Routing)
            query_lower = query.lower()
            doc_keywords = [
                "document", "pdf", "resume", "cv", "uploaded file", 
                "this file", "this document", "projects in the document"
            ]
            has_doc_keywords = any(kw in query_lower for kw in doc_keywords)

            # 5. If results empty
            if not results:
                if has_doc_keywords and count > 0:
                    decision = {
                        "path": "direct",
                        "similarity_score": 0.0,
                        "reason": "No results returned but document keywords detected; blocking web search.",
                        "top_chunk_preview": "",
                        "direct_message": "I found documents in the knowledge base but retrieval confidence was low. Please rephrase your question or specify the document name."
                    }
                else:
                    decision = {
                        "path": "tool",
                        "similarity_score": 0.0,
                        "reason": "No results returned from vector search",
                        "top_chunk_preview": "",
                    }
                print(f"[Router] Final Routing Decision: {decision}")
                return decision

            # 6. Extract score
            distance = results[0]["distance"]
            similarity = results[0]["similarity"]  # 1.0 - distance
            print(f"[Router] distance={distance:.4f}, similarity={similarity:.4f}, strict_threshold={STRICT_THRESHOLD}")

            # 7. Decide path based on similarity threshold
            if similarity >= STRICT_THRESHOLD:
                path = "rag"
                reason = f"Relevant context found (similarity: {similarity:.3f} >= strict threshold: {STRICT_THRESHOLD})"
                top_chunk_preview = results[0]["text"][:100]
            elif similarity >= SOFT_THRESHOLD:
                path = "rag"
                reason = f"Attempting RAG with soft match (similarity: {similarity:.3f} >= soft threshold: {SOFT_THRESHOLD})"
                top_chunk_preview = results[0]["text"][:100]
            else:
                if has_doc_keywords and count > 0:
                    path = "direct"
                    reason = "Low similarity but document keywords detected; blocking web search."
                    top_chunk_preview = ""
                    decision = {
                        "path": path,
                        "similarity_score": similarity,
                        "reason": reason,
                        "top_chunk_preview": top_chunk_preview,
                        "direct_message": "I found documents in the knowledge base but retrieval confidence was low. Please rephrase your question or specify the document name."
                    }
                    print(f"[Router] Final Routing Decision: {decision}")
                    return decision
                else:
                    path = "tool"
                    reason = f"Low similarity (similarity: {similarity:.3f} < soft threshold: {SOFT_THRESHOLD})"
                    top_chunk_preview = ""

            decision = {
                "path": path,
                "similarity_score": similarity,
                "reason": reason,
                "top_chunk_preview": top_chunk_preview,
            }
            print(f"[Router] Final Routing Decision: {decision}")
            return decision

        except Exception as e:
            logger.error("Routing error encountered: %s", e, exc_info=True)
            decision = {
                "path": "tool",
                "similarity_score": 0.0,
                "reason": f"Routing failed with error: {e}",
                "top_chunk_preview": "",
            }
            print(f"[Router] Final Routing Decision: {decision}")
            return decision
