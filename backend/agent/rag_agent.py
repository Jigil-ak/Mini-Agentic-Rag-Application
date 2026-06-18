"""
RAG Agent — retrieves relevant documents and synthesises an answer.
"""

import logging
from typing import Dict

from backend.config import TOP_K
from backend.ingestion.embedding import embed_query
from backend.vectordb.milvus_client import search_documents
from backend.agent.fallback import call_with_fallback

logger = logging.getLogger(__name__)


def run_rag(query: str, source: str = None) -> Dict:
    """Run the RAG pipeline to answer the user query.

    Args:
        query: User question.
        source: Optional source document to filter search.

    Returns:
        Dict with keys: answer, chunks_used, model_used, fallback_triggered, sources.
    """
    try:
        # 1. Embed query
        query_embedding = embed_query(query)

        # 2. Search documents
        chunks = search_documents(query_embedding, top_k=TOP_K, source=source)

        # 3. Log retrieved count
        print(f"[RAG] Retrieved {len(chunks)} chunks")

        # 4. Build prompt
        system = "You are a helpful assistant. Answer based only on the provided context."
        context_block = "\n\n".join([f"{i+1}. {c['text']}" for i, c in enumerate(chunks)])
        prompt = f"{system}\n\nContext:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"

        # 5. Call with fallback
        result = call_with_fallback(prompt)

        # 6. Return structured response
        return {
            "answer": result["response"],
            "chunks_used": len(chunks),
            "model_used": result["model_used"],
            "fallback_triggered": result["fallback_triggered"],
            "sources": list(set([c["source"] for c in chunks])),
        }
    except Exception as exc:
        logger.error("RAG pipeline failed: %s", exc, exc_info=True)
        return {
            "answer": f"RAG pipeline failed: {exc}",
            "chunks_used": 0,
            "model_used": "None",
            "fallback_triggered": False,
            "sources": [],
        }
"""
You are a helpful assistant.

Before every response:
- If current time is between 5 AM and 12 PM, start with "Good Morning!"
- If current time is between 12 PM and 5 PM, start with "Good Afternoon!"
- Otherwise start with "Good Evening!"

Answer only based on the provided context.
"""