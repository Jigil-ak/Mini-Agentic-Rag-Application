"""
RAG agent — retrieval-augmented generation pipeline.

Embeds the user query, retrieves relevant chunks from Milvus,
builds a context-augmented prompt, and generates an answer via
the fallback-aware LLM orchestrator.
"""

import logging
from typing import Dict, List

from backend.config import TOP_K
from backend.ingestion.embedding import embed_query
from backend.vectordb.milvus_client import search_documents
from backend.agent.fallback import call_with_fallback

logger = logging.getLogger(__name__)

# ── System prompt template ─────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

INSTRUCTIONS:
- Answer the question using ONLY the information in the context below.
- If the context does not contain enough information to answer, say so clearly.
- Be concise and accurate.
- Cite the source when possible.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""


def _build_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a numbered context block.

    Args:
        chunks: List of dicts with 'text' and 'source' keys.

    Returns:
        A numbered, formatted context string.
    """
    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        lines.append(f"[{idx}] (Source: {source})\n{text}")
    return "\n\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────

def run_rag(query: str) -> Dict:
    """Execute the RAG pipeline: retrieve → build prompt → generate.

    Args:
        query: The user's question.

    Returns:
        Dict with keys:
            answer (str): The generated answer.
            chunks_used (int): Number of context chunks used.
            model_used (str): Which LLM produced the answer.
            fallback_triggered (bool): Whether the fallback LLM was used.
            sources (List[str]): Unique source identifiers from chunks.
    """
    logger.info("RAG agent: processing query '%s'", query[:80])

    # 1. Embed the query
    query_embedding = embed_query(query)

    # 2. Retrieve top-K chunks from Milvus
    results = search_documents(query_embedding, top_k=TOP_K)

    if not results:
        logger.warning("RAG agent: no chunks retrieved — generating without context")
        context_str = "No relevant context was found in the knowledge base."
        sources = []
        chunks_used = 0
    else:
        context_str = _build_context(results)
        sources = list({r.get("source", "unknown") for r in results})
        chunks_used = len(results)
        logger.info("RAG agent: using %d chunks from sources: %s", chunks_used, sources)

    # 3. Build the full prompt
    prompt = _SYSTEM_PROMPT.format(context=context_str, query=query)

    # 4. Generate answer with fallback support
    llm_result = call_with_fallback(prompt)

    return {
        "answer": llm_result["response"],
        "chunks_used": chunks_used,
        "model_used": llm_result["model_used"],
        "fallback_triggered": llm_result["fallback_triggered"],
        "sources": sources,
    }
