"""
FastAPI routes for the Agentic RAG application.
"""

import logging
import os
import tempfile
import traceback
from typing import Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.agent.agent import RAGAgent
from backend.ingestion.pdf_loader import load_pdf
from backend.ingestion.url_loader import load_url
from backend.ingestion.chunking import chunk_texts
from backend.ingestion.embedding import embed_texts
from backend.vectordb.milvus_client import (
    ensure_collection,
    insert_documents,
    get_document_count,
    collection_exists,
    is_available,
    reset_collection,
)
from backend.tracing.tracer import TraceManager

logger = logging.getLogger(__name__)

# ── Pydantic Models ────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request schema for query processing."""
    query: str = Field(..., min_length=1, description="The user's query text")
    source: Optional[str] = Field(None, description="Optional document source to restrict search")


class QueryResponse(BaseModel):
    """Response schema for query processing."""
    answer: str
    trace: Dict


class LoadResponse(BaseModel):
    """Response schema for data ingestion."""
    status: str
    chunks_loaded: int
    source: str


class HealthResponse(BaseModel):
    """Response schema for system health check."""
    status: str
    milvus: str
    documents_indexed: int


# ── Router and Shared State ────────────────────────────────────────────────

router = APIRouter()
trace_manager = TraceManager()


# ── POST /load ─────────────────────────────────────────────────────────────

@router.post("/load", response_model=LoadResponse)
async def load_documents(
    file: UploadFile = File(None),
    url: str = Form(None),
):
    """Ingest a document (PDF upload or URL) into the knowledge base."""
    # 1. Validate inputs
    if not file and (not url or not url.strip()):
        raise HTTPException(
            status_code=400,
            detail="Provide either a PDF file upload or a URL",
        )

    # Make sure collection exists
    try:
        ensure_collection()
    except Exception as exc:
        logger.error("[Load] Failed to ensure collection exists: %s", exc)
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {exc}")

    # 2. If file:
    if file:
        print(f"[Load] File received: {file.filename}, type: {file.content_type}")
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            print(f"[Load] Saved temp file: {tmp_path}, size: {os.path.getsize(tmp_path)} bytes")

            pages = load_pdf(tmp_path)
            print(f"[Load] Extracted {len(pages)} pages")

            chunks = chunk_texts(pages)
            print(f"[Load] Generated {len(chunks)} chunks")

            embeddings = embed_texts(chunks)
            print(f"[Load] Generated {len(embeddings)} embeddings")

            inserted = insert_documents(chunks, embeddings, source=file.filename)
            print(f"[Load] Inserted {inserted} documents. Total: {get_document_count()}")

            return LoadResponse(status="success", chunks_loaded=inserted, source=file.filename)

        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"PDF processing failed: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
                print("[Load] Temp file cleaned up")

    # 3. If url:
    else:
        url = url.strip()
        print(f"[Load] URL received: {url}")
        try:
            pages = load_url(url)
            chunks = chunk_texts(pages)
            embeddings = embed_texts(chunks)
            inserted = insert_documents(chunks, embeddings, source=url)
            return LoadResponse(status="success", chunks_loaded=inserted, source=url)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"URL processing failed: {e}")


# ── POST /query ────────────────────────────────────────────────────────────


def _get_routing_type(selected_tools: List[str]) -> str:
    """Derive a categorical routing type from the selected tools list."""
    if len(selected_tools) > 1:
        return "multi_tool"
    return selected_tools[0] if selected_tools else "unknown"


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process query using the single-agent architecture with tracing."""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_text = request.query.strip()
    trace_id = trace_manager.start_trace(query_text)

    try:
        agent = RAGAgent()
        result = agent.run(query_text, source=request.source)

        trace_manager.update_trace(
            trace_id,
            # New fields
            selected_tools=result["selected_tools"],
            tool_reasoning=result["tool_reasoning"],
            routing_decision_timestamp=result["routing_decision_timestamp"],
            routing_method=result["routing_method"],
            routing_type=_get_routing_type(result["selected_tools"]),
            tool_execution_results={
                k: str(v)[:500]
                for k, v in result["tool_execution_results"].items()
            },
            tool_execution_status=result["tool_execution_status"],
            # Backward-compatible existing fields
            tool_used=result["selected_tools"][0] if result["selected_tools"] else None,
            path_taken="+".join(result["selected_tools"]),
            retrieval_hit=result["retrieval_hit"],
            similarity_score=result.get("top_similarity_score", 0.0),
            primary_model=result["model_used"],
            fallback_triggered=result["fallback_triggered"],
            chunks_used=result.get("chunks_used", 0),
            # New diagnostic fields
            gemini_status=result.get("gemini_status", "not_attempted"),
            knowledge_base_attempted=result.get("knowledge_base_attempted", False),
            knowledge_base_selected=result.get("knowledge_base_selected", False),
            knowledge_base_results_found=result.get("knowledge_base_results_found", 0),
            web_search_results_count=result.get("web_search_results_count", 0),
            tool_failure_reason=result.get("tool_failure_reason"),
            routing_explanation=result.get("routing_explanation", ""),
            documents_indexed_at_query_time=result.get("documents_indexed_at_query_time", 0),
            document_relevance_score=result.get("document_relevance_score", 0.0),
            # Phase 2 Evidence diagnostics
            top_similarity_score=result.get("top_similarity_score", 0.0),
            top_chunk_source=result.get("top_chunk_source", "N/A"),
            top_chunk_preview=result.get("top_chunk_preview", "N/A"),
            source_filter_used=result.get("source_filter_used"),
            retrieved_chunk_count=result.get("retrieved_chunk_count", 0),
        )

        trace = trace_manager.finish_trace(trace_id)
        return QueryResponse(answer=result["answer"], trace=trace)

    except Exception as exc:
        logger.error("Query API failed: %s", exc, exc_info=True)
        traceback.print_exc()
        try:
            trace_manager.update_trace(trace_id, error=str(exc))
            trace_manager.finish_trace(trace_id)
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(exc)}",
        )


# ── POST /reset ────────────────────────────────────────────────────────────

@router.post("/reset")
async def reset_database():
    """Clear the vector database."""
    try:
        reset_collection()
        return {"status": "success", "message": "Vector database cleared"}
    except Exception as exc:
        logger.error("Failed to reset database: %s", exc)
        raise HTTPException(status_code=500, detail=f"Reset failed: {exc}")


# ── GET /health ────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_endpoint():
    """Health check for the application components."""
    try:
        available = is_available()
        count = get_document_count() if available else 0
        milvus_status = "connected" if available else "unavailable"
    except Exception as exc:
        logger.error("Health check database check failed: %s", exc)
        milvus_status = "error"
        count = 0

    return HealthResponse(
        status="ok",
        milvus=milvus_status,
        documents_indexed=count,
    )


# ── GET /traces ────────────────────────────────────────────────────────────

@router.get("/traces")
async def traces_endpoint():
    """Retrieve the recent trace execution logs."""
    return {"traces": trace_manager.get_recent_traces(20)}


# ── GET /debug/status ──────────────────────────────────────────────────────

@router.get("/debug/status")
async def debug_status():
    """Retrieve detailed status of system database and configuration."""
    try:
        milvus_ready = is_available()
        doc_count = get_document_count() if milvus_ready else 0
        exists = collection_exists() if milvus_ready else False
    except Exception as exc:
        logger.error("Debug status lookup failed: %s", exc)
        milvus_ready = False
        doc_count = 0
        exists = False

    from backend.config import GEMINI_API_KEY, GROQ_API_KEY, COLLECTION_NAME, SIMILARITY_THRESHOLD
    return {
        "milvus": {
            "available": milvus_ready,
            "collection_exists": exists,
            "collection_name": COLLECTION_NAME,
            "document_count": doc_count,
        },
        "llm_api_keys": {
            "gemini_api_key_configured": bool(GEMINI_API_KEY),
            "groq_api_key_configured": bool(GROQ_API_KEY),
        },
        "config": {
            "similarity_threshold": SIMILARITY_THRESHOLD,
        },
    }
