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

from backend.agent.router import RouterAgent
from backend.agent.rag_agent import run_rag
from backend.agent.tool_agent import run_tool
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

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process query using RAG or Tool agent with tracing."""
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    trace_id = trace_manager.start_trace(query_text)

    try:
        routing = RouterAgent().route(query_text, request.source)

        trace_manager.update_trace(
            trace_id,
            retrieval_hit=(routing["path"] == "rag"),
            similarity_score=routing["similarity_score"],
            path_taken=routing["path"],
            routing_reason=routing["reason"],
        )

        if routing["path"] == "rag":
            agent_result = run_rag(query_text, request.source)
            tool_used = None
        elif routing["path"] == "direct":
            agent_result = {
                "answer": routing.get("direct_message", "Request blocked."),
                "model_used": "system",
                "fallback_triggered": False,
                "chunks_used": 0,
            }
            tool_used = None
        else:
            agent_result = run_tool(query_text)
            tool_used = agent_result.get("tool_used")

        trace_manager.update_trace(
            trace_id,
            primary_model=agent_result["model_used"],
            fallback_triggered=agent_result["fallback_triggered"],
            tool_used=tool_used,
            chunks_used=agent_result.get("chunks_used"),
        )

        trace = trace_manager.finish_trace(trace_id)
        return QueryResponse(answer=agent_result["answer"], trace=trace)

    except Exception as exc:
        logger.error("Query API failed: %s", exc, exc_info=True)
        try:
            trace_manager.update_trace(trace_id, error=str(exc))
            trace_manager.finish_trace(trace_id)
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {exc}",
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
