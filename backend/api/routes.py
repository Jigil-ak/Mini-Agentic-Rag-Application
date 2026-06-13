"""
FastAPI routes for the Agentic RAG application.

Endpoints:
    POST /query   — Ask a question, get an agent-generated answer + trace
    POST /load    — Ingest a PDF file or URL into the knowledge base
    GET  /health  — Health check with Milvus status and document count
    GET  /traces  — Retrieve recent trace logs
"""

import logging
import os
import tempfile
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
)
from backend.tracing.tracer import TraceManager

logger = logging.getLogger(__name__)

# ── Pydantic Models ────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""
    query: str = Field(..., min_length=1, description="The user's question")


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""
    answer: str
    trace: Dict


class LoadResponse(BaseModel):
    """Response body for the /load endpoint."""
    status: str
    chunks_loaded: int
    source: str


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str
    milvus: str
    documents_indexed: int


class TraceEntry(BaseModel):
    """A single trace log entry."""
    trace_id: str
    query: str
    timestamp: str
    path_taken: str
    similarity_score: float
    model_used: Optional[str] = None
    fallback_triggered: bool = False
    response_time_ms: float = 0.0


# ── Router and shared state ───────────────────────────────────────────────

router = APIRouter()
_router_agent = RouterAgent()
_trace_manager = TraceManager()


# ── POST /query ────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a user query through the agentic RAG pipeline.

    1. Start a trace
    2. Route the query (RAG vs Tool)
    3. Execute the chosen agent
    4. Finish the trace and return the result

    Never lets raw exceptions bubble up — always returns structured JSON.
    """
    query = request.query.strip()
    trace_id = _trace_manager.start_trace(query)

    try:
        # ── Route the query ────────────────────────────────────────
        routing = _router_agent.route(query)
        path = routing["path"]
        similarity = routing["similarity_score"]

        _trace_manager.update_trace(
            trace_id,
            path_taken=path,
            similarity_score=similarity,
            routing_reason=routing["reason"],
            retrieval_hit=(path == "rag"),
        )

        # ── Execute the appropriate agent ──────────────────────────
        if path == "rag":
            result = run_rag(query)
            _trace_manager.update_trace(
                trace_id,
                primary_model=result["model_used"],
                fallback_triggered=result["fallback_triggered"],
                chunks_used=result["chunks_used"],
            )
            answer = result["answer"]

        else:  # tool path
            result = run_tool(query)
            _trace_manager.update_trace(
                trace_id,
                primary_model=result["model_used"],
                fallback_triggered=result["fallback_triggered"],
                tool_used=result["tool_used"],
                tool_output_preview=result.get("tool_output", "")[:200],
            )
            answer = result["answer"]

        # ── Finalise trace ─────────────────────────────────────────
        trace = _trace_manager.finish_trace(trace_id)
        return QueryResponse(answer=answer, trace=trace)

    except Exception as exc:
        logger.error("Query processing failed: %s", exc, exc_info=True)
        _trace_manager.update_trace(trace_id, error=str(exc))
        try:
            _trace_manager.finish_trace(trace_id)
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {exc}",
        )


# ── POST /load ─────────────────────────────────────────────────────────────

@router.post("/load", response_model=LoadResponse)
async def load_endpoint(
    file: UploadFile = File(None),
    url: str = Form(None),
):
    """Ingest a document (PDF upload or URL) into the knowledge base.

    At least one of `file` or `url` must be provided.
    """
    if file is None and (url is None or not url.strip()):
        raise HTTPException(
            status_code=400,
            detail="Provide either a PDF file upload or a URL",
        )

    # Ensure the Milvus collection exists
    ensure_collection()

    try:
        # ── PDF Upload ─────────────────────────────────────────────
        if file is not None:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail="Only PDF files are supported",
                )

            # Save to temp file, process, then delete
            suffix = ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            try:
                raw_texts = load_pdf(tmp_path)
                source = file.filename
            finally:
                os.unlink(tmp_path)

        # ── URL Ingestion ──────────────────────────────────────────
        else:
            url = url.strip()
            raw_texts = load_url(url)
            source = url

        # ── Chunk → Embed → Store ──────────────────────────────────
        chunks = chunk_texts(raw_texts)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text chunks could be extracted from the source",
            )

        embeddings = embed_texts(chunks)
        count = insert_documents(chunks, embeddings, source)

        logger.info("Loaded %d chunks from source: %s", count, source)
        return LoadResponse(status="success", chunks_loaded=count, source=source)

    except HTTPException:
        raise
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


# ── GET /health ────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_endpoint():
    """Health check — reports Milvus connection status and document count.

    Always returns HTTP 200, even when Milvus is unavailable.
    The milvus field indicates the connection status.
    """
    try:
        if not is_available():
            return HealthResponse(
                status="ok",
                milvus="error - milvus_lite not installed",
                documents_indexed=0,
            )

        exists = collection_exists()
        doc_count = get_document_count() if exists else 0
        milvus_status = "connected"
    except Exception as exc:
        logger.error("Health check — Milvus error: %s", exc)
        return HealthResponse(
            status="ok",
            milvus=f"error - {exc}",
            documents_indexed=0,
        )

    return HealthResponse(
        status="ok",
        milvus=milvus_status,
        documents_indexed=doc_count,
    )


# ── GET /traces ────────────────────────────────────────────────────────────

@router.get("/traces")
async def traces_endpoint():
    """Return the most recent 20 trace logs."""
    traces = _trace_manager.get_recent_traces(n=20)
    return {"traces": traces}
