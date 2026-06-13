"""
FastAPI application entry point.

Creates the app, configures CORS, includes routes, and ensures
required directories and the Milvus collection exist on startup.

Startup validation checks and logs Milvus connection, collection status,
and embedding model availability — but never crashes on failure.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.config import FASTAPI_PORT, COLLECTION_NAME
from backend.vectordb.milvus_client import ensure_collection, is_available

# ── Logging configuration ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Startup / Shutdown lifecycle ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: setup on startup, cleanup on shutdown."""
    # ── Startup ────────────────────────────────────────────────────
    logger.info("Starting Agentic RAG application...")

    # Create required directories
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    logger.info("Ensured data/ and logs/ directories exist")

    # ── Validate Milvus connection ─────────────────────────────────
    try:
        if is_available():
            logger.info("✅ Milvus connected successfully")
            try:
                ensure_collection()
                logger.info("✅ Collection '%s' ready", COLLECTION_NAME)
            except Exception as exc:
                logger.error("❌ Failed to ensure collection '%s': %s", COLLECTION_NAME, exc)
        else:
            logger.warning("❌ Milvus unavailable: milvus_lite may not be installed. "
                           "Install with: pip install 'pymilvus[milvus_lite]'")
    except Exception as exc:
        logger.error("❌ Milvus unavailable: %s", exc)

    # ── Validate embedding model ───────────────────────────────────
    try:
        from backend.ingestion.embedding import embed_query
        test_embedding = embed_query("startup test")
        if test_embedding and len(test_embedding) == 384:
            logger.info("✅ Embedding model loaded (dimension: 384)")
        else:
            logger.warning("❌ Embedding model returned unexpected output")
    except Exception as exc:
        logger.error("❌ Embedding model failed to load: %s", exc)

    logger.info("Agentic RAG application started successfully")
    yield

    # ── Shutdown ───────────────────────────────────────────────────
    logger.info("Shutting down Agentic RAG application")


# ── FastAPI application ────────────────────────────────────────────────────

app = FastAPI(
    title="Mini Agentic RAG",
    description=(
        "A production-ready mini agentic RAG application with intelligent routing, "
        "LLM fallback, tool use (calculator + web search), and structured tracing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS middleware (allow all origins for development) ────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include routes ─────────────────────────────────────────────────────────
app.include_router(router)


# ── Uvicorn runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=FASTAPI_PORT,
        reload=True,
    )
