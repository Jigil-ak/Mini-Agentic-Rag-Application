"""
Central configuration module.

Loads all settings from a .env file using python-dotenv.
Uses simple module-level constants — no Pydantic settings, no dataclass.
Every downstream module imports values from here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ────────────────────────────────────────────
# Walk up from this file (backend/config.py) to the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# ── LLM API Keys ──────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ── Agent Configuration ───────────────────────────────────────────────────
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))

# ── Chunking Configuration ───────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Retrieval Configuration ──────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "3"))

# ── Model Configuration ──────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "gemini-2.0-flash")
FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "llama-3.1-8b-instant")

# ── Storage Configuration ────────────────────────────────────────────────
MILVUS_DB_PATH: str = os.getenv("MILVUS_DB_PATH", "./data/milvus_demo.db")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_documents")
LOG_FILE: str = os.getenv("LOG_FILE", "./logs/agent_logs.jsonl")

# ── Server Configuration ─────────────────────────────────────────────────
FASTAPI_PORT: int = int(os.getenv("FASTAPI_PORT", "8000"))
