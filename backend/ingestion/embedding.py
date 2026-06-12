"""
Sentence-transformer embedding module.

Uses the all-MiniLM-L6-v2 model (384-dimensional vectors) loaded once
via a singleton pattern to avoid repeated model downloads and GPU/CPU
initialization overhead.
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from backend.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# ── Singleton model instance ───────────────────────────────────────────────
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazily load the SentenceTransformer model exactly once."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded — dimension: %d", _model.get_sentence_embedding_dimension())
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of text strings into 384-dimensional vectors.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors, each a plain Python list of floats.
        Numpy arrays are converted to Python lists for JSON/Milvus
        compatibility.

    Raises:
        ValueError: If texts is empty.
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts")

    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False)

    # Convert numpy arrays → Python lists of floats
    return [embedding.tolist() for embedding in embeddings]


def embed_query(query: str) -> List[float]:
    """Embed a single query string into a 384-dimensional vector.

    Args:
        query: The search query to embed.

    Returns:
        A single embedding vector as a plain Python list of floats.

    Raises:
        ValueError: If query is empty or whitespace-only.
    """
    if not query or not query.strip():
        raise ValueError("Cannot embed an empty query")

    model = _get_model()
    embedding = model.encode(query, show_progress_bar=False)

    # Convert numpy array → Python list of floats
    return embedding.tolist()
