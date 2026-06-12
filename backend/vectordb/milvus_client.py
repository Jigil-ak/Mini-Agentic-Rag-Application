"""
Milvus Lite vector database client.

Uses MilvusClient with a local SQLite-backed file — no Docker,
no external deployment. Collection schema uses COSINE metric
with IVF_FLAT index for similarity search.

IMPORTANT — Similarity Score Handling:
    Milvus COSINE metric returns *distance* where lower = more similar.
    We normalise to similarity = 1.0 - distance before returning results,
    so consumers always see 1.0 = identical, 0.0 = opposite.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from backend.config import COLLECTION_NAME, MILVUS_DB_PATH

logger = logging.getLogger(__name__)

# ── Singleton client ───────────────────────────────────────────────────────
_client: MilvusClient | None = None


def _get_client() -> MilvusClient:
    """Lazily create the MilvusClient pointing at the local DB file."""
    global _client
    if _client is None:
        # Ensure the parent directory exists
        db_path = Path(MILVUS_DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Connecting to Milvus Lite at %s", MILVUS_DB_PATH)
        _client = MilvusClient(MILVUS_DB_PATH)
    return _client


# ── Schema definition ─────────────────────────────────────────────────────

def _build_schema() -> CollectionSchema:
    """Build the collection schema with id, text, embedding, source fields."""
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        ),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=65535,
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=384,
        ),
        FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=500,
        ),
    ]
    return CollectionSchema(fields=fields, description="RAG document chunks")


# ── Public API ─────────────────────────────────────────────────────────────

def ensure_collection() -> None:
    """Create the collection with IVF_FLAT + COSINE index if it doesn't exist."""
    client = _get_client()

    if client.has_collection(COLLECTION_NAME):
        logger.info("Collection '%s' already exists", COLLECTION_NAME)
        return

    schema = _build_schema()
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
    )

    # Create IVF_FLAT index on the embedding field with COSINE metric
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128},
    )
    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params,
    )

    logger.info("Created collection '%s' with IVF_FLAT/COSINE index", COLLECTION_NAME)


def insert_documents(
    chunks: List[str],
    embeddings: List[List[float]],
    source: str,
) -> int:
    """Insert text chunks with their embeddings into the collection.

    Args:
        chunks: List of text strings.
        embeddings: Corresponding embedding vectors (384-dim each).
        source: Origin identifier (filename or URL).

    Returns:
        Number of documents inserted.

    Raises:
        ValueError: If chunks and embeddings have mismatched lengths.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
        )

    if not chunks:
        logger.warning("No documents to insert")
        return 0

    client = _get_client()

    data = [
        {
            "text": chunk,
            "embedding": emb,
            "source": source[:500],  # truncate to field max length
        }
        for chunk, emb in zip(chunks, embeddings)
    ]

    result = client.insert(collection_name=COLLECTION_NAME, data=data)
    count = result.get("insert_count", len(chunks))
    logger.info("Inserted %d documents from source '%s'", count, source)
    return count


def search_documents(
    query_embedding: List[float],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Search for the most similar documents to the query embedding.

    Args:
        query_embedding: 384-dimensional query vector.
        top_k: Number of results to return.

    Returns:
        List of dicts with keys: text, source, distance, similarity.
        Similarity is normalised as 1.0 - distance (COSINE metric).
        Results are sorted by similarity descending (best match first).
    """
    client = _get_client()

    raw_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=top_k,
        output_fields=["text", "source"],
        search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
    )

    results: List[Dict[str, Any]] = []

    if not raw_results or not raw_results[0]:
        return results

    for hit in raw_results[0]:
        distance = hit.get("distance", 1.0)
        similarity = 1.0 - distance  # normalise: 1.0 = identical
        entity = hit.get("entity", {})
        results.append(
            {
                "text": entity.get("text", ""),
                "source": entity.get("source", ""),
                "distance": distance,
                "similarity": round(similarity, 6),
            }
        )

    return results


def get_document_count() -> int:
    """Return the total number of documents in the collection.

    Returns 0 if the collection doesn't exist yet.
    """
    client = _get_client()
    if not client.has_collection(COLLECTION_NAME):
        return 0

    stats = client.get_collection_stats(COLLECTION_NAME)
    return stats.get("row_count", 0)


def collection_exists() -> bool:
    """Check whether the RAG collection exists."""
    client = _get_client()
    return client.has_collection(COLLECTION_NAME)
