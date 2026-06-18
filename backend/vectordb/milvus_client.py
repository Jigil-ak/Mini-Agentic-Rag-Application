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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try importing from pymilvus safely to prevent import-time crashes if pymilvus
# or any of its C-extensions/milvus-lite components are missing or broken.
try:
    from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
    _pymilvus_import_error = None
except Exception as e:
    _pymilvus_import_error = e

from backend.config import COLLECTION_NAME, MILVUS_DB_PATH

logger = logging.getLogger(__name__)

# ── Singleton client ───────────────────────────────────────────────────────
_client: Any = None
_client_available: bool = False


def _get_client() -> Any:
    """Lazily create the MilvusClient pointing at the local DB file.

    Raises:
        RuntimeError: If the Milvus Lite client cannot be initialised
                       (e.g. milvus_lite package not installed).
    """
    global _client, _client_available
    if _client is not None:
        return _client

    if _pymilvus_import_error is not None:
        _client_available = False
        raise RuntimeError(
            f"Failed to import pymilvus or its dependencies: {_pymilvus_import_error}. "
            "Ensure milvus_lite is installed: pip install 'pymilvus[milvus_lite]'"
        )

    try:
        # Ensure the parent directory exists before connecting
        db_path = Path(MILVUS_DB_PATH)
        os.makedirs(str(db_path.parent), exist_ok=True)

        logger.info("Connecting to Milvus Lite at %s", MILVUS_DB_PATH)
        _client = MilvusClient(MILVUS_DB_PATH)
        _client_available = True
        logger.info("Milvus Lite client initialised successfully")
    except Exception as exc:
        _client_available = False
        _client = None
        raise RuntimeError(
            f"Failed to initialise Milvus Lite client at '{MILVUS_DB_PATH}': {exc}. "
            "Ensure milvus_lite is installed: pip install 'pymilvus[milvus_lite]'"
        ) from exc

    return _client


# ── Schema definition ─────────────────────────────────────────────────────

def _build_schema() -> CollectionSchema:
    """Build the collection schema with id, text, embedding, source fields."""
    if _pymilvus_import_error is not None:
        raise RuntimeError(
            f"Cannot build schema because pymilvus failed to import: {_pymilvus_import_error}"
        )
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

def is_available() -> bool:
    """Check whether the Milvus Lite client is available and functional.

    Attempts to connect if not already connected. Returns True if the
    client initialised successfully, False otherwise. Never raises.
    """
    global _client_available
    if _client is not None:
        return _client_available
    try:
        _get_client()
        return _client_available
    except Exception as exc:
        logger.warning("Milvus is not available: %s", exc)
        return False


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


def _format_search_hits(search_results) -> List[Dict[str, Any]]:
    """Format raw search results from Milvus client."""
    formatted_hits: List[Dict[str, Any]] = []
    if search_results and len(search_results) > 0:
        for hit in search_results[0]:
            # Handle both dictionary-key access and object-attribute access seamlessly
            distance = hit.get("distance", 1.0) if isinstance(hit, dict) else getattr(hit, "distance", 1.0)
            entity = hit.get("entity", {}) if isinstance(hit, dict) else getattr(hit, "entity", {})
            
            text = entity.get("text", "") if isinstance(entity, dict) else getattr(entity, "text", "")
            source = entity.get("source", "") if isinstance(entity, dict) else getattr(entity, "source", "")
            
            normalized_similarity = 1.0 - distance
            formatted_hits.append({
                "text": text,
                "source": source,
                "distance": distance,
                "similarity": max(0.0, min(1.0, normalized_similarity))
            })
    return formatted_hits


def search_documents(
    query_embedding: List[float],
    top_k: int = 3,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search for the most similar documents to the query embedding.

    Args:
        query_embedding: 384-dimensional query vector.
        top_k: Number of results to return.
        source: Optional source string to restrict the search.

    Returns:
        List of dicts with keys: text, source, distance, similarity.
        Similarity is normalised as 1.0 - distance (COSINE metric).
        Results are sorted by similarity descending (best match first).
    """
    client = _get_client()

    # Crucial: explicit load_collection call to guarantee collection is in memory
    try:
        client.load_collection(COLLECTION_NAME)
        logger.info("[MilvusClient] Collection '%s' loaded into memory successfully", COLLECTION_NAME)
    except Exception as exc:
        logger.warning("[MilvusClient] Failed to load collection '%s': %s", COLLECTION_NAME, exc)

    # Prepare search arguments
    search_kwargs = {
        "collection_name": COLLECTION_NAME,
        "data": [query_embedding],
        "limit": top_k,
        "output_fields": ["text", "source"],
    }
    
    if source:
        escaped_source = source.replace('"', '\\"')
        search_kwargs["filter"] = f'source == "{escaped_source}"'
        logger.info("[MilvusClient] Searching with source filter: '%s'", source)
    else:
        logger.info("[MilvusClient] Searching without source filter")

    # Wrap client search in try/except with fallback
    try:
        search_results = client.search(
            **search_kwargs,
            search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
        )
    except Exception as exc:
        logger.warning(
            "[MilvusClient] Search with custom parameters failed: %s. Falling back to empty search params.",
            exc,
        )
        try:
            search_results = client.search(
                **search_kwargs,
                search_params={},
            )
        except Exception as fallback_exc:
            logger.error("[MilvusClient] Fallback search failed: %s", fallback_exc)
            raise fallback_exc

    formatted_hits = _format_search_hits(search_results)

    retrieved_chunk_count = len(formatted_hits)
    top_similarity = formatted_hits[0]["similarity"] if retrieved_chunk_count > 0 else 0.0
    top_source = formatted_hits[0]["source"] if retrieved_chunk_count > 0 else "N/A"
    top_chunk_preview = formatted_hits[0]["text"][:100] if retrieved_chunk_count > 0 else "N/A"

    logger.info(
        "[MilvusClient] search_documents output - retrieved_chunk_count=%d, "
        "top_similarity=%.4f, top_source='%s', top_chunk_preview='%s'",
        retrieved_chunk_count, top_similarity, top_source, top_chunk_preview
    )

    # Parallel comparison check when source filter is active
    if source:
        no_filter_kwargs = {
            "collection_name": COLLECTION_NAME,
            "data": [query_embedding],
            "limit": top_k,
            "output_fields": ["text", "source"],
        }
        try:
            raw_no_filter = client.search(
                **no_filter_kwargs,
                search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
            )
            no_filter_hits = _format_search_hits(raw_no_filter)
            logger.info(
                "[MilvusClient] Source comparison - with_filter_count=%d, without_filter_count=%d",
                retrieved_chunk_count, len(no_filter_hits)
            )
            if retrieved_chunk_count == 0 and len(no_filter_hits) > 0:
                logger.warning(
                    "[MilvusClient] Filter '%s' returned 0 results, but search without filter returned %d results! "
                    "Top unfiltered source: '%s', top similarity: %.4f",
                    source, len(no_filter_hits), no_filter_hits[0]["source"], no_filter_hits[0]["similarity"]
                )
        except Exception as comp_exc:
            logger.warning("[MilvusClient] Failed parallel comparison search: %s", comp_exc)

    return formatted_hits


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


def reset_collection() -> None:
    """Drop the collection if it exists and recreate it."""
    client = _get_client()
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        logger.info("Dropped collection '%s'", COLLECTION_NAME)
    ensure_collection()
