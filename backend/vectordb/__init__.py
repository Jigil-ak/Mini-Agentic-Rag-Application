"""Vector DB package — Milvus Lite client."""

from backend.vectordb.milvus_client import (
    ensure_collection,
    insert_documents,
    search_documents,
    get_document_count,
    collection_exists,
    is_available,
)

__all__ = [
    "ensure_collection",
    "insert_documents",
    "search_documents",
    "get_document_count",
    "collection_exists",
    "is_available",
]
