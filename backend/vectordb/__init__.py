"""Vector DB package — Milvus Lite client."""

from backend.vectordb.milvus_client import (
    ensure_collection,
    insert_documents,
    search_documents,
    get_document_count,
    collection_exists,
)

__all__ = [
    "ensure_collection",
    "insert_documents",
    "search_documents",
    "get_document_count",
    "collection_exists",
]
