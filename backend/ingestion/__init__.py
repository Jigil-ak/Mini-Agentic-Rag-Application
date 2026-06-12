"""Ingestion package — document loading, chunking, and embedding."""

from backend.ingestion.chunking import chunk_texts
from backend.ingestion.embedding import embed_texts, embed_query
from backend.ingestion.pdf_loader import load_pdf
from backend.ingestion.url_loader import load_url

__all__ = ["chunk_texts", "embed_texts", "embed_query", "load_pdf", "load_url"]
