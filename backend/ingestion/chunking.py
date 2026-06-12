"""
Text chunking using LangChain's RecursiveCharacterTextSplitter.

This is the ONLY LangChain component used in the entire application.
Splits text into overlapping chunks suitable for embedding and retrieval.
"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP

# Pre-configured splitter instance (stateless, safe to reuse)
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_texts(texts: List[str]) -> List[str]:
    """Split a list of text strings into a flat list of chunks.

    Args:
        texts: Raw text strings (e.g. one per PDF page or one per URL).

    Returns:
        Flat list of all chunks produced from all input texts.
        Empty inputs or whitespace-only strings are silently skipped.
    """
    if not texts:
        return []

    all_chunks: List[str] = []
    for text in texts:
        if not text or not text.strip():
            continue
        chunks = _splitter.split_text(text)
        all_chunks.extend(chunks)

    return all_chunks
