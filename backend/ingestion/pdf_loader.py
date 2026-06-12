"""
PDF document loader using pdfplumber.

Extracts text page by page from PDF files with robust error handling
for encrypted, empty, or corrupted documents.
"""

import logging
from pathlib import Path
from typing import List

import pdfplumber

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> List[str]:
    """Extract text from a PDF file, one string per page.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of strings, one per page that contained extractable text.
        Empty pages are silently skipped.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the PDF is encrypted, yields no text at all,
                     or cannot be opened.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    pages_text: List[str] = []

    try:
        with pdfplumber.open(path) as pdf:
            if not pdf.pages:
                raise ValueError(f"PDF has no pages: {file_path}")

            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        pages_text.append(text.strip())
                    else:
                        logger.debug("Page %d is empty, skipping", page_num)
                except Exception as exc:
                    logger.warning(
                        "Failed to extract text from page %d: %s", page_num, exc
                    )
                    continue

    except pdfplumber.pdfminer.pdfparser.PDFSyntaxError as exc:
        raise ValueError(f"Invalid or corrupted PDF file: {exc}") from exc
    except Exception as exc:
        # Catch encrypted PDFs and other pdfplumber errors
        error_msg = str(exc).lower()
        if "encrypt" in error_msg or "password" in error_msg:
            raise ValueError(
                f"PDF is encrypted and cannot be read: {file_path}"
            ) from exc
        raise ValueError(f"Failed to open PDF: {exc}") from exc

    if not pages_text:
        raise ValueError(
            f"No extractable text found in PDF: {file_path}. "
            "The document may be scanned/image-based."
        )

    logger.info(
        "Loaded %d pages with text from %s", len(pages_text), path.name
    )
    return pages_text
