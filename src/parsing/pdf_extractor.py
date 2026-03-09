"""PDF text extraction using PyMuPDF."""

import fitz  # PyMuPDF
import re
import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract and clean text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Cleaned text extracted from all pages.

    Raises:
        ValueError: If the file cannot be processed.
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = []

        for page in doc:
            text = page.get_text()
            full_text.append(text)

        doc.close()

        raw_text = "\n".join(full_text)
        return _clean_text(raw_text)

    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        raise ValueError(f"Failed to process PDF: {e}")


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove artifacts from extracted PDF text."""
    text = text.replace('\x00', '')
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    return text.strip()
