"""ATS formatting compliance checker using PyMuPDF."""

import logging
from typing import Dict, Any, List

import fitz  # PyMuPDF

from ..config import ATS_FORMAT_RULES

logger = logging.getLogger(__name__)


class FormattingChecker:
    """Checks a PDF resume for ATS-unfriendly formatting issues."""

    def check(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyse a PDF for formatting issues.

        Returns dict with 'issues' list and 'score' (0–100).
        """
        issues: List[Dict[str, str]] = []

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Cannot open PDF: {e}")
            return {"issues": [{"type": "critical", "message": f"Cannot open PDF: {e}"}], "score": 0}

        # Page count
        page_count = len(doc)
        if page_count > ATS_FORMAT_RULES["max_pages"]:
            issues.append({
                "type": "high",
                "message": f"Resume is {page_count} pages (recommended max {ATS_FORMAT_RULES['max_pages']})",
            })

        # Check for images / non-text content
        total_text_len = 0
        total_images = 0
        for page in doc:
            total_text_len += len(page.get_text())
            total_images += len(page.get_images())

        if doc.is_encrypted:
            issues.append({"type": "critical", "message": "PDF is encrypted — ATS cannot read it"})

        if total_images > ATS_FORMAT_RULES["max_images"]:
            issues.append({
                "type": "medium",
                "message": f"Found {total_images} images — ATS may not parse image-heavy resumes",
            })

        if total_text_len < ATS_FORMAT_RULES["min_text_length"]:
            issues.append({
                "type": "high",
                "message": "Very little extractable text — resume may be image-based",
            })

        # Text-to-total ratio heuristic
        if page_count > 0:
            avg_text = total_text_len / page_count
            if avg_text < 200:
                issues.append({
                    "type": "medium",
                    "message": "Low text density per page — consider adding more content",
                })

        doc.close()

        # Score: start at 100, deduct per issue severity
        penalty_map = {"critical": 40, "high": 20, "medium": 10, "info": 5}
        score = 100
        for issue in issues:
            score -= penalty_map.get(issue["type"], 5)
        score = max(0, score)

        return {"issues": issues, "score": score}
