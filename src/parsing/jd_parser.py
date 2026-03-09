"""Job description parser — keyword extraction from JD text."""

from typing import List


def extract_keywords(text: str, max_keywords: int = 20) -> List[str]:
    """
    Extract unique keywords (>4 chars) from job description text.

    Args:
        text: Raw job description string.
        max_keywords: Maximum number of keywords to return.

    Returns:
        Deduplicated list of keywords.
    """
    words = [w.strip() for w in text.split() if len(w) > 4]
    seen: set[str] = set()
    unique: List[str] = []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            unique.append(w)
    return unique[:max_keywords]
