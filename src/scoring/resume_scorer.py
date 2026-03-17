"""
Deterministic resume sub-score computation.

Used by:
  - scripts/fix_dataset.py  (to generate ground-truth training labels)
  - src/app/streamlit_app.py (to fill in sub-scores the model omits or zeros)

All functions are pure — no LLM calls, no randomness.
"""

import re
from typing import Tuple, List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STRONG_VERBS = {
    "achieved", "architected", "automated", "built", "consolidated",
    "decreased", "delivered", "deployed", "designed", "developed",
    "drove", "eliminated", "engineered", "established", "exceeded",
    "expanded", "generated", "implemented", "improved", "increased",
    "integrated", "launched", "led", "managed", "migrated",
    "negotiated", "optimized", "orchestrated", "pioneered", "processed",
    "produced", "published", "rebuilt", "redesigned", "reduced",
    "refactored", "replaced", "scaled", "secured", "spearheaded",
    "streamlined", "transformed", "upgraded",
}

_WEAK_PATTERNS = [
    r"\bresponsible for\b",
    r"\bworked on\b",
    r"\bhelped with\b",
    r"\bassisted\b",
    r"\bparticipated in\b",
    r"\binvolved in\b",
]

_METRIC_RE = re.compile(
    r"\$[\d,.]+|"
    r"\d+\s*%|"
    r"\d+[KkMm]\+?\b|"
    r"\d{2,}[\+]?\s*(users|requests|transactions|events|employees|"
    r"applications|teams|developers|records|services|engineers|"
    r"stakeholders|customers|tests|modules|endpoints|dashboards|"
    r"environments|business units|alerts|incidents|assessments|"
    r"bullet|page|hours|minutes|days|months|years)",
    re.IGNORECASE,
)

_EXPECTED_SECTIONS = {"summary", "experience", "education", "skills", "certifications", "projects"}
_IDEAL_ORDER = ["summary", "experience", "skills", "education", "certifications", "projects"]

_REWRITE_MAP = {
    "responsible for": "Led",
    "worked on": "Developed",
    "helped with": "Contributed to",
    "assisted": "Supported",
    "participated in": "Engaged in",
    "involved in": "Drove",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_bullets(resume_text: str) -> List[str]:
    """Return all bullet-point lines (starting with '- ') from the resume."""
    bullets = []
    for line in resume_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
    return bullets


# ---------------------------------------------------------------------------
# Sub-score functions
# ---------------------------------------------------------------------------

def score_bullet_quality(resume_text: str) -> int:
    """Score bullet quality 0-100 based on action verbs, metrics, and phrasing."""
    bullets = extract_bullets(resume_text)
    if not bullets:
        return 40

    total = 0
    for bullet in bullets:
        b = 50
        lower = bullet.lower()
        first_word = lower.split()[0] if lower.split() else ""

        if first_word in _STRONG_VERBS:
            b += 20
        elif any(v in lower for v in _STRONG_VERBS):
            b += 10

        if _METRIC_RE.search(bullet):
            b += 20

        for pattern in _WEAK_PATTERNS:
            if re.search(pattern, lower):
                b -= 15
                break

        if len(bullet.split()) >= 10:
            b += 10

        total += min(max(b, 0), 100)

    return min(int(round(total / len(bullets))), 100)


def score_formatting(resume_text: str) -> Tuple[int, List[str]]:
    """Score formatting 0-100 and return (score, list_of_issues)."""
    score = 85
    issues = []
    lower = resume_text.lower()

    found_sections = set()
    for section in _EXPECTED_SECTIONS:
        if re.search(r"\b" + section + r"\b", lower):
            found_sections.add(section)

    missing_core = {"experience", "education", "skills"} - found_sections
    if missing_core:
        score -= len(missing_core) * 8
        for s in sorted(missing_core):
            issues.append(f"Missing '{s.title()}' section header")

    if "summary" not in found_sections:
        score -= 5
        issues.append("No summary section detected")

    if not extract_bullets(resume_text):
        score -= 10
        issues.append("No bullet points found — use bullet format for experience")

    if not re.search(r"[\w.-]+@[\w.-]+\.\w+", resume_text):
        score -= 5
        issues.append("No email address detected")

    skills_match = re.search(r"(?i)skills\n(.+)", resume_text)
    if skills_match and skills_match.group(1).count(",") > 5:
        score -= 3
        issues.append("Skills section uses flat comma list — consider grouping by category")

    if not issues:
        issues.append("Resume formatting is clean and ATS-compatible")

    return min(max(score, 15), 100), issues


def score_structure(resume_text: str) -> int:
    """Score structure 0-100 based on section presence and ordering."""
    lower = resume_text.lower()
    score = 70

    positions = {}
    for section in _IDEAL_ORDER:
        m = re.search(r"\b" + section + r"\b", lower)
        if m:
            positions[section] = m.start()

    if "experience" in positions:
        score += 10
    if "skills" in positions:
        score += 5
    if "education" in positions:
        score += 5
    if "summary" in positions:
        score += 5

    if ("experience" in positions and "education" in positions
            and positions["experience"] < positions["education"]):
        score += 5

    if positions:
        max_pos = max(positions.values())
        if positions.get("skills", 0) == max_pos and len(positions) > 1:
            score -= 5

    return min(max(score, 20), 100)


def identify_weak_bullets(resume_text: str) -> List[dict]:
    """Identify weak bullets with deterministic rewrites. Returns up to 4."""
    bullets = extract_bullets(resume_text)
    weak = []
    resume_lower = resume_text.lower()

    for bullet in bullets:
        lower = bullet.lower()
        issue = None
        improved = bullet

        for pattern in _WEAK_PATTERNS:
            m = re.search(pattern, lower)
            if m:
                issue = f"Passive phrasing: '{m.group()}' — use a strong action verb"
                for phrase, replacement in _REWRITE_MAP.items():
                    if phrase in lower:
                        improved = re.sub(
                            re.escape(phrase), replacement,
                            bullet, count=1, flags=re.IGNORECASE,
                        )
                        break
                break

        if not issue and not _METRIC_RE.search(bullet) and len(bullet.split()) >= 5:
            issue = "Missing quantifiable metrics — add numbers, percentages, or impact"
            improved = bullet + " [add specific metrics]"

        if not issue and len(bullet.split()) < 6:
            issue = "Too brief — expand with specific details and measurable results"
            improved = bullet + " [expand with specifics and metrics]"

        if issue and bullet.lower() in resume_lower:
            weak.append({"original": bullet, "issue": issue, "improved": improved})

    return weak[:4]
