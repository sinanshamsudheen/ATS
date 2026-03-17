"""
Deterministically fix corrupted training data in data/raw_dataset.json.

Problems corrected (no LLM calls required):
  1. matched_skills  — replaced with ground-truth keyword intersection
  2. missing_skills  — replaced with JD keywords absent from the resume
  3. keyword_coverage — recomputed from actual overlap
  4. bullet_quality   — deterministically scored from resume bullet analysis
  5. formatting       — deterministically scored from resume structure checks
  6. structure        — deterministically scored from section order/presence
  7. ats_score        — recomputed using weighted breakdown components
  8. weak_bullets     — cross-contaminated entries removed; weak bullets
                        identified and rewritten deterministically
  9. formatting_issues — regenerated from actual resume analysis
 10. overall_feedback  — regenerated from actual matched/missing skills

Run from project root:
    python scripts/fix_dataset.py
"""

import json
import os
import re
import sys
from pathlib import Path

# Ensure project root is on the path so src.* imports work
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.scoring import compute_overlap, extract_skills  # noqa: E402
from src.training.dataset_loader import prepare_dataset  # noqa: E402

RAW_PATH = ROOT / "colab" / "data" / "raw_dataset.json"
TRAIN_PATH = ROOT / "colab" / "data" / "train.json"
VAL_PATH = ROOT / "colab" / "data" / "validation.json"

# Weights for recomputing ats_score from breakdown components
SCORE_WEIGHTS = {
    "keyword_coverage": 0.50,
    "bullet_quality": 0.25,
    "formatting": 0.15,
    "structure": 0.10,
}

# ---------------------------------------------------------------
# Bullet-quality scoring helpers
# ---------------------------------------------------------------

# Strong action verbs that indicate good bullet quality
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

# Passive / weak phrases
_WEAK_PATTERNS = [
    r"\bresponsible for\b",
    r"\bworked on\b",
    r"\bhelped with\b",
    r"\bassisted\b",
    r"\bparticipated in\b",
    r"\binvolved in\b",
]

# Metric indicators ($, %, numbers with context)
_METRIC_RE = re.compile(
    r"\$[\d,.]+|"           # dollar amounts
    r"\d+\s*%|"             # percentages
    r"\d+[KkMm]\+?\b|"     # shorthand numbers like 5M, 10K
    r"\d{2,}[\+]?\s*(users|requests|transactions|events|employees|"
    r"applications|teams|developers|records|services|engineers|"
    r"stakeholders|customers|tests|modules|endpoints|dashboards|"
    r"environments|business units|alerts|incidents|assessments|"
    r"bullet|page|hours|minutes|days|months|years)",
    re.IGNORECASE,
)


def _extract_bullets(resume_text: str):
    """Extract bullet-point lines from resume text."""
    bullets = []
    for line in resume_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
    return bullets


def _score_bullet_quality(resume_text: str) -> int:
    """Score bullet quality 0-100 based on action verbs, metrics, and phrasing."""
    bullets = _extract_bullets(resume_text)
    if not bullets:
        return 40  # no bullets found — mediocre default

    total_score = 0
    for bullet in bullets:
        b_score = 50  # baseline per bullet
        lower = bullet.lower()
        first_word = lower.split()[0] if lower.split() else ""

        # +20 for strong action verb at start
        if first_word in _STRONG_VERBS:
            b_score += 20
        # +15 for strong verb anywhere
        elif any(v in lower for v in _STRONG_VERBS):
            b_score += 10

        # +20 for quantifiable metrics
        if _METRIC_RE.search(bullet):
            b_score += 20

        # -15 for weak/passive phrasing
        for pattern in _WEAK_PATTERNS:
            if re.search(pattern, lower):
                b_score -= 15
                break

        # +10 for sufficient length (detailed bullet)
        if len(bullet.split()) >= 10:
            b_score += 10

        total_score += min(max(b_score, 0), 100)

    return min(int(round(total_score / len(bullets))), 100)


# ---------------------------------------------------------------
# Formatting scoring helpers
# ---------------------------------------------------------------

_EXPECTED_SECTIONS = {
    "summary", "experience", "education", "skills",
    "certifications", "projects",
}


def _score_formatting(resume_text: str) -> tuple:
    """Score formatting 0-100 and return (score, list_of_issues)."""
    score = 85  # start high, deduct for problems
    issues = []
    lower = resume_text.lower()

    # Check for section headers
    found_sections = set()
    for section in _EXPECTED_SECTIONS:
        if re.search(r"\b" + section + r"\b", lower):
            found_sections.add(section)

    missing_sections = {"experience", "education", "skills"} - found_sections
    if missing_sections:
        deduction = len(missing_sections) * 8
        score -= deduction
        for s in sorted(missing_sections):
            issues.append(f"Missing '{s.title()}' section header")

    if "summary" not in found_sections:
        score -= 5
        issues.append("No summary section detected")

    # Check for bullet consistency
    bullets = _extract_bullets(resume_text)
    if not bullets:
        score -= 10
        issues.append("No bullet points found — use bullet format for experience")

    # Check for contact info
    has_email = bool(re.search(r"[\w.-]+@[\w.-]+\.\w+", resume_text))
    has_phone = bool(re.search(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", resume_text))
    if not has_email:
        score -= 5
        issues.append("No email address detected")
    if not has_phone:
        # Minor — many resumes omit phone
        pass

    # Check for consistent date formatting
    date_formats = set()
    if re.search(r"\b\w{3}\s+\d{4}\b", resume_text):  # "Jan 2020"
        date_formats.add("abbr")
    if re.search(r"\b\w+\s+\d{4}\b", resume_text):  # "January 2020"
        date_formats.add("full")
    if re.search(r"\d{2}/\d{2}/\d{4}", resume_text):  # "01/01/2020"
        date_formats.add("numeric")

    # Check for skills section using comma-separated list (ok but could be categorized)
    skills_match = re.search(r"(?i)skills\n(.+)", resume_text)
    if skills_match:
        skills_line = skills_match.group(1)
        if skills_line.count(",") > 5:
            score -= 3
            issues.append("Skills section uses flat comma list — consider grouping by category")

    if not issues:
        issues.append("Resume formatting is clean and ATS-compatible")

    return min(max(score, 15), 100), issues


# ---------------------------------------------------------------
# Structure scoring helpers
# ---------------------------------------------------------------

# Preferred section order
_IDEAL_ORDER = ["summary", "experience", "skills", "education", "certifications", "projects"]


def _score_structure(resume_text: str) -> int:
    """Score structure 0-100 based on section presence and ordering."""
    lower = resume_text.lower()
    score = 70  # baseline

    # Find section positions
    section_positions = {}
    for section in _IDEAL_ORDER:
        match = re.search(r"\b" + section + r"\b", lower)
        if match:
            section_positions[section] = match.start()

    # Must have at least experience and skills
    if "experience" in section_positions:
        score += 10
    if "skills" in section_positions:
        score += 5
    if "education" in section_positions:
        score += 5
    if "summary" in section_positions:
        score += 5

    # Check ordering: experience should come before education
    if ("experience" in section_positions and "education" in section_positions
            and section_positions["experience"] < section_positions["education"]):
        score += 5

    # Check that skills don't come at the very end
    if section_positions:
        max_pos = max(section_positions.values())
        if section_positions.get("skills", 0) == max_pos and len(section_positions) > 1:
            score -= 5

    return min(max(score, 20), 100)


# ---------------------------------------------------------------
# Weak bullet identification and rewriting
# ---------------------------------------------------------------

_REWRITE_MAP = {
    "responsible for": "Led",
    "worked on": "Developed",
    "helped with": "Contributed to",
    "assisted": "Supported",
    "participated in": "Engaged in",
    "involved in": "Drove",
}


def _identify_weak_bullets(resume_text: str) -> list:
    """Identify weak bullets and generate deterministic improvements."""
    bullets = _extract_bullets(resume_text)
    weak = []

    for bullet in bullets:
        lower = bullet.lower()
        first_word = lower.split()[0] if lower.split() else ""
        issue = None
        improved = bullet

        # Check for weak/passive phrasing
        for pattern in _WEAK_PATTERNS:
            match = re.search(pattern, lower)
            if match:
                issue = f"Passive phrasing: '{match.group()}' — use strong action verb"
                # Attempt deterministic rewrite
                for weak_phrase, replacement in _REWRITE_MAP.items():
                    if weak_phrase in lower:
                        improved = re.sub(
                            re.escape(weak_phrase), replacement,
                            bullet, count=1, flags=re.IGNORECASE,
                        )
                        break
                break

        # Check for missing metrics
        if not issue and not _METRIC_RE.search(bullet):
            if len(bullet.split()) >= 5:  # only flag non-trivial bullets
                issue = "Missing quantifiable metrics — add numbers, percentages, or impact"
                improved = bullet + " [add specific metrics]"

        # Check for very short bullets
        if not issue and len(bullet.split()) < 6:
            issue = "Too brief — expand with specific details and measurable results"
            improved = bullet + " [expand with specifics and metrics]"

        if issue:
            weak.append({
                "original": bullet,
                "issue": issue,
                "improved": improved,
            })

    # Return at most 4 weak bullets to avoid overwhelming output
    return weak[:4]


# ---------------------------------------------------------------
# Overall feedback generation
# ---------------------------------------------------------------

_DOMAIN_MAP = {
    "cloud": {"aws", "azure", "gcp", "terraform", "cloudformation", "kubernetes",
              "docker", "serverless", "infrastructure as code", "cloud security"},
    "frontend": {"javascript", "typescript", "react", "angular", "vue", "css",
                 "html", "nextjs", "webpack", "tailwind"},
    "backend": {"java", "python", "node.js", "postgresql", "mongodb", "redis",
                "kafka", "microservices", "rest api", "graphql", "sql"},
    "devops": {"docker", "kubernetes", "terraform", "jenkins", "ci/cd", "ansible",
               "prometheus", "grafana", "argocd", "linux", "devops", "sre"},
    "ml": {"machine learning", "deep learning", "tensorflow", "pytorch", "pandas",
           "scikit-learn", "mlflow", "data science", "python", "spark"},
    "security": {"siem", "penetration testing", "vulnerability assessment",
                 "incident response", "splunk", "nessus", "iso 27001", "soc 2",
                 "zero trust", "devsecops", "cloud security"},
    "data": {"sql", "python", "tableau", "pandas", "spark", "hadoop",
             "data science", "data analysis", "a/b testing", "big data"},
    "product": {"product management", "product roadmap", "agile", "scrum",
                "communication", "product analytics", "leadership"},
}


def _detect_domain(text: str) -> str:
    """Detect the primary domain of a text based on skill overlap."""
    lower = text.lower()
    scores = {}
    for domain, keywords in _DOMAIN_MAP.items():
        count = sum(1 for kw in keywords if kw in lower)
        scores[domain] = count
    if scores:
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
    return "general"


def _generate_feedback(
    matched: list, missing: list, keyword_coverage: int,
    resume_text: str, jd_text: str,
) -> str:
    """Generate grounded overall_feedback from actual data."""
    resume_domain = _detect_domain(resume_text)
    jd_domain = _detect_domain(jd_text)

    parts = []

    # Domain mismatch detection
    if keyword_coverage == 0 and resume_domain != jd_domain:
        parts.append(
            f"This resume focuses on {resume_domain} while the JD targets {jd_domain}. "
            f"There is no meaningful skill overlap."
        )
    elif keyword_coverage < 20 and resume_domain != jd_domain:
        parts.append(
            f"Limited alignment: resume background is {resume_domain}, "
            f"JD requires {jd_domain} expertise. "
            f"Only {len(matched)} overlapping skill{'s' if len(matched) != 1 else ''} found."
        )
    elif matched:
        matched_str = ", ".join(matched[:5])
        parts.append(f"Good alignment on: {matched_str}.")
    else:
        parts.append("No matching skills detected between resume and JD.")

    # Missing skills
    if missing:
        missing_str = ", ".join(missing[:5])
        count = len(missing)
        parts.append(
            f"Key gaps ({count} missing): {missing_str}"
            + (f" and {count - 5} more." if count > 5 else ".")
        )

    # Bullet quality note
    bullets = _extract_bullets(resume_text)
    weak_count = 0
    for b in bullets:
        lower = b.lower()
        for pattern in _WEAK_PATTERNS:
            if re.search(pattern, lower):
                weak_count += 1
                break
    if weak_count > 0:
        parts.append(
            f"{weak_count} bullet{'s' if weak_count != 1 else ''} "
            f"use{'s' if weak_count == 1 else ''} passive phrasing — rewrite with action verbs and metrics."
        )

    if not parts:
        parts.append("Review resume for better alignment with JD requirements.")

    return " ".join(parts)


# ---------------------------------------------------------------
# Main fix logic
# ---------------------------------------------------------------

def _parse_resume_jd(input_text: str):
    """Split the combined input into (resume_text, jd_text)."""
    marker = "\nJOB DESCRIPTION:\n"
    if marker in input_text:
        parts = input_text.split(marker, 1)
        resume = parts[0].replace("RESUME:\n", "", 1).strip()
        jd = parts[1].strip()
        return resume, jd
    # Fallback — treat the whole thing as resume with empty JD
    return input_text, ""


def _fix_sample(sample: dict) -> dict:
    """Return a corrected copy of one training sample."""
    try:
        output = json.loads(sample["output"])
    except (json.JSONDecodeError, TypeError):
        return sample  # can't parse — leave as-is

    resume_text, jd_text = _parse_resume_jd(sample.get("input", ""))

    # 1. Recompute skills deterministically
    overlap = compute_overlap(resume_text, jd_text)
    output["matched_skills"] = overlap["matched"]
    output["missing_skills"] = overlap["missing"]

    # 2. Fix keyword_coverage in score_breakdown
    breakdown = output.get("score_breakdown", {})
    breakdown["keyword_coverage"] = overlap["keyword_coverage"]

    # 3. Deterministically score bullet_quality, formatting, structure
    breakdown["bullet_quality"] = _score_bullet_quality(resume_text)
    fmt_score, fmt_issues = _score_formatting(resume_text)
    breakdown["formatting"] = fmt_score
    breakdown["structure"] = _score_structure(resume_text)
    output["score_breakdown"] = breakdown

    # 4. Recompute ats_score from weighted breakdown
    score = 0.0
    for key, weight in SCORE_WEIGHTS.items():
        component = breakdown.get(key, 0)
        if isinstance(component, (int, float)):
            score += component * weight
    raw_score = int(round(min(max(score, 0), 100)))

    # 5. Domain-mismatch penalty — cap score at 20 if coverage < 20
    if breakdown["keyword_coverage"] < 20:
        raw_score = min(raw_score, 20)

    output["ats_score"] = raw_score

    # 6. Fix weak_bullets — remove cross-contaminated, generate from analysis
    weak_bullets = _identify_weak_bullets(resume_text)
    # Keep only bullets whose original text is actually in the resume
    resume_lower = resume_text.lower()
    output["weak_bullets"] = [
        b for b in weak_bullets
        if b["original"].lower() in resume_lower
    ]

    # 7. Fix formatting_issues from actual analysis
    output["formatting_issues"] = fmt_issues

    # 8. Fix overall_feedback from actual matched/missing skills
    output["overall_feedback"] = _generate_feedback(
        overlap["matched"], overlap["missing"],
        overlap["keyword_coverage"], resume_text, jd_text,
    )

    fixed = dict(sample)
    fixed["output"] = json.dumps(output, ensure_ascii=False)
    # Remove stale formatted prompt — will be regenerated by prepare_dataset
    fixed.pop("text", None)
    return fixed


def _oversample(bucket: list, target: int) -> list:
    """Duplicate samples in *bucket* until it reaches *target* size."""
    if not bucket or target <= 0:
        return []
    multiplier = max(target // len(bucket), 1)
    remainder = target - multiplier * len(bucket)
    return bucket * multiplier + bucket[:max(remainder, 0)]


# Target total dataset size — adjust this single constant to grow the dataset.
TARGET_TOTAL = 500


def _balance_dataset(samples: list) -> list:
    """Oversample all three tiers to reach TARGET_TOTAL (~equal per tier).

        - high   (keyword_coverage >= 50): strong domain match
        - partial (keyword_coverage 20-49): adjacent-domain / partial match
        - mismatch (keyword_coverage < 20): cross-domain mismatch
    """
    high, partial, mismatch = [], [], []
    for s in samples:
        try:
            kc = json.loads(s["output"])["score_breakdown"]["keyword_coverage"]
        except Exception:
            mismatch.append(s)
            continue
        if kc >= 50:
            high.append(s)
        elif kc >= 20:
            partial.append(s)
        else:
            mismatch.append(s)

    per_tier = TARGET_TOTAL // 3
    print(f"\n  Balancing dataset (target {TARGET_TOTAL} total, ~{per_tier} per tier):")

    balanced_high = _oversample(high, per_tier)
    print(f"    high    : {len(high):3d} -> {len(balanced_high)}")

    balanced_partial = _oversample(partial, per_tier)
    print(f"    partial : {len(partial):3d} -> {len(balanced_partial)}")

    balanced_mismatch = _oversample(mismatch, per_tier)
    print(f"    mismatch: {len(mismatch):3d} -> {len(balanced_mismatch)}")

    balanced = balanced_high + balanced_partial + balanced_mismatch
    print(f"  Total: {len(samples)} -> {len(balanced)}")
    return balanced


def main():
    print("=" * 60)
    print("ATS Dataset Fix Script (v2 — full sub-score repair)")
    print("=" * 60)

    if not RAW_PATH.exists():
        print(f"ERROR: {RAW_PATH} not found.")
        sys.exit(1)

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"\nLoaded {len(raw_data)} samples from {RAW_PATH}")

    fixed_data = []
    score_stats = {"min": 100, "max": 0, "sum": 0}
    for i, sample in enumerate(raw_data):
        fixed = _fix_sample(sample)
        fixed_data.append(fixed)

        try:
            out = json.loads(fixed["output"])
            s = out["ats_score"]
            score_stats["min"] = min(score_stats["min"], s)
            score_stats["max"] = max(score_stats["max"], s)
            score_stats["sum"] += s
        except Exception:
            pass

    avg = score_stats["sum"] / len(fixed_data) if fixed_data else 0

    print(f"\nFixed {len(fixed_data)} samples.")
    print(f"Score range: {score_stats['min']} - {score_stats['max']}  "
          f"(avg: {avg:.1f})")

    # Balance by oversampling under-represented match tiers
    balanced_data = _balance_dataset(fixed_data)

    # Overwrite raw_dataset.json with balanced data
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(balanced_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved balanced data -> {RAW_PATH}")

    # Regenerate train/validation splits
    print()
    prepare_dataset(
        raw_path=str(RAW_PATH),
        train_path=str(TRAIN_PATH),
        val_path=str(VAL_PATH),
    )


if __name__ == "__main__":
    main()
