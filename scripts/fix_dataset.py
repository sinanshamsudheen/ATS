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
from src.scoring.resume_scorer import (  # noqa: E402
    extract_bullets,
    score_bullet_quality,
    score_formatting,
    score_structure,
    identify_weak_bullets,
)
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
    bullets = extract_bullets(resume_text)
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
    breakdown["bullet_quality"] = score_bullet_quality(resume_text)
    fmt_score, fmt_issues = score_formatting(resume_text)
    breakdown["formatting"] = fmt_score
    breakdown["structure"] = score_structure(resume_text)
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
    weak_bullets = identify_weak_bullets(resume_text)
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
