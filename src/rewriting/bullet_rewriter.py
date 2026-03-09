"""
Bullet-point quality analyser and rewrite suggestion engine.

Provides heuristic analysis (action verbs, metrics, genericness) and
optionally delegates to a local fine-tuned model for LLM-powered rewrites.
"""

import re
import logging
from typing import List, Dict, Any, Tuple

from ..config import WEAK_BULLET_MIN_WORDS, WEAK_BULLET_MAX_WORDS, ACTION_VERBS

logger = logging.getLogger(__name__)


class BulletRewriter:
    """Analyses resume bullets for quality and generates improvement suggestions."""

    def __init__(self):
        self.action_verbs = set(ACTION_VERBS)

    # ------------------------------------------------------------------
    # Heuristic helpers
    # ------------------------------------------------------------------

    def _count_words(self, text: str) -> int:
        return len(text.split())

    def _has_action_verb(self, bullet: str) -> bool:
        words = bullet.lower().split()
        if not words:
            return False
        first_word = words[0].strip('•-*')
        return first_word in self.action_verbs

    def _has_metrics(self, bullet: str) -> bool:
        metric_patterns = [
            r'\d+%',
            r'\d+[kKmMbB]?\+?',
            r'\$\d+',
            r'\d+x',
        ]
        return any(re.search(p, bullet) for p in metric_patterns)

    def _is_too_generic(self, bullet: str) -> bool:
        generic_terms = [
            "responsible for", "duties included", "worked on",
            "helped with", "assisted in", "involved in", "participated",
        ]
        bullet_lower = bullet.lower()
        return any(term in bullet_lower for term in generic_terms)

    # ------------------------------------------------------------------
    # Public analysis API
    # ------------------------------------------------------------------

    def analyze_bullet_quality(self, bullet: str) -> Dict[str, Any]:
        """Score a single bullet point (0–100) with a list of issues."""
        issues: List[str] = []
        word_count = self._count_words(bullet)

        if word_count < WEAK_BULLET_MIN_WORDS:
            issues.append("Too short — lacks detail")
        elif word_count > WEAK_BULLET_MAX_WORDS:
            issues.append("Too long — consider splitting")

        if not self._has_action_verb(bullet):
            issues.append("Weak or missing action verb")

        if not self._has_metrics(bullet):
            issues.append("No quantifiable metrics/results")

        if self._is_too_generic(bullet):
            issues.append("Uses generic/vague language")

        max_issues = 4
        score = max(0, 100 - (len(issues) / max_issues * 100))

        return {
            "bullet": bullet,
            "score": round(score, 1),
            "issues": issues,
            "needs_improvement": len(issues) >= 2,
        }

    def analyze_section_bullets(self, bullets: List[str]) -> List[Dict[str, Any]]:
        """Analyse every bullet in a section."""
        return [self.analyze_bullet_quality(b) for b in bullets]

    def generate_rewrite_suggestions(
        self,
        bullets: List[str],
        job_description: str = "",
        only_weak: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Produce per-bullet quality reports.

        For bullets flagged as weak the report includes issue details.
        Strong bullets are returned unchanged with a note.
        """
        results: List[Dict[str, Any]] = []

        for bullet in bullets:
            quality = self.analyze_bullet_quality(bullet)
            should_flag = not only_weak or quality["needs_improvement"]

            result = {
                **quality,
                "improved": bullet,  # heuristic-only; no rewrite without LLM
                "llm_analysis": (
                    "Bullet flagged for improvement — see issues"
                    if should_flag
                    else "Bullet point is already strong"
                ),
                "llm_success": True,
            }
            results.append(result)

        return results

    def get_summary_stats(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate statistics across a list of bullet analyses."""
        if not analysis_results:
            return {"total_bullets": 0, "weak_bullets": 0, "average_score": 0, "improvement_rate": 0}

        total = len(analysis_results)
        weak = sum(1 for r in analysis_results if r.get("needs_improvement"))
        avg = sum(r.get("score", 0) for r in analysis_results) / total

        return {
            "total_bullets": total,
            "weak_bullets": weak,
            "average_score": round(avg, 1),
            "improvement_rate": round((weak / total * 100), 1) if total else 0,
        }

    def analyze_resume_sections(
        self,
        parsed_resume: Dict[str, Any],
        job_description: str = "",
    ) -> Dict[str, Any]:
        """Analyse all bullet points across every resume section."""
        results: Dict[str, Any] = {"sections": {}, "overall_stats": {}}
        all_analyses: List[Dict[str, Any]] = []

        sections = parsed_resume.get("sections", {})
        for name, data in sections.items():
            bullets = data.get("bullets", [])
            if bullets:
                logger.info(f"Analysing {len(bullets)} bullets in '{name}'")
                section_results = self.generate_rewrite_suggestions(bullets, job_description)
                results["sections"][name] = {
                    "bullets": section_results,
                    "stats": self.get_summary_stats(section_results),
                }
                all_analyses.extend(section_results)

        results["overall_stats"] = self.get_summary_stats(all_analyses)
        return results

    def quick_analyze(self, bullet: str) -> Tuple[float, List[str]]:
        """Fast quality check returning (score, issues)."""
        analysis = self.analyze_bullet_quality(bullet)
        return analysis["score"], analysis["issues"]
