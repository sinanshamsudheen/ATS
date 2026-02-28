"""
Rewrite Engine Module
Analyzes resume content and provides LLM-powered improvement suggestions
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from .llm_inference import get_llm
from ..config import WEAK_BULLET_MIN_WORDS, WEAK_BULLET_MAX_WORDS, MIN_ACTION_VERBS

logger = logging.getLogger(__name__)

class RewriteEngine:
    """
    Analyzes resume bullets and generates improvement suggestions.
    Uses heuristics to identify weak bullets, then applies LLM for rewrites.
    """
    
    def __init__(self):
        """Initialize the rewrite engine."""
        self.llm = None  # Lazy load to avoid slow initialization
        self.action_verbs = set(MIN_ACTION_VERBS)
    
    def _ensure_llm_loaded(self):
        """Lazy load the LLM only when needed."""
        if self.llm is None:
            logger.info("Loading LLM for rewrite suggestions...")
            self.llm = get_llm()
    
    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def _has_action_verb(self, bullet: str) -> bool:
        """Check if bullet starts with a strong action verb."""
        words = bullet.lower().split()
        if not words:
            return False
        
        first_word = words[0].strip('•-*')
        return first_word in self.action_verbs
    
    def _has_metrics(self, bullet: str) -> bool:
        """Check if bullet contains quantifiable metrics."""
        # Look for patterns like: numbers, percentages, dollar amounts, etc.
        metric_patterns = [
            r'\d+%',  # Percentages
            r'\d+[kKmMbB]?\+?',  # Numbers (with optional K, M, B)
            r'\$\d+',  # Dollar amounts
            r'\d+x',  # Multipliers
        ]
        
        for pattern in metric_patterns:
            if re.search(pattern, bullet):
                return True
        return False
    
    def _is_too_generic(self, bullet: str) -> bool:
        """Check if bullet uses generic/vague language."""
        generic_terms = [
            "responsible for", "duties included", "worked on",
            "helped with", "assisted in", "involved in", "participated"
        ]
        
        bullet_lower = bullet.lower()
        return any(term in bullet_lower for term in generic_terms)
    
    def analyze_bullet_quality(self, bullet: str) -> Dict[str, Any]:
        """
        Analyze a single bullet point for quality issues.
        
        Args:
            bullet: The bullet point text to analyze
            
        Returns:
            Dictionary with quality assessment
        """
        issues = []
        word_count = self._count_words(bullet)
        
        # Check length
        if word_count < WEAK_BULLET_MIN_WORDS:
            issues.append("Too short - lacks detail")
        elif word_count > WEAK_BULLET_MAX_WORDS:
            issues.append("Too long - consider splitting")
        
        # Check action verb
        if not self._has_action_verb(bullet):
            issues.append("Weak or missing action verb")
        
        # Check metrics
        if not self._has_metrics(bullet):
            issues.append("No quantifiable metrics/results")
        
        # Check genericness
        if self._is_too_generic(bullet):
            issues.append("Uses generic/vague language")
        
        # Calculate quality score (0-100)
        max_issues = 4
        score = max(0, 100 - (len(issues) / max_issues * 100))
        
        return {
            "bullet": bullet,
            "score": round(score, 1),
            "issues": issues,
            "needs_improvement": len(issues) >= 2  # Flag if 2+ issues (50% score or lower)
        }
    
    def analyze_section_bullets(self, bullets: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple bullets from a resume section.
        
        Args:
            bullets: List of bullet point strings
            
        Returns:
            List of quality assessments
        """
        return [self.analyze_bullet_quality(bullet) for bullet in bullets]
    
    def generate_rewrite_suggestions(
        self, 
        bullets: List[str],
        job_description: str = "",
        only_weak: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate LLM-powered rewrite suggestions for bullets.
        
        Args:
            bullets: List of bullet points to analyze
            job_description: Optional JD for context
            only_weak: If True, only rewrite bullets flagged as weak
            
        Returns:
            List of dictionaries with original, analysis, and improved versions
        """
        self._ensure_llm_loaded()
        
        results = []
        
        for bullet in bullets:
            # First analyze quality
            quality_analysis = self.analyze_bullet_quality(bullet)
            
            # Decide if we need to generate a rewrite
            should_rewrite = not only_weak or quality_analysis["needs_improvement"]
            
            if should_rewrite:
                logger.info(f"Generating rewrite for: {bullet[:50]}...")
                
                # Get LLM suggestion
                llm_result = self.llm.analyze_bullet(bullet, job_description)
                
                result = {
                    **quality_analysis,
                    "llm_analysis": llm_result["analysis"],
                    "improved": llm_result["improved"],
                    "llm_success": llm_result["success"]
                }
            else:
                result = {
                    **quality_analysis,
                    "llm_analysis": "Bullet point is already strong",
                    "improved": bullet,
                    "llm_success": True
                }
            
            results.append(result)
        
        return results
    
    def get_summary_stats(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics from bullet analysis results.
        
        Args:
            analysis_results: List of analysis result dictionaries
            
        Returns:
            Summary statistics dictionary
        """
        if not analysis_results:
            return {
                "total_bullets": 0,
                "weak_bullets": 0,
                "average_score": 0,
                "improvement_rate": 0
            }
        
        total = len(analysis_results)
        weak_count = sum(1 for r in analysis_results if r.get("needs_improvement", False))
        avg_score = sum(r.get("score", 0) for r in analysis_results) / total
        
        return {
            "total_bullets": total,
            "weak_bullets": weak_count,
            "average_score": round(avg_score, 1),
            "improvement_rate": round((weak_count / total * 100), 1) if total > 0 else 0
        }
    
    def analyze_resume_sections(
        self, 
        parsed_resume: Dict[str, Any],
        job_description: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze all bullet points across all resume sections.
        
        Args:
            parsed_resume: Parsed resume dictionary from ResumeParser
            job_description: Optional JD for context
            
        Returns:
            Complete analysis with suggestions for all sections
        """
        results = {
            "sections": {},
            "overall_stats": {}
        }
        
        all_analyses = []
        
        # Process each section that has bullets
        sections = parsed_resume.get("sections", {})
        
        for section_name, section_data in sections.items():
            bullets = section_data.get("bullets", [])
            
            if bullets:
                logger.info(f"Analyzing {len(bullets)} bullets in {section_name} section")
                
                # Generate suggestions for this section
                section_results = self.generate_rewrite_suggestions(
                    bullets, 
                    job_description
                )
                
                results["sections"][section_name] = {
                    "bullets": section_results,
                    "stats": self.get_summary_stats(section_results)
                }
                
                all_analyses.extend(section_results)
        
        # Calculate overall stats
        results["overall_stats"] = self.get_summary_stats(all_analyses)
        
        return results
    
    def quick_analyze(self, bullet: str) -> Tuple[int, List[str]]:
        """
        Quick quality check without LLM (for fast initial screening).
        
        Args:
            bullet: Bullet point text
            
        Returns:
            Tuple of (score, list of issues)
        """
        analysis = self.analyze_bullet_quality(bullet)
        return analysis["score"], analysis["issues"]


# Module-level convenience function
def create_rewrite_engine() -> RewriteEngine:
    """Create a new RewriteEngine instance."""
    return RewriteEngine()
