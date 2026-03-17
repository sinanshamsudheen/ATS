"""
ATS report generation using Groq + Llama3-8B.

Fast cloud-based inference as an alternative to local Phi-3 model.
"""

import json
import logging
import os
import re
import time
from typing import Dict, Any, Optional

from ..config import GROQ_API_KEY, GROQ_MODEL
from ._prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = SYSTEM_PROMPT

_USER_PROMPT = USER_PROMPT_TEMPLATE + """
{{
    "ats_score": <number 0-100>,
    "score_breakdown": {{
        "keyword_coverage": <number 0-100>,
        "bullet_quality": <number 0-100>,
        "formatting": <number 0-100>,
        "structure": <number 0-100>
    }},
    "matched_skills": ["skill1", "skill2", ...],
    "missing_skills": ["skill1", "skill2", ...],
    "weak_bullets": [
        {{
            "original": "exact verbatim bullet from the resume above",
            "issue": "specific problem (vague verb / no metric / passive voice)",
            "improved": "rewritten bullet with action verb + metric + impact"
        }}
    ],
    "formatting_issues": ["issue1", "issue2", ...],
    "overall_feedback": "2-3 sentence actionable summary"
}}"""

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 1.0


class GroqInference:
    """Groq client for ATS report generation."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not found. Set it in .env or environment variables.")
        
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
            self._initialized = True
            logger.info(f"Groq client initialized (model: {GROQ_MODEL})")
        except ImportError:
            raise RuntimeError("groq package not installed. Run: pip install groq")
    
    def generate(
        self,
        resume_text: str,
        job_description: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Generate ATS evaluation report using Groq."""
        user_prompt = _USER_PROMPT.format(
            resume_text=resume_text,
            job_description=job_description,
        )
        return self._call_with_retry(_SYSTEM_PROMPT, user_prompt, max_tokens, temperature)
    
    def _call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Dict[str, Any]:
        """Call Groq API with retry logic for rate limits."""
        from groq import RateLimitError, APIError
        
        delay = RETRY_DELAY
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                
                raw_output = response.choices[0].message.content
                return self._parse_response(raw_output)
            
            except RateLimitError as e:
                last_error = e
                logger.warning(f"Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
                delay *= 2
            
            except APIError as e:
                last_error = e
                logger.warning(f"Groq API error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= 2
        
        raise RuntimeError(f"Groq API failed after {MAX_RETRIES} retries: {last_error}")
    
    def _parse_response(self, raw_output: str) -> Dict[str, Any]:
        """Parse and validate the JSON response."""
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', raw_output)
        if json_match:
            try:
                result = json.loads(json_match.group())
                if self._validate_output(result):
                    result["valid_json"] = True
                    return result
                else:
                    logger.warning("JSON output missing required ATS fields")
                    return {"raw_output": raw_output, "parsed": result, "valid_json": False}
            except json.JSONDecodeError:
                pass
        
        logger.warning("Could not extract valid JSON from Groq response")
        return {"raw_output": raw_output, "valid_json": False}
    
    def _validate_output(self, output: dict) -> bool:
        """Return True if output contains all required ATS fields."""
        required = {"ats_score", "score_breakdown", "matched_skills",
                    "missing_skills", "weak_bullets", "formatting_issues",
                    "overall_feedback"}
        if not required.issubset(output.keys()):
            return False
        if not isinstance(output["ats_score"], (int, float)):
            return False
        if not (0 <= output["ats_score"] <= 100):
            return False
        return True


def generate_with_groq(resume_text: str, job_description: str) -> Dict[str, Any]:
    """Convenience function for generating ATS report with Groq."""
    client = GroqInference()
    return client.generate(resume_text, job_description)


def generate_bullet_improvements(resume_text: str, job_description: str) -> list:
    """Use Groq to generate weak bullet improvements only.

    Called as a supplement when Phi-3 GGUF handles the main analysis but
    lacks the context window to produce quality bullet rewrites.
    Returns a list of {"original", "issue", "improved"} dicts, or [].
    """
    if not is_groq_available():
        return []

    system = (
        "You are a resume bullet point optimizer. Given a resume and job description, "
        "identify the 3-5 weakest bullet points and rewrite them with action verbs, "
        "metrics, and impact. Respond with ONLY a JSON array of objects, each with "
        '"original" (verbatim from resume), "issue" (specific problem), and '
        '"improved" (rewritten bullet). No markdown, no prose.'
    )
    user = f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{job_description}"

    try:
        client = GroqInference()
        from groq import RateLimitError, APIError
        response = client._client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=1024,
            temperature=0.1,
        )
        raw = response.choices[0].message.content
        # Extract JSON array from response
        match = re.search(r'\[[\s\S]*\]', raw)
        if match:
            bullets = json.loads(match.group())
            if isinstance(bullets, list):
                return [
                    {
                        "original": b.get("original", ""),
                        "issue": b.get("issue", ""),
                        "improved": b.get("improved", ""),
                    }
                    for b in bullets
                    if isinstance(b, dict) and b.get("original", "").strip()
                ]
    except Exception as e:
        logger.warning(f"Groq bullet improvement call failed: {e}")

    return []


def is_groq_available() -> bool:
    """Check if Groq is configured and available."""
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not api_key:
        return False
    try:
        import groq
        return True
    except ImportError:
        return False
