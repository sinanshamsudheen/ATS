"""
LLM Inference Module for ATS Resume Optimizer
Uses OpenAI API (GPT-4o-mini) for generating resume rewrite suggestions
"""

import openai
import logging
from typing import Optional, Dict, Any
from ..config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)

class LLMInference:
    """
    Lightweight wrapper for OpenAI API to generate resume bullet improvements.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMInference, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize OpenAI client."""
        if not hasattr(self, '_initialized'):
            self._initialize_client()
            self._initialized = True
    
    def _initialize_client(self):
        """Set up OpenAI API client."""
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key not found. Set OPENAI_API_KEY in config or environment.")
            raise ValueError("OPENAI_API_KEY is required for LLM inference")
        
        openai.api_key = OPENAI_API_KEY
        logger.info(f"OpenAI client initialized with model: {OPENAI_MODEL}")
    
    def generate(self, messages: list, max_tokens: int = 512, temperature: float = None) -> str:
        """
        Generate completion using OpenAI API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (uses config default if None)
            
        Returns:
            Generated text string
        """
        try:
            if temperature is None:
                temperature = LLM_TEMPERATURE
            
            response = openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"Generation failed: {e}")
    
    def analyze_bullet(self, bullet: str, job_description: str = "") -> Dict[str, Any]:
        """
        Analyze a resume bullet point and suggest improvements using OpenAI.
        
        Args:
            bullet: The bullet point text to analyze
            job_description: Optional JD context for tailored suggestions
            
        Returns:
            Dictionary with analysis and suggestions
        """
        system_message = """You are an expert ATS (Applicant Tracking System) resume optimizer. 
Your job is to analyze resume bullet points and provide specific, actionable improvements.

Strong bullets should have:
1. Strong action verbs (Developed, Engineered, Implemented, etc.)
2. Quantifiable metrics/results (%, numbers, $, time saved)
3. Specific and concrete details (not generic)
4. Concise length (15-25 words ideal)
5. Alignment with job requirements when provided

IMPORTANT: If the bullet is already strong (has metrics, strong verb, specific), return it UNCHANGED or with only MINOR refinements. Don't rewrite bullets that are already excellent."""

        user_message = f"""Analyze this resume bullet point:

Original: {bullet}
"""
        if job_description:
            user_message += f"\nJob Context: {job_description[:300]}...\n"
        
        user_message += """
Provide your response in this format:
ANALYSIS: [Brief assessment - if already strong, say "Already strong" and explain why]
IMPROVED: [The rewritten bullet OR the original if already excellent]

Remember: Only make significant changes if there are clear weaknesses. Preserve strong bullets."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = self.generate(messages, max_tokens=300)
            
            # Parse response
            analysis = ""
            improved = ""
            
            if "ANALYSIS:" in response:
                parts = response.split("IMPROVED:")
                analysis = parts[0].replace("ANALYSIS:", "").strip()
                if len(parts) > 1:
                    improved = parts[1].strip()
            else:
                # Fallback if format not followed
                improved = response.strip()
            
            return {
                "original": bullet,
                "analysis": analysis,
                "improved": improved,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error analyzing bullet: {e}")
            return {
                "original": bullet,
                "analysis": f"Error: {str(e)}",
                "improved": bullet,
                "success": False
            }
    
    def batch_analyze_bullets(self, bullets: list, job_description: str = "") -> list:
        """
        Analyze multiple bullet points.
        
        Args:
            bullets: List of bullet point strings
            job_description: Optional JD context
            
        Returns:
            List of analysis dictionaries
        """
        results = []
        for bullet in bullets:
            result = self.analyze_bullet(bullet, job_description)
            results.append(result)
        
        return results
    
    def is_loaded(self) -> bool:
        """Check if API client is initialized."""
        return hasattr(self, '_initialized') and self._initialized


# Convenience function for module-level access
_llm_instance = None

def get_llm() -> LLMInference:
    """Get or create the singleton LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMInference()
    return _llm_instance
