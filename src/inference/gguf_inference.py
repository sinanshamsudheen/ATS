"""
ATS report generation using a GGUF-quantized model via llama-cpp-python.

Lightweight CPU inference — replaces the heavy HuggingFace pipeline when
a GGUF file is available under models/.
"""

import json
import os
import re
from typing import Dict, Any, Optional

from llama_cpp import Llama


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_model(
    model_path: str,
    n_ctx: int = 4096,
    n_threads: int = 0,
) -> Llama:
    """Load a GGUF model.  *n_threads=0* → auto (all logical cores)."""
    if n_threads <= 0:
        n_threads = os.cpu_count() or 4

    print(f"Loading GGUF model: {model_path}  (threads: {n_threads})")
    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        verbose=False,
    )
    print("GGUF model loaded successfully")
    return model


# ------------------------------------------------------------------
# Prompt / parsing (mirrors groq_inference format)
# ------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert ATS (Applicant Tracking System) analyzer. "
    "Evaluate resumes against job descriptions and provide detailed "
    "compliance analysis.  Always respond with valid JSON only, no "
    "additional text."
)

_USER_PROMPT = """\
Analyze this resume against the job description and return a JSON evaluation.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Return ONLY a JSON object with this exact structure:
{{
    "ats_score": <number 0-100>,
    "score_breakdown": {{
        "keyword_coverage": <number 0-100>,
        "bullet_quality": <number 0-100>,
        "formatting": <number 0-100>,
        "structure": <number 0-100>
    }},
    "matched_skills": ["skill1", "skill2"],
    "missing_skills": ["skill1", "skill2"],
    "weak_bullets": [
        {{
            "original": "bullet text",
            "issue": "what's wrong",
            "improved": "better version"
        }}
    ],
    "formatting_issues": ["issue1", "issue2"],
    "overall_feedback": "summary feedback"
}}"""


def _extract_json(text: str) -> Optional[dict]:
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _validate(output: dict) -> bool:
    required = {
        "ats_score", "score_breakdown", "matched_skills",
        "missing_skills", "weak_bullets", "formatting_issues",
        "overall_feedback",
    }
    if not required.issubset(output.keys()):
        return False
    if not isinstance(output["ats_score"], (int, float)):
        return False
    if not (0 <= output["ats_score"] <= 100):
        return False
    breakdown_keys = {"keyword_coverage", "bullet_quality", "formatting", "structure"}
    if not breakdown_keys.issubset(output.get("score_breakdown", {}).keys()):
        return False
    return True


# ------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------

def generate(
    model: Llama,
    resume_text: str,
    job_description: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """Run chat-completion on the GGUF model and return the ATS dict."""

    user_msg = _USER_PROMPT.format(
        resume_text=resume_text,
        job_description=job_description,
    )

    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.1,
    )

    generated_text = response["choices"][0]["message"]["content"]

    result = _extract_json(generated_text)
    if result is None:
        print("WARNING: Could not extract valid JSON from GGUF output")
        print(f"Raw output:\n{generated_text[:500]}")
        return {"raw_output": generated_text, "valid_json": False}

    if not _validate(result):
        print("WARNING: JSON output missing required ATS fields")
        return {"raw_output": generated_text, "parsed": result, "valid_json": False}

    result["valid_json"] = True
    return result
