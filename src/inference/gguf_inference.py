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
# Prompt — must EXACTLY match the training format (Cell 8 of notebook):
#
#   <|user|>
#   {instruction}
#
#   {input}<|end|>
#   <|assistant|>
#   {output}<|end|>
#
# The model was trained with NO system prompt and NO schema skeleton.
# Instruction is a single sentence; input is "RESUME:\n...\n\nJOB DESCRIPTION:\n..."
# ------------------------------------------------------------------

_INSTRUCTION = (
    "Evaluate the following resume against the job description and provide a detailed "
    "ATS (Applicant Tracking System) compliance analysis. "
    "Return ONLY a raw JSON object (no markdown, no prose) with exactly these keys: "
    '{"ats_score": <0-100>, "score_breakdown": {"keyword_coverage": <0-100>, '
    '"bullet_quality": <0-100>, "formatting": <0-100>, "structure": <0-100>}, '
    '"matched_skills": [...], "missing_skills": [...], '
    '"weak_bullets": [{"original": "...", "issue": "...", "improved": "..."}], '
    '"formatting_issues": [...], "overall_feedback": "..."}'
)

# Token budget: 4096 total − ~50 instruction − ~1024 response = ~3022 left for inputs
# At ~4 chars/token: ~12 000 chars total; split 60/40 resume/JD.
_MAX_RESUME_CHARS = 4000
_MAX_JD_CHARS     = 2800


def _extract_json(text: str) -> Optional[dict]:
    # 1. Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    # 2. Direct parse
    try:
        return _normalize(json.loads(text))
    except json.JSONDecodeError:
        pass

    # 3. Extract outermost {...} then parse
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return _normalize(json.loads(match.group()))
        except json.JSONDecodeError:
            pass

    # 4. Repair truncated / malformed JSON (model cuts off before final })
    try:
        from json_repair import repair_json
        repaired = repair_json(text)
        result = json.loads(repaired)
        if isinstance(result, dict):
            return _normalize(result)
    except Exception:
        pass

    return None


def _normalize(result: dict) -> dict:
    """Coerce model output variants into the expected schema shape and fill defaults."""
    # matched_skills / missing_skills: list of dicts → list of strings
    for key in ("matched_skills", "missing_skills"):
        items = result.get(key, [])
        if items and isinstance(items[0], dict):
            result[key] = [
                item.get("name") or item.get("skill") or next(iter(item.values()), "")
                for item in items
            ]
        elif not isinstance(items, list):
            result[key] = []

    # overall_feedback: dict → string
    feedback = result.get("overall_feedback", "")
    if isinstance(feedback, dict):
        result["overall_feedback"] = (
            feedback.get("description")
            or feedback.get("summary")
            or " ".join(str(v) for v in feedback.values() if v)
        )

    # weak_bullets: ensure each item has original / issue / improved keys
    bullets = result.get("weak_bullets", [])
    normalised = []
    for b in bullets:
        if isinstance(b, str):
            normalised.append({"original": b, "issue": "", "improved": b})
        elif isinstance(b, dict):
            normalised.append({
                "original": b.get("original") or b.get("bullet") or "",
                "issue":    b.get("issue")    or b.get("problem") or "",
                "improved": b.get("improved") or b.get("rewrite") or b.get("improvement") or "",
            })
    result["weak_bullets"] = normalised

    # score_breakdown: fill any missing sub-scores with 0
    bd = result.get("score_breakdown", {})
    if not isinstance(bd, dict):
        bd = {}
    for k in ("keyword_coverage", "bullet_quality", "formatting", "structure"):
        bd.setdefault(k, 0)
    result["score_breakdown"] = bd

    # ensure all top-level list/string fields exist
    result.setdefault("formatting_issues", [])
    result.setdefault("overall_feedback", "")

    result["valid_json"] = True
    return result


def _validate(output: dict) -> bool:
    """Only require ats_score — all other fields have defaults in _normalize."""
    score = output.get("ats_score")
    return isinstance(score, (int, float)) and 0 <= score <= 100


# ------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------

def generate(
    model: Llama,
    resume_text: str,
    job_description: str,
    max_tokens: int = 1500,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """Run chat-completion on the GGUF model and return the ATS dict."""

    # Truncate inputs to stay within the 4096-token context window
    if len(resume_text) > _MAX_RESUME_CHARS:
        resume_text = resume_text[:_MAX_RESUME_CHARS] + "\n[truncated]"
    if len(job_description) > _MAX_JD_CHARS:
        job_description = job_description[:_MAX_JD_CHARS] + "\n[truncated]"

    input_block = f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{job_description}"

    # Exact Phi-3 training format (Cell 8 of ats_fine_tuning_pipeline.ipynb):
    #   <|user|>\n{instruction}\n\n{input}<|end|>\n<|assistant|>\n
    # No system prompt, no schema skeleton — the fine-tuned model was never
    # trained with those, so including them causes garbage output.
    prompt = (
        f"<|user|>\n{_INSTRUCTION}\n\n{input_block}<|end|>\n"
        f"<|assistant|>\n"
    )

    response = model(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["<|end|>", "<|user|>"],
    )

    generated_text = response["choices"][0]["text"]

    result = _extract_json(generated_text)
    if result is None:
        print("WARNING: Could not extract valid JSON from GGUF output")
        print(f"Raw output:\n{generated_text[:500]}")
        return {"raw_output": generated_text, "valid_json": False}

    if not _validate(result):
        print("WARNING: JSON output missing required ATS fields — ats_score absent or out of range")
        return {"raw_output": generated_text, "parsed": result, "valid_json": False}

    return result  # valid_json=True already set by _normalize
