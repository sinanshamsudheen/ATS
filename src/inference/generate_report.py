"""
ATS report generation using the fine-tuned Phi + LoRA model.

Provides importable functions used by ``scripts/run_inference.py``.
"""

import json
import os
import re
import warnings
from typing import Dict, Any

# Increase HuggingFace download timeout (default 10s is too short for large model shards)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

# Phi-3 uses device_map="auto" which builds some layers on meta device; when PEFT
# copies the adapter weights into those slots the operation is a no-op (the real
# copy happens via assign= semantics).  The warning is harmless – suppress it.
warnings.filterwarnings(
    "ignore",
    message="copying from a non-meta parameter in the checkpoint to a meta parameter",
    category=UserWarning,
)

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

def load_config(path: str = "configs/training_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_model(base_model_name: str, adapter_path: str, use_4bit: bool = True):
    """Load base model, attach LoRA adapter, and return (model, tokenizer)."""
    print(f"Loading base model: {base_model_name}")

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
    }
    # device_map="auto" is only needed (and safe) when running on CUDA;
    # on CPU it adds accelerate dispatch hooks that break PEFT's module
    # renaming (KeyError on 'base_model.model.model.lm_head').
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    print("Model loaded successfully")

    return model, tokenizer


# ------------------------------------------------------------------
# Prompt formatting
# ------------------------------------------------------------------

def format_prompt(instruction: str, resume_text: str, job_description: str) -> str:
    """Build an instruction-tuning prompt for inference."""
    input_text = f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{job_description}"
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )


# ------------------------------------------------------------------
# Output parsing / validation
# ------------------------------------------------------------------

def extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from model output."""
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def validate_ats_output(output: dict) -> bool:
    """Return *True* if *output* contains all required ATS fields."""
    required = {"ats_score", "score_breakdown", "matched_skills",
                "missing_skills", "weak_bullets", "formatting_issues",
                "overall_feedback"}
    if not required.issubset(output.keys()):
        return False
    if not isinstance(output["ats_score"], (int, float)):
        return False
    if not (0 <= output["ats_score"] <= 100):
        return False
    score_keys = {"keyword_coverage", "bullet_quality", "formatting", "structure"}
    if not score_keys.issubset(output.get("score_breakdown", {}).keys()):
        return False
    return True


# ------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------

_DEFAULT_INSTRUCTION = (
    "Evaluate the following resume against the job description and provide a detailed "
    "ATS (Applicant Tracking System) compliance analysis. Return a structured JSON evaluation "
    "including ATS score, score breakdown, matched skills, missing skills, weak bullet analysis "
    "with improvements, formatting issues, and overall feedback."
)


def generate(
    model,
    tokenizer,
    resume_text: str,
    job_description: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> dict:
    """Run inference and return the ATS evaluation dict."""
    prompt = format_prompt(_DEFAULT_INSTRUCTION, resume_text, job_description)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    result = extract_json(generated_text)
    if result is None:
        print("WARNING: Could not extract valid JSON from model output")
        print(f"Raw output:\n{generated_text[:500]}")
        return {"raw_output": generated_text, "valid_json": False}

    if not validate_ats_output(result):
        print("WARNING: JSON output missing required ATS fields")
        return {"raw_output": generated_text, "parsed": result, "valid_json": False}

    result["valid_json"] = True
    return result
