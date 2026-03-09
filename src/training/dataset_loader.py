"""
Dataset loader and preparation utilities for ATS fine-tuning.

Provides importable functions used by ``scripts/prepare_dataset.py``.
"""

import json
import os
import random
import sys
from typing import List, Tuple

import yaml


def load_config(config_path: str = "configs/training_config.yaml") -> dict:
    """Load training configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_sample(sample: dict) -> bool:
    """Return *True* if *sample* has the correct ATS structure."""
    required_keys = {"instruction", "input", "output"}
    if not required_keys.issubset(sample.keys()):
        return False

    try:
        output = json.loads(sample["output"])
    except (json.JSONDecodeError, TypeError):
        return False

    required_output_keys = {
        "ats_score", "score_breakdown", "matched_skills",
        "missing_skills", "weak_bullets", "formatting_issues",
        "overall_feedback",
    }
    if not required_output_keys.issubset(output.keys()):
        return False

    if not isinstance(output["ats_score"], (int, float)):
        return False
    if not (0 <= output["ats_score"] <= 100):
        return False

    return True


def format_prompt(sample: dict, eos_token: str = "<|endoftext|>") -> str:
    """Format a single sample into the instruction-tuning prompt."""
    return (
        f"### Instruction:\n{sample['instruction']}\n\n"
        f"### Input:\n{sample['input']}\n\n"
        f"### Response:\n{sample['output']}{eos_token}"
    )


def prepare_dataset(
    raw_path: str = "data/raw_dataset.json",
    train_path: str = "data/train.json",
    val_path: str = "data/validation.json",
    train_split: float = 0.9,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """
    Full preparation pipeline:
    1. Load raw JSON  2. Validate  3. Format prompts  4. Split  5. Save
    """
    print("=" * 60)
    print("ATS Dataset Preparation Pipeline")
    print("=" * 60)

    # 1. Load
    print(f"\n[1/5] Loading raw dataset from {raw_path}...")
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"  Loaded {len(raw_data)} samples")

    # 2. Validate
    print("\n[2/5] Validating JSON structure...")
    valid_samples: List[dict] = []
    invalid_count = 0
    for i, sample in enumerate(raw_data):
        if validate_sample(sample):
            valid_samples.append(sample)
        else:
            invalid_count += 1
            print(f"  WARNING: Sample {i} is malformed, skipping")
    print(f"  Valid: {len(valid_samples)} | Invalid: {invalid_count}")

    if not valid_samples:
        print("ERROR: No valid samples found. Aborting.")
        sys.exit(1)

    # 3. Format prompts
    print("\n[3/5] Formatting prompts...")
    for sample in valid_samples:
        sample["text"] = format_prompt(sample)
    print(f"  Formatted {len(valid_samples)} prompts")

    # 4. Split
    print(f"\n[4/5] Splitting dataset ({train_split:.0%} / {1 - train_split:.0%})...")
    random.seed(seed)
    random.shuffle(valid_samples)
    split_idx = int(len(valid_samples) * train_split)
    train_data = valid_samples[:split_idx]
    val_data = valid_samples[split_idx:]
    print(f"  Train: {len(train_data)}  |  Validation: {len(val_data)}")

    # 5. Save
    print("\n[5/5] Saving processed datasets...")
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print(f"  Train: {len(train_data)}  |  Validation: {len(val_data)}")
    print("=" * 60)

    return train_data, val_data
