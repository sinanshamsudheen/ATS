"""Merge the LoRA adapter into the base Phi-3 model and convert to GGUF.

Usage
-----
    python scripts/merge_and_convert.py                 # default Q8_0
    python scripts/merge_and_convert.py --quant f16     # keep F16
    python scripts/merge_and_convert.py --keep-merged   # don't delete HF dir
"""

import argparse
import gc
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MERGED_DIR = ROOT / "models" / "phi3-ats-merged"
GGUF_DIR = ROOT / "models"
LLAMA_CPP_DIR = ROOT / "vendor" / "llama.cpp"


# ------------------------------------------------------------------
# Step 1 — Merge LoRA into base model
# ------------------------------------------------------------------
def merge_lora() -> None:
    import torch
    from transformers import Phi3ForCausalLM, AutoTokenizer
    from peft import PeftModel
    from src.config import BASE_MODEL_NAME, LORA_ADAPTER_PATH

    print("=" * 60)
    print("Step 1: Merging LoRA adapter into base model")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = Phi3ForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    print(f"  Loading LoRA adapter from {LORA_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
    print("  Merging weights ...")
    model = model.merge_and_unload()

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Saving merged model to {MERGED_DIR}")
    model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)

    # Free memory before conversion
    del model, tokenizer
    gc.collect()
    print("  Done.\n")


# ------------------------------------------------------------------
# Step 2 — Ensure llama.cpp repo is available
# ------------------------------------------------------------------
def ensure_llama_cpp() -> None:
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if convert_script.exists():
        print("llama.cpp already present — skipping clone.\n")
        return

    print("=" * 60)
    print("Step 2: Shallow-cloning llama.cpp")
    print("=" * 60)
    LLAMA_CPP_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git", "clone", "--depth", "1",
            "https://github.com/ggml-org/llama.cpp",
            str(LLAMA_CPP_DIR),
        ],
        check=True,
    )
    print("  Done.\n")


# ------------------------------------------------------------------
# Step 3 — Install the gguf pip package (used by the converter)
# ------------------------------------------------------------------
def ensure_gguf_package() -> None:
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("Installing gguf package ...")
        req = LLAMA_CPP_DIR / "requirements" / "requirements-convert_hf_to_gguf.txt"
        if req.exists():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req)],
                check=True,
            )
        else:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "gguf"],
                check=True,
            )


# ------------------------------------------------------------------
# Step 4 — Convert to GGUF
# ------------------------------------------------------------------
def convert_to_gguf(quant: str) -> Path:
    print("=" * 60)
    print(f"Step 3: Converting to GGUF ({quant})")
    print("=" * 60)

    outfile = GGUF_DIR / f"phi3-ats-{quant}.gguf"
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        raise FileNotFoundError(f"Converter not found: {convert_script}")

    cmd = [
        sys.executable,
        str(convert_script),
        str(MERGED_DIR),
        "--outtype", quant,
        "--outfile", str(outfile),
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    size_gb = outfile.stat().st_size / (1024 ** 3)
    print(f"  GGUF saved to {outfile}  ({size_gb:.2f} GB)\n")
    return outfile


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA → GGUF")
    parser.add_argument(
        "--quant",
        default="q8_0",
        choices=["f32", "f16", "bf16", "q8_0"],
        help="Quantisation type written by the converter (default: q8_0)",
    )
    parser.add_argument(
        "--keep-merged",
        action="store_true",
        help="Keep the intermediate merged HuggingFace model directory",
    )
    args = parser.parse_args()

    merge_lora()
    ensure_llama_cpp()
    ensure_gguf_package()
    gguf_path = convert_to_gguf(args.quant)

    if not args.keep_merged and MERGED_DIR.exists():
        print("Cleaning up merged model directory ...")
        shutil.rmtree(MERGED_DIR)

    print("=" * 60)
    print(f"All done!  GGUF model → {gguf_path}")
    print("Run the app:  streamlit run src/app/streamlit_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
