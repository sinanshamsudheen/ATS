"""Thin entry-point: fine-tune the Phi model with LoRA."""
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.lora_training import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ATS Phi model with LoRA")
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_training(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
    )
