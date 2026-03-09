"""Thin entry-point: prepare the ATS fine-tuning dataset."""
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.dataset_loader import prepare_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ATS dataset for fine-tuning")
    parser.add_argument("--raw", default="data/raw_dataset.json")
    parser.add_argument("--train", default="data/train.json")
    parser.add_argument("--val", default="data/validation.json")
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(
        raw_path=args.raw,
        train_path=args.train,
        val_path=args.val,
        train_split=args.split,
        seed=args.seed,
    )
