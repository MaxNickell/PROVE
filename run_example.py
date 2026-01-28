#!/usr/bin/env python3
"""
Run PROVE pipeline on a random NLVR2 example.

Usage:
    python run_example.py                              # random example
    python run_example.py --identifier test1-366-0-0  # specific example
    python run_example.py --save-logs                 # save detailed logs
"""

import argparse
import json
import random
from pathlib import Path

from src import PROVE


# Data paths
DATA_FILE = "data/nlvr/nlvr2/data/test1.json"
IMAGES_DIR = "data/nlvr/nlvr2/images"


def get_image_paths(identifier: str) -> tuple:
    """Convert identifier to image paths."""
    parts = identifier.rsplit("-", 1)
    prefix = parts[0]
    img0 = f"{IMAGES_DIR}/{prefix}-img0.png"
    img1 = f"{IMAGES_DIR}/{prefix}-img1.png"
    return img0, img1


def load_valid_examples() -> list:
    """Load examples where both images exist locally."""
    with open(DATA_FILE, 'r') as f:
        examples = [json.loads(line) for line in f]

    valid = []
    for ex in examples:
        img0, img1 = get_image_paths(ex["identifier"])
        if Path(img0).exists() and Path(img1).exists():
            valid.append(ex)
    return valid


def main():
    parser = argparse.ArgumentParser(description="Run PROVE on NLVR2 example")
    parser.add_argument("--identifier", type=str, help="Specific example identifier")
    parser.add_argument("--save-logs", action="store_true", help="Save detailed logs")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for deterministic mode")
    args = parser.parse_args()

    # Load examples
    print("Loading NLVR2 examples...")
    examples = load_valid_examples()
    print(f"Found {len(examples)} valid examples\n")

    # Select example
    if args.identifier:
        example = next((e for e in examples if e["identifier"] == args.identifier), None)
        if not example:
            print(f"Example {args.identifier} not found")
            return
    else:
        example = random.choice(examples)

    identifier = example["identifier"]
    question = example["sentence"]
    ground_truth = example["label"]
    img0_path, img1_path = get_image_paths(identifier)

    print("=" * 60)
    print(f"Example: {identifier}")
    print(f"Question: {question}")
    print(f"Ground Truth: {ground_truth}")
    print("=" * 60)

    # Run PROVE
    model = PROVE(threshold=args.threshold)

    try:
        result = model.predict_with_details(
            image_a_path=img0_path,
            image_b_path=img1_path,
            question=question,
            save_logs=args.save_logs,
            log_dir="logs"
        )

        # Evaluate
        prob_correct = (result.probabilistic.final_answer == ground_truth)
        det_correct = (result.deterministic.final_answer == ground_truth)

        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)
        print(f"Ground Truth: {ground_truth}")
        print(f"Probabilistic: {result.probabilistic.final_answer} ({'CORRECT' if prob_correct else 'WRONG'})")
        print(f"Deterministic: {result.deterministic.final_answer} ({'CORRECT' if det_correct else 'WRONG'})")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
