#!/usr/bin/env python3
"""
Run PROVE pipeline on a single example from NLVR2 or GQA.

Usage:
    python run_example.py                                          # random NLVR2 example
    python run_example.py --identifier test1-366-0-0               # specific NLVR2 example
    python run_example.py --dataset gqa                            # random GQA example
    python run_example.py --dataset gqa --identifier 201307251     # specific GQA example
    python run_example.py --save-logs                              # save detailed logs
"""

import argparse
import json
import os
import random

from src import PROVE


# Default data paths (set PROVE_DATA_ROOT env var or use --data-file/--img-dir)
_DATA_ROOT = os.environ.get("PROVE_DATA_ROOT", "")
NLVR2_DATA = f"{_DATA_ROOT}/nlvr2_data/balanced_test1.json"
NLVR2_IMAGES = f"{_DATA_ROOT}/nlvr2_data/images"
GQA_DATA = f"{_DATA_ROOT}/gqa_data/testdev_balanced_yn.json"
GQA_IMAGES = f"{_DATA_ROOT}/gqa_data/images"


def load_nlvr2_examples(data_file, img_dir):
    """Load NLVR2 examples where both images exist."""
    examples = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            ident = ex["identifier"]
            prefix = ident.rsplit("-", 1)[0]
            img0 = os.path.join(img_dir, f"{prefix}-img0.png")
            img1 = os.path.join(img_dir, f"{prefix}-img1.png")
            if os.path.exists(img0) and os.path.exists(img1):
                examples.append({
                    "identifier": ident,
                    "question": ex["sentence"],
                    "label": ex["label"] == "True" if isinstance(ex["label"], str) else bool(ex["label"]),
                    "image_paths": {"image_a": img0, "image_b": img1},
                })
    return examples


def load_gqa_examples(data_file, img_dir):
    """Load GQA yes/no examples where image exists."""
    with open(data_file) as f:
        data = json.load(f)
    examples = []
    for qid, entry in data.items():
        img_id = entry["imageId"]
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        answer = entry["answer"].strip().lower()
        if answer not in ("yes", "no"):
            continue
        examples.append({
            "identifier": qid,
            "question": entry["question"],
            "label": answer == "yes",
            "image_paths": {"image_a": img_path},
        })
    return examples


def main():
    parser = argparse.ArgumentParser(description="Run PROVE on a single example")
    parser.add_argument("--dataset", choices=["nlvr2", "gqa"], default="nlvr2",
                        help="Dataset to use (default: nlvr2)")
    parser.add_argument("--identifier", type=str, help="Specific example identifier")
    parser.add_argument("--save-logs", action="store_true", help="Save detailed logs")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold (default: 0.5)")
    parser.add_argument("--data-file", type=str, help="Override default data file path")
    parser.add_argument("--img-dir", type=str, help="Override default image directory")
    args = parser.parse_args()

    # Set paths
    if args.dataset == "gqa":
        data_file = args.data_file or GQA_DATA
        img_dir = args.img_dir or GQA_IMAGES
    else:
        data_file = args.data_file or NLVR2_DATA
        img_dir = args.img_dir or NLVR2_IMAGES

    # Load examples
    print(f"Loading {args.dataset.upper()} examples from {data_file}...")
    if args.dataset == "gqa":
        examples = load_gqa_examples(data_file, img_dir)
    else:
        examples = load_nlvr2_examples(data_file, img_dir)
    print(f"Found {len(examples)} valid examples\n")

    if not examples:
        print("No valid examples found. Check data and image paths.")
        return

    # Select example
    if args.identifier:
        example = next((e for e in examples if e["identifier"] == args.identifier), None)
        if not example:
            print(f"Example '{args.identifier}' not found")
            return
    else:
        example = random.choice(examples)

    # Print info
    print("=" * 60)
    print(f"Dataset:      {args.dataset.upper()}")
    print(f"Example:      {example['identifier']}")
    print(f"Question:     {example['question']}")
    print(f"Ground Truth: {example['label']}")
    print(f"Images:       {list(example['image_paths'].keys())}")
    print("=" * 60)

    # Run PROVE
    model = PROVE(threshold=args.threshold)

    try:
        result = model.predict_with_details(
            image_paths=example["image_paths"],
            question=example["question"],
            save_logs=args.save_logs,
            log_dir="logs"
        )

        prob_correct = (result.probabilistic.final_answer == example["label"])
        det_correct = (result.deterministic.final_answer == example["label"])

        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)
        print(f"Ground Truth:    {example['label']}")
        print(f"Probabilistic:   {result.probabilistic.final_answer} "
              f"({'CORRECT' if prob_correct else 'WRONG'})")
        print(f"Deterministic:   {result.deterministic.final_answer} "
              f"({'CORRECT' if det_correct else 'WRONG'})")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
