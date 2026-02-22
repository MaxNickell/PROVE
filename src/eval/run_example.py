#!/usr/bin/env python3
"""
Run PROVE pipeline on a single example.

Usage:
    python run_example.py                                          # random test1 example
    python run_example.py --identifier test1-366-0-0               # specific NLVR2 example
    python run_example.py --dataset gqa                            # random GQA example
    python run_example.py --dataset gqa --identifier 201307251     # specific GQA example
    python run_example.py --dataset vqav2                          # random VQAv2 example
    python run_example.py --save-logs                              # save detailed logs
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

from src import PROVE
from src.eval.run_eval import DATASET_PRESETS, nlvr2_id_to_image_paths, load_single_image_samples


def load_paired_examples(data_file, img_dir, z_filter=0):
    """Load NLVR2-style paired-image examples where both images exist."""
    examples = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            ident = ex["identifier"]
            if z_filter is not None and not ident.endswith(f'-{z_filter}'):
                continue
            img_a, img_b = nlvr2_id_to_image_paths(ident, img_dir, directory=ex.get("directory"))
            if os.path.exists(img_a) and os.path.exists(img_b):
                examples.append({
                    "identifier": ident,
                    "question": ex["sentence"],
                    "label": ex["label"] == "True" if isinstance(ex["label"], str) else bool(ex["label"]),
                    "image_paths": {"image_a": img_a, "image_b": img_b},
                })
    return examples


def load_single_examples(data_file, img_dir):
    """Load single-image yes/no examples (GQA, VQAv2, etc.) where image exists."""
    samples = load_single_image_samples(data_file, img_dir)
    examples = []
    for s in samples:
        if os.path.exists(s["image_path"]):
            examples.append({
                "identifier": s["identifier"],
                "question": s["sentence"],
                "label": s["label"],
                "image_paths": {"image_a": s["image_path"]},
            })
    return examples


def main():
    parser = argparse.ArgumentParser(description="Run PROVE on a single example")
    parser.add_argument("--dataset", choices=list(DATASET_PRESETS.keys()), default="test1",
                        help=f"Dataset preset (default: test1)")
    parser.add_argument("--identifier", type=str, help="Specific example identifier")
    parser.add_argument("--save-logs", action="store_true", help="Save detailed logs")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold (default: 0.5)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root directory for datasets (overrides PROVE_DATA_ROOT)")
    args = parser.parse_args()

    data_root = args.data_root or os.environ.get("PROVE_DATA_ROOT", "")
    preset = DATASET_PRESETS[args.dataset]
    data_file = os.path.join(data_root, preset["test_json"])
    img_dir = os.path.join(data_root, preset["img_dir"])

    # Load examples
    print(f"Loading {args.dataset} examples from {data_file}...")
    if preset["type"] == "single":
        examples = load_single_examples(data_file, img_dir)
    else:
        examples = load_paired_examples(data_file, img_dir, z_filter=preset["z_filter"])
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
    print(f"Dataset:      {args.dataset}")
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
