#!/usr/bin/env python3
"""
NLVR2 Evaluation Script for PROVE

Evaluates PROVE on the NLVR2 benchmark with robust checkpointing.
Outputs predictions in official CSV format for metrics.py evaluation.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

from src import PROVE


# =============================================================================
# Configuration
# =============================================================================

DATA_FILE_MAP = {
    "dev": "data/nlvr/nlvr2/data/dev.json",
    "test1": "data/nlvr/nlvr2/data/test1.json",
    "test2": "data/nlvr/nlvr2/data/test2.json",
    "balanced_dev": "data/nlvr/nlvr2/data/balanced/balanced_dev.json",
    "balanced_test1": "data/nlvr/nlvr2/data/balanced/balanced_test1.json",
}

IMAGE_DIR = "data/nlvr/nlvr2/images"
CHECKPOINT_FILE = "nlvr2_checkpoint.json"


# =============================================================================
# Helper Functions
# =============================================================================

def load_nlvr2_data(split: str) -> List[Dict[str, Any]]:
    """Load NLVR2 data from JSON file."""
    data_path = DATA_FILE_MAP[split]
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, 'r') as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(examples)} examples from {split}")
    return examples


def get_image_path(identifier: str, img_idx: int) -> str:
    """
    Convert identifier to image path.

    identifier: "test1-0-1-0" → "test1-0-1-img0.png"
    img_idx: 0 for left image, 1 for right image
    """
    parts = identifier.rsplit('-', 1)  # Split off sentence ID
    base = parts[0]  # "test1-0-1"
    return f"{IMAGE_DIR}/{base}-img{img_idx}.png"


def check_images_valid(examples: List[Dict], skip_invalid: bool = True) -> List[Dict]:
    """
    Check which examples have both images available AND valid (not corrupted).

    Args:
        examples: List of NLVR2 examples
        skip_invalid: If True, return only examples with valid images

    Returns:
        List of examples with valid images (if skip_invalid=True)
    """
    from PIL import Image

    available = []
    missing_count = 0
    corrupted_count = 0
    corrupted_ids = []

    print("Validating images...")
    for example in examples:
        left_img = get_image_path(example['identifier'], 0)
        right_img = get_image_path(example['identifier'], 1)

        # Check existence
        left_exists = Path(left_img).exists()
        right_exists = Path(right_img).exists()

        if not (left_exists and right_exists):
            missing_count += 1
            continue

        # Check validity (can be loaded by PIL)
        left_valid = False
        right_valid = False

        try:
            with Image.open(left_img) as img:
                img.verify()  # Verify without loading full image
            left_valid = True
        except Exception:
            pass

        try:
            with Image.open(right_img) as img:
                img.verify()
            right_valid = True
        except Exception:
            pass

        if left_valid and right_valid:
            available.append(example)
        else:
            corrupted_count += 1
            corrupted_ids.append({
                'identifier': example['identifier'],
                'left_img': left_img,
                'right_img': right_img,
                'left_valid': left_valid,
                'right_valid': right_valid
            })
            if not skip_invalid:
                print(f"Warning: Corrupted images for {example['identifier']} "
                      f"(left: {left_valid}, right: {right_valid})")

    # Summary
    if missing_count > 0:
        print(f"⚠  {missing_count} examples have missing images")

    if corrupted_count > 0:
        print(f"⚠  {corrupted_count} examples have corrupted/invalid images")

        # Save corrupted image list for debugging
        corrupted_file = "corrupted_images.json"
        with open(corrupted_file, 'w') as f:
            json.dump(corrupted_ids, f, indent=2)
        print(f"   Corrupted image list saved to: {corrupted_file}")

    if skip_invalid:
        print(f"✓  {len(available)} valid examples ready")

    return available if skip_invalid else examples


def load_checkpoint(filepath: str) -> Optional[Dict]:
    """Load checkpoint JSON if exists."""
    if not Path(filepath).exists():
        return None

    try:
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        print(f"Loaded checkpoint: {checkpoint['completed']}/{checkpoint['total_examples']} examples completed")
        return checkpoint
    except Exception as e:
        print(f"Warning: Failed to load checkpoint: {e}")
        return None


def save_checkpoint(
    split: str,
    total_examples: int,
    completed: int,
    predictions: List[Dict],
    failed_examples: List[Dict],
    filepath: str
):
    """Save checkpoint JSON with atomic write."""
    checkpoint = {
        'split': split,
        'total_examples': total_examples,
        'completed': completed,
        'predictions': predictions,
        'failed_examples': failed_examples,
        'timestamp': datetime.now().isoformat()
    }

    # Atomic write: write to temp file, then rename
    temp_file = filepath + '.tmp'
    with open(temp_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    os.rename(temp_file, filepath)


def save_predictions_csv(predictions: List[Dict], output_path: str):
    """Save predictions in official CSV format: identifier,prediction"""
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(f"{pred['identifier']},{pred['prediction']}\n")
    print(f"Saved predictions to {output_path}")


def save_detailed_results(predictions: List[Dict], failed_examples: List[Dict], output_path: str):
    """Save detailed results as JSON."""
    results = {
        'total_predictions': len(predictions),
        'total_failed': len(failed_examples),
        'predictions': predictions,
        'failed_examples': failed_examples,
        'timestamp': datetime.now().isoformat()
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved detailed results to {output_path}")


def run_official_metrics(csv_path: str, data_path: str):
    """Run official metrics.py evaluation script."""
    metrics_script = "data/nlvr/nlvr2/eval/metrics.py"

    if not Path(metrics_script).exists():
        print(f"Warning: Official metrics script not found at {metrics_script}")
        return

    print(f"\n{'='*80}")
    print("OFFICIAL EVALUATION")
    print('='*80)
    cmd = f"python {metrics_script} {csv_path} {data_path}"
    print(f"Running: {cmd}\n")
    os.system(cmd)
    print('='*80)


def generate_analysis_report(predictions: List[Dict], failed_examples: List[Dict], output_path: str):
    """Generate detailed analysis report."""
    report = []
    report.append("="*80)
    report.append("NLVR2 EVALUATION ANALYSIS")
    report.append("="*80)
    report.append("")

    # Basic stats
    total = len(predictions)
    failed = len(failed_examples)
    correct = sum(1 for p in predictions if p['prediction'] == p['ground_truth'])
    accuracy = correct / total if total > 0 else 0.0

    report.append(f"Total examples processed: {total}")
    report.append(f"Failed examples: {failed}")
    report.append(f"Correct predictions: {correct}")
    report.append(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    report.append("")

    # Label-wise breakdown
    true_preds = [p for p in predictions if p['ground_truth'] == 'True']
    false_preds = [p for p in predictions if p['ground_truth'] == 'False']

    if true_preds:
        true_correct = sum(1 for p in true_preds if p['prediction'] == 'True')
        true_acc = true_correct / len(true_preds)
        report.append(f"True examples: {len(true_preds)} (acc: {true_acc:.4f})")

    if false_preds:
        false_correct = sum(1 for p in false_preds if p['prediction'] == 'False')
        false_acc = false_correct / len(false_preds)
        report.append(f"False examples: {len(false_preds)} (acc: {false_acc:.4f})")

    report.append("")

    # Timing stats
    if predictions:
        times = [p['time'] for p in predictions]
        avg_time = sum(times) / len(times)
        total_time = sum(times)
        report.append(f"Average time per example: {avg_time:.2f}s")
        report.append(f"Total time: {total_time/3600:.2f}h ({total_time/60:.1f}m)")
        report.append("")

    # Confusion matrix
    tp = sum(1 for p in predictions if p['prediction'] == 'True' and p['ground_truth'] == 'True')
    fp = sum(1 for p in predictions if p['prediction'] == 'True' and p['ground_truth'] == 'False')
    tn = sum(1 for p in predictions if p['prediction'] == 'False' and p['ground_truth'] == 'False')
    fn = sum(1 for p in predictions if p['prediction'] == 'False' and p['ground_truth'] == 'True')

    report.append("Confusion Matrix:")
    report.append("              Predicted")
    report.append("              True  False")
    report.append(f"Actual True   {tp:4d}  {fn:4d}")
    report.append(f"       False  {fp:4d}  {tn:4d}")
    report.append("")

    # Failed examples summary
    if failed_examples:
        report.append(f"Failed Examples: {len(failed_examples)}")
        error_types = {}
        for ex in failed_examples:
            error = ex['error']
            error_types[error] = error_types.get(error, 0) + 1

        for error, count in sorted(error_types.items(), key=lambda x: -x[1]):
            report.append(f"  {error}: {count}")

    report.append("="*80)

    # Write report
    report_text = '\n'.join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)

    print("\n" + report_text)


# =============================================================================
# Main Evaluation Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate PROVE on NLVR2 benchmark with checkpointing'
    )
    parser.add_argument(
        '--split',
        default='balanced_test1',
        choices=['dev', 'test1', 'test2', 'balanced_dev', 'balanced_test1'],
        help='Which split to evaluate on (default: balanced_test1)'
    )
    parser.add_argument(
        '--num_examples',
        type=int,
        default=None,
        help='Limit number of examples (default: all)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if exists'
    )
    parser.add_argument(
        '--checkpoint_freq',
        type=int,
        default=10,
        help='Save checkpoint every N examples (default: 10)'
    )
    parser.add_argument(
        '--output_dir',
        default='nlvr2_results',
        help='Output directory for results (default: nlvr2_results)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print verbose output from PROVE (default: True)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load checkpoint if resuming
    checkpoint = None
    start_idx = 0
    predictions = []
    failed_examples = []

    if args.resume:
        checkpoint = load_checkpoint(CHECKPOINT_FILE)
        if checkpoint:
            if checkpoint['split'] != args.split:
                print(f"Warning: Checkpoint split ({checkpoint['split']}) doesn't match requested split ({args.split})")
                print("Starting fresh...")
                checkpoint = None
            else:
                start_idx = checkpoint['completed']
                predictions = checkpoint['predictions']
                failed_examples = checkpoint['failed_examples']

    # Load dataset
    print(f"\nLoading {args.split} split...")
    all_examples = load_nlvr2_data(args.split)

    # Limit examples if requested
    if args.num_examples:
        all_examples = all_examples[:args.num_examples]
        print(f"Limited to {len(all_examples)} examples")

    total_examples = len(all_examples)

    # Check images exist and are valid (not corrupted)
    print("\nValidating images...")
    all_examples = check_images_valid(all_examples, skip_invalid=True)

    if len(all_examples) < total_examples:
        print(f"\nNote: Download images using:")
        print(f"  python data/nlvr/nlvr2/util/download_images.py \\")
        print(f"    {DATA_FILE_MAP[args.split]} \\")
        print(f"    {IMAGE_DIR} \\")
        print(f"    data/nlvr/nlvr2/util/hashes/{args.split.replace('balanced_', '')}_hashes.json")

    total_examples = len(all_examples)

    # Skip already processed examples
    examples_to_process = all_examples[start_idx:]

    if not examples_to_process:
        print("\nAll examples already processed!")
        print(f"Results saved in {args.output_dir}/")
        return

    print(f"\nProcessing {len(examples_to_process)} examples (starting from {start_idx})...")
    print(f"Checkpointing every {args.checkpoint_freq} examples")
    print(f"Output directory: {args.output_dir}/")
    print()

    # Initialize PROVE model
    print("Initializing PROVE model...")
    model = PROVE(verbose=args.verbose)
    print("Model ready!\n")

    # Process examples
    start_time = time.time()

    for i, example in enumerate(examples_to_process):
        current_idx = start_idx + i
        identifier = example['identifier']

        try:
            # Get image paths
            left_img = get_image_path(identifier, 0)
            right_img = get_image_path(identifier, 1)

            # Run PROVE
            example_start = time.time()
            prediction = model.predict(left_img, right_img, example['sentence'])
            elapsed = time.time() - example_start

            # Store result
            predictions.append({
                'identifier': identifier,
                'prediction': prediction,
                'ground_truth': example['label'],
                'sentence': example['sentence'],
                'time': elapsed
            })

            # Progress update
            correct_so_far = sum(1 for p in predictions if p['prediction'] == p['ground_truth'])
            acc_so_far = correct_so_far / len(predictions)
            avg_time = (time.time() - start_time) / (i + 1)
            eta = avg_time * (len(examples_to_process) - i - 1)

            match_symbol = "✓" if prediction == example['label'] else "✗"
            print(f"[{current_idx+1}/{total_examples}] {identifier}: "
                  f"{prediction} {match_symbol} (GT: {example['label']}) "
                  f"[{elapsed:.1f}s] [Acc: {acc_so_far:.3f}] [ETA: {eta/60:.1f}m]")

            # Save checkpoint periodically
            if (i + 1) % args.checkpoint_freq == 0:
                save_checkpoint(
                    args.split,
                    total_examples,
                    current_idx + 1,
                    predictions,
                    failed_examples,
                    CHECKPOINT_FILE
                )

        except Exception as e:
            print(f"[{current_idx+1}/{total_examples}] {identifier}: FAILED - {str(e)}")
            failed_examples.append({
                'identifier': identifier,
                'sentence': example['sentence'],
                'error': str(e)
            })
            continue

    # Final checkpoint
    save_checkpoint(
        args.split,
        total_examples,
        total_examples,
        predictions,
        failed_examples,
        CHECKPOINT_FILE
    )

    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print('='*80)

    # Save outputs
    csv_path = output_dir / f"predictions_{args.split}.csv"
    save_predictions_csv(predictions, str(csv_path))

    detailed_path = output_dir / f"detailed_results_{args.split}.json"
    save_detailed_results(predictions, failed_examples, str(detailed_path))

    # Generate analysis report
    report_path = output_dir / f"analysis_{args.split}.txt"
    generate_analysis_report(predictions, failed_examples, str(report_path))

    # Run official metrics (only if we processed the full dataset)
    if args.num_examples is None:
        data_path = DATA_FILE_MAP[args.split]
        run_official_metrics(str(csv_path), data_path)
    else:
        print(f"\nSkipping official metrics (only tested on {args.num_examples} examples)")
        print(f"Run without --num_examples to get official accuracy/consistency scores")

    print(f"\nAll results saved to {args.output_dir}/")
    print(f"Checkpoint saved to {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
