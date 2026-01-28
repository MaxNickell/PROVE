#!/usr/bin/env python3
"""
VSR Directional Spatial Reasoning Evaluation

Compares geometric reasoning (SpatialReasoner) vs VLM verification on VSR dataset.
Uses Florence-2 for object detection to get bounding boxes.

Usage:
    python eval/vsr_directional_eval.py --max-samples 100 --vlm-only
    python eval/vsr_directional_eval.py --max-samples 500 --full
"""

import os
import sys
import json
import time
import math
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import requests
from PIL import Image

# Add src to path for PROVE imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SpatialReasoner from existing module
from src.vision.spatial_reasoning import SpatialReasoner, bbox_center, bbox_size

# =============================================================================
# Relation Mappings
# =============================================================================

# VSR relations -> our geometric relation names
DIRECTIONAL_RELATIONS = {
    # Horizontal
    'left of': 'left_of',
    'to the left of': 'left_of',
    'right of': 'right_of',
    'to the right of': 'right_of',

    # Vertical
    'above': 'above',
    'over': 'above',
    'below': 'below',
    'under': 'below',
    'beneath': 'below',

    # Depth (need depth estimation)
    'in front of': 'in_front_of',
    'behind': 'behind',
    'at the back of': 'behind',

    # Topological
    'on': 'on_top_of',
    'on top of': 'on_top_of',
    'inside': 'inside',
    'within': 'inside',
}

# Pure directional only (no depth needed)
PURE_DIRECTIONAL = {'left_of', 'right_of', 'above', 'below'}

# Depth-based relations
DEPTH_RELATIONS = {'in_front_of', 'behind'}


# =============================================================================
# Entity Candidate (compatible with SpatialReasoner)
# =============================================================================

@dataclass
class EntityCandidate:
    """
    Lightweight entity reference for spatial reasoning.
    Compatible with SpatialReasoner.verify() signature.
    """
    entity_id: str
    label: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float = 1.0  # Default to 1.0 for evaluation


# =============================================================================
# VLM Verifier
# =============================================================================

class VLMVerifier:
    """Uses Qwen VLM for spatial verification."""

    def __init__(self):
        self._model = None
        self._processor = None

    def verify(self, image: Image.Image, subject: str, relation: str, obj: str) -> Tuple[float, str]:
        """
        Verify spatial relation.

        Returns:
            (probability, raw_response)
        """
        self._load_model()

        import torch

        question = f"Is the {subject} {relation} the {obj}?"
        prompt = f"{question}\n\nAnswer with ONLY 'Yes' or 'No'.\n\nAnswer:"

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(text=[text], images=[image], padding=True, return_tensors="pt")
        for k, v in inputs.items():
            if hasattr(v, 'to'):
                inputs[k] = v.to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=10, temperature=0.0,
                do_sample=False, return_dict_in_generate=True,
                output_scores=True, pad_token_id=self._processor.tokenizer.eos_token_id
            )

        input_len = inputs.input_ids.shape[1]
        response = self._processor.decode(
            outputs.sequences[0][input_len:], skip_special_tokens=True
        ).strip()

        prob = self._extract_probability(outputs.scores)
        return prob, response

    def _extract_probability(self, scores) -> float:
        """Extract P(Yes) from logits."""
        if not scores:
            return 0.5

        try:
            logits = scores[0][0]

            yes_ids = []
            no_ids = []
            for variant in ["Yes", "yes", "YES"]:
                yes_ids.extend(self._processor.tokenizer.encode(variant, add_special_tokens=False))
            for variant in ["No", "no", "NO"]:
                no_ids.extend(self._processor.tokenizer.encode(variant, add_special_tokens=False))

            yes_logits = [logits[i].item() for i in yes_ids if i < len(logits)]
            no_logits = [logits[i].item() for i in no_ids if i < len(logits)]

            if not yes_logits or not no_logits:
                return 0.5

            yes_max = max(yes_logits)
            no_max = max(no_logits)

            exp_yes = math.exp(yes_max - max(yes_max, no_max))
            exp_no = math.exp(no_max - max(yes_max, no_max))

            return exp_yes / (exp_yes + exp_no)

        except Exception as e:
            print(f"Warning: probability extraction failed: {e}")
            return 0.5

    def _load_model(self):
        """Lazy load VLM."""
        if self._model is not None:
            return

        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        print("Loading Qwen VLM...")
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="eager"
        )
        self._model.eval()


# =============================================================================
# Object Detection (for getting bboxes)
# =============================================================================

class ObjectDetector:
    """Florence-2 object detector for getting bboxes."""

    def __init__(self):
        self._model = None
        self._processor = None

    def detect_objects(self, image: Image.Image, objects: List[str]) -> Dict[str, List[float]]:
        """
        Detect specific objects and return their bboxes.

        Args:
            image: PIL Image
            objects: List of object names to detect

        Returns:
            Dict mapping object name to bbox [x1, y1, x2, y2]
        """
        self._load_model()

        import torch

        results = {}

        for obj_name in objects:
            task = "<OPEN_VOCABULARY_DETECTION>"
            prompt = f"{task}{obj_name}"

            inputs = self._processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs, max_new_tokens=1024, num_beams=3
                )

            text = self._processor.batch_decode(outputs, skip_special_tokens=False)[0]
            parsed = self._processor.post_process_generation(
                text, task=task, image_size=(image.width, image.height)
            )

            bboxes = parsed.get(task, {}).get("bboxes", [])
            if bboxes:
                # Return first (most confident) detection
                results[obj_name] = bboxes[0]

        return results

    def _load_model(self):
        """Lazy load Florence-2."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        print("Loading Florence-2...")
        model_name = "microsoft/Florence-2-large"
        self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            self._model = self._model.cuda()
        self._model.eval()


# =============================================================================
# Dataset Loading
# =============================================================================

def load_vsr_dataset(split: str = "random", subset: str = "test"):
    """Load VSR dataset from HuggingFace."""
    from datasets import load_dataset

    print(f"Loading VSR dataset (split={split}, subset={subset})...")
    dataset = load_dataset(
        f"cambridgeltl/vsr_{split}",
        data_files={"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    )
    return dataset[subset]


def filter_directional(dataset, relations: set = None):
    """Filter to directional relations only."""
    if relations is None:
        relations = set(DIRECTIONAL_RELATIONS.keys())

    def is_match(example):
        rel = example.get('relation', '').lower()
        return rel in relations

    return dataset.filter(is_match)


def download_image(image_name: str, cache_dir: str) -> Optional[str]:
    """Download COCO image."""
    path = os.path.join(cache_dir, image_name)
    if os.path.exists(path):
        return path

    os.makedirs(cache_dir, exist_ok=True)

    # Try val2017 then train2017
    for split in ["val2017", "train2017"]:
        url = f"http://images.cocodataset.org/{split}/{image_name}"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    f.write(r.content)
                return path
        except:
            continue

    return None


def parse_caption(caption: str, relation: str) -> Tuple[str, str]:
    """
    Extract subject and object from VSR caption.

    Format: "The {subject} is {relation} the {object}."
    """
    # Simple parsing - could be improved
    words = caption.split()

    # Find "is" position
    try:
        is_idx = [w.lower() for w in words].index('is')
    except ValueError:
        return None, None

    # Subject: words before "is" (skip article)
    subj_start = 1 if words[0].lower() in ['the', 'a', 'an'] else 0
    subject = ' '.join(words[subj_start:is_idx])

    # Find object after relation
    rel_words = relation.lower().split()
    rel_len = len(rel_words)

    remaining = words[is_idx + 1:]
    if len(remaining) > rel_len:
        obj_start = rel_len
        # Skip article
        if remaining[obj_start].lower() in ['the', 'a', 'an']:
            obj_start += 1
        obj = ' '.join(remaining[obj_start:]).rstrip('.')
        return subject, obj

    return None, None


# =============================================================================
# Evaluation
# =============================================================================

@dataclass
class Result:
    """Single evaluation result."""
    idx: int
    image: str
    caption: str
    relation: str
    mapped_relation: str
    label: bool
    subject: str
    obj: str

    # Detection results
    subject_bbox: List[float] = None
    obj_bbox: List[float] = None
    detection_success: bool = False

    # Geometric results
    geo_prob: float = None
    geo_pred: bool = None
    geo_correct: bool = None
    geo_time_ms: float = None

    # VLM results
    vlm_prob: float = None
    vlm_pred: bool = None
    vlm_correct: bool = None
    vlm_response: str = None
    vlm_time_ms: float = None


def run_evaluation(
        max_samples: int = 100,
        run_geometric: bool = True,
        run_vlm: bool = True,
        use_depth: bool = False,
        directional_only: bool = True,
        output_dir: str = "eval/vsr_results",
        image_cache: str = "data/coco_images"
):
    """
    Run VSR evaluation.

    Args:
        max_samples: Maximum samples to evaluate
        run_geometric: Run geometric reasoning
        run_vlm: Run VLM verification
        use_depth: Use depth estimation for geometric (ignored - SpatialReasoner handles this)
        directional_only: Only pure directional (left/right/above/below)
        output_dir: Output directory
        image_cache: Image cache directory
    """
    print("\n" + "=" * 70)
    print("VSR Directional Spatial Reasoning Evaluation")
    print("=" * 70)

    # Load dataset
    dataset = load_vsr_dataset()

    # Filter relations
    if directional_only:
        # Only left/right/above/below
        target_relations = {k for k, v in DIRECTIONAL_RELATIONS.items()
                            if v in PURE_DIRECTIONAL}
    else:
        target_relations = set(DIRECTIONAL_RELATIONS.keys())

    dataset = filter_directional(dataset, target_relations)
    print(f"Filtered to {len(dataset)} samples with target relations")

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        print(f"Limited to {max_samples} samples")

    # Initialize components
    detector = ObjectDetector() if run_geometric else None
    spatial_reasoner = SpatialReasoner(alpha=5.0) if run_geometric else None
    vlm_verifier = VLMVerifier() if run_vlm else None

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_cache, exist_ok=True)

    # Run evaluation
    results = []
    stats = defaultdict(lambda: {'total': 0, 'geo_correct': 0, 'vlm_correct': 0})

    print(f"\nEvaluating {len(dataset)} samples...")
    print("-" * 70)

    for i, sample in enumerate(dataset):
        image_name = sample.get('image', '')
        caption = sample.get('caption', '')
        relation = sample.get('relation', '').lower()
        label = sample.get('label', 0) == 1

        mapped = DIRECTIONAL_RELATIONS.get(relation)
        if not mapped:
            continue

        # Parse subject/object
        subject, obj = parse_caption(caption, relation)
        if not subject or not obj:
            continue

        # Download image
        image_path = download_image(image_name, image_cache)
        if not image_path:
            continue

        try:
            image = Image.open(image_path).convert('RGB')
        except:
            continue

        result = Result(
            idx=i, image=image_name, caption=caption,
            relation=relation, mapped_relation=mapped,
            label=label, subject=subject, obj=obj
        )

        # Detect objects
        if detector:
            bboxes = detector.detect_objects(image, [subject, obj])
            result.subject_bbox = bboxes.get(subject)
            result.obj_bbox = bboxes.get(obj)
            result.detection_success = bool(result.subject_bbox and result.obj_bbox)

        # Geometric evaluation using SpatialReasoner
        if spatial_reasoner and result.detection_success:
            # Create EntityCandidate objects for SpatialReasoner
            subject_candidate = EntityCandidate(
                entity_id=f"{subject}_0",
                label=subject,
                bbox=result.subject_bbox,
                confidence=1.0
            )
            obj_candidate = EntityCandidate(
                entity_id=f"{obj}_0",
                label=obj,
                bbox=result.obj_bbox,
                confidence=1.0
            )

            start = time.time()

            # Check if relation is supported by SpatialReasoner
            if spatial_reasoner.can_verify(mapped):
                geo_prob, debug_info = spatial_reasoner.verify(
                    image=image,
                    subject=subject_candidate,
                    obj=obj_candidate,
                    relation=mapped,
                    image_id=image_name  # Use image name as cache key
                )
                result.geo_prob = geo_prob
            else:
                # Unsupported relation
                result.geo_prob = 0.5

            result.geo_time_ms = (time.time() - start) * 1000
            result.geo_pred = result.geo_prob >= 0.5
            result.geo_correct = result.geo_pred == label

        # VLM evaluation
        if vlm_verifier:
            start = time.time()
            result.vlm_prob, result.vlm_response = vlm_verifier.verify(
                image, subject, relation, obj
            )
            result.vlm_time_ms = (time.time() - start) * 1000
            result.vlm_pred = result.vlm_prob >= 0.5
            result.vlm_correct = result.vlm_pred == label

        results.append(result)

        # Update stats
        stats[relation]['total'] += 1
        if result.geo_correct:
            stats[relation]['geo_correct'] += 1
        if result.vlm_correct:
            stats[relation]['vlm_correct'] += 1

        # Progress
        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(dataset)}] Processed...")

    # Clear depth cache after evaluation
    if spatial_reasoner:
        spatial_reasoner.clear_depth_cache()

    # Print results
    print_results(results, stats)

    # Save results
    save_results(results, stats, output_dir)

    return results, stats


def print_results(results: List[Result], stats: Dict):
    """Print evaluation results."""
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    total = len(results)

    # Detection success rate
    det_success = sum(1 for r in results if r.detection_success)
    print(f"\nDetection success: {det_success}/{total} ({100 * det_success / max(total, 1):.1f}%)")

    # Geometric accuracy (only on detection success)
    geo_results = [r for r in results if r.geo_correct is not None]
    geo_correct = sum(1 for r in geo_results if r.geo_correct)
    geo_acc = geo_correct / max(len(geo_results), 1)

    # VLM accuracy
    vlm_results = [r for r in results if r.vlm_correct is not None]
    vlm_correct = sum(1 for r in vlm_results if r.vlm_correct)
    vlm_acc = vlm_correct / max(len(vlm_results), 1)

    print(f"\n--- Overall Accuracy ---")
    print(f"  Geometric: {geo_acc:.1%} ({geo_correct}/{len(geo_results)})")
    print(f"  VLM:       {vlm_acc:.1%} ({vlm_correct}/{len(vlm_results)})")

    # Timing
    geo_times = [r.geo_time_ms for r in results if r.geo_time_ms]
    vlm_times = [r.vlm_time_ms for r in results if r.vlm_time_ms]

    if geo_times and vlm_times:
        avg_geo = sum(geo_times) / len(geo_times)
        avg_vlm = sum(vlm_times) / len(vlm_times)
        print(f"\n--- Inference Time ---")
        print(f"  Geometric: {avg_geo:.1f} ms")
        print(f"  VLM:       {avg_vlm:.1f} ms")
        print(f"  Speedup:   {avg_vlm / max(avg_geo, 1):.1f}x")

    # Per-relation breakdown
    print(f"\n--- Per-Relation Accuracy ---")
    print(f"  {'Relation':<20} {'N':>6} {'Geo':>10} {'VLM':>10}")
    print("  " + "-" * 48)

    for rel, s in sorted(stats.items(), key=lambda x: -x[1]['total']):
        geo_rate = s['geo_correct'] / max(s['total'], 1)
        vlm_rate = s['vlm_correct'] / max(s['total'], 1)
        print(f"  {rel:<20} {s['total']:>6} {geo_rate:>9.1%} {vlm_rate:>9.1%}")


def save_results(results: List[Result], stats: Dict, output_dir: str):
    """Save results to JSON."""
    # Predictions
    predictions = []
    for r in results:
        predictions.append({
            'idx': r.idx,
            'image': r.image,
            'caption': r.caption,
            'relation': r.relation,
            'mapped_relation': r.mapped_relation,
            'label': r.label,
            'subject': r.subject,
            'obj': r.obj,
            'detection_success': r.detection_success,
            'geo_prob': r.geo_prob,
            'geo_pred': r.geo_pred,
            'geo_correct': r.geo_correct,
            'geo_time_ms': r.geo_time_ms,
            'vlm_prob': r.vlm_prob,
            'vlm_pred': r.vlm_pred,
            'vlm_correct': r.vlm_correct,
            'vlm_response': r.vlm_response,
            'vlm_time_ms': r.vlm_time_ms,
        })

    with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
        json.dump(predictions, f, indent=2)

    # Summary
    summary = {
        'total_samples': len(results),
        'detection_success': sum(1 for r in results if r.detection_success),
        'geo_correct': sum(1 for r in results if r.geo_correct),
        'vlm_correct': sum(1 for r in results if r.vlm_correct),
        'by_relation': dict(stats)
    }

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}/")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="VSR Directional Evaluation")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--vlm-only", action="store_true", help="Only run VLM")
    parser.add_argument("--geo-only", action="store_true", help="Only run geometric")
    parser.add_argument("--full", action="store_true", help="Include all geometric relations")
    parser.add_argument("--use-depth", action="store_true", help="Use depth for front/behind")
    parser.add_argument("--output-dir", default="eval/vsr_results")
    parser.add_argument("--image-cache", default="data/coco_images")

    args = parser.parse_args()

    run_evaluation(
        max_samples=args.max_samples,
        run_geometric=not args.vlm_only,
        run_vlm=not args.geo_only,
        use_depth=args.use_depth,
        directional_only=not args.full,
        output_dir=args.output_dir,
        image_cache=args.image_cache
    )


if __name__ == "__main__":
    main()