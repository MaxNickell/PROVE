#!/usr/bin/env python3
"""
Run PROVE pipeline on a single example with full optimized config.

Uses the same pipeline as run_eval.py: Qwen verifier, scoring config,
dampened semiring, and apply_config_to_facts.

Usage:
    # Custom image + question:
    python run_example.py --image photo.jpg --question "Is there a dog on the couch?"
    python run_example.py --image img1.jpg --image_b img2.jpg --question "Is there a cat in both images?"

    # Dataset examples:
    python run_example.py                                          # random test1 example
    python run_example.py --identifier test1-366-0-0               # specific NLVR2 example
    python run_example.py --dataset gqa                            # random GQA example
    python run_example.py --dataset gqa --identifier 201307251     # specific GQA example
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

from src.core.model_manager import ModelManager
from src.core.knowledge_base import KnowledgeBase
from src.core.types import ProbLogFact
from src.pipeline.detector import Detector
from src.pipeline.unified_agent import UnifiedAgent
from src.pipeline.problog_executor import ProbLogExecutor
from src.pipeline.problog_builder import ProbLogFactBuilder
from src.vision.qwen_verifier import QwenVerifier
from src.language import create_llm_client
from src.eval.run_eval import DATASET_PRESETS, LLM_MODELS, nlvr2_id_to_image_paths, load_single_image_samples
from src.eval.problog_utils import apply_config_to_facts

# Scoring configs (same as run_eval.py)
_CONFIGS_PATH = os.path.join(os.path.dirname(__file__), "configs.json")
with open(_CONFIGS_PATH) as _f:
    _ALL_CONFIGS = json.load(_f)


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
    parser = argparse.ArgumentParser(description="Run PROVE on a single example (full optimized pipeline)")
    parser.add_argument("--image", type=str, help="Path to image (custom mode)")
    parser.add_argument("--image_b", type=str, help="Path to second image for paired comparison")
    parser.add_argument("--question", type=str, help="Question to ask about the image(s)")
    parser.add_argument("--dataset", choices=list(DATASET_PRESETS.keys()), default="test1",
                        help=f"Dataset preset (default: test1)")
    parser.add_argument("--identifier", type=str, help="Specific example identifier")
    parser.add_argument("--save-logs", action="store_true", help="Save detailed logs")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override decision threshold (default: use config threshold)")
    parser.add_argument("--llm", choices=list(LLM_MODELS.keys()), default="llama",
                        help=f"LLM for entity extraction (default: llama). "
                             f"Choices: {', '.join(LLM_MODELS.keys())}")
    parser.add_argument("--config", type=str, default="v5_perlm",
                        choices=list(_ALL_CONFIGS.keys()),
                        help="Scoring config preset (default: v5_perlm)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root directory for datasets (overrides PROVE_DATA_ROOT)")
    args = parser.parse_args()

    # Set LLM model ID
    os.environ["LLAMA33_MODEL_ID"] = LLM_MODELS[args.llm]
    print(f"LLM: {args.llm} → {LLM_MODELS[args.llm]}")

    # Load scoring config
    scoring_preset = _ALL_CONFIGS[args.config]
    llm_cfg = scoring_preset['llm_configs'].get(args.llm, scoring_preset['fallback'])
    dampened_alpha = llm_cfg['dampened_alpha']
    ep = "orig" if llm_cfg["entity_prob"] is None else llm_cfg["entity_prob"]
    ag = llm_cfg["agreement_mode"] or "none"
    print(f"Config: {args.config} | attr={llm_cfg['attr_score_type']}, "
          f"rel={llm_cfg['rel_score_type']}, ep={ep}, da={dampened_alpha}, ag={ag}")

    # Compute threshold from config
    thresh_base = scoring_preset["threshold"]["base"]
    thresh_slope = scoring_preset["threshold"]["slope"]
    if args.threshold is not None:
        threshold = args.threshold
    else:
        # For a single example, use base threshold (n_facts unknown until after evidence)
        threshold = thresh_base
    print(f"Threshold: base={thresh_base}, slope={thresh_slope}")

    # Custom image mode
    if args.image:
        if not args.question:
            parser.error("--question is required when using --image")
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}")
            return
        image_paths = {"image_a": args.image}
        if args.image_b:
            if not os.path.exists(args.image_b):
                print(f"Image not found: {args.image_b}")
                return
            image_paths["image_b"] = args.image_b

        example = {
            "identifier": "custom",
            "question": args.question,
            "label": None,
            "image_paths": image_paths,
        }
    else:
        # Dataset mode
        data_root = args.data_root or os.environ.get("PROVE_DATA_ROOT", "")
        preset = DATASET_PRESETS[args.dataset]
        data_file = os.path.join(data_root, preset["test_json"])
        img_dir = os.path.join(data_root, preset["img_dir"])

        print(f"Loading {args.dataset} examples from {data_file}...")
        if preset["type"] == "single":
            examples = load_single_examples(data_file, img_dir)
        else:
            examples = load_paired_examples(data_file, img_dir, z_filter=preset["z_filter"])
        print(f"Found {len(examples)} valid examples\n")

        if not examples:
            print("No valid examples found. Check data and image paths.")
            return

        if args.identifier:
            example = next((e for e in examples if e["identifier"] == args.identifier), None)
            if not example:
                print(f"Example '{args.identifier}' not found")
                return
        else:
            example = random.choice(examples)

    question = example["question"]
    image_paths = example["image_paths"]

    # Print info
    print("\n" + "=" * 60)
    if args.image:
        print(f"Mode:         Custom image")
    else:
        print(f"Dataset:      {args.dataset}")
    print(f"Example:      {example['identifier']}")
    print(f"Question:     {question}")
    if example['label'] is not None:
        print(f"Ground Truth: {example['label']}")
    print(f"Images:       {list(image_paths.keys())}")
    print("=" * 60)

    # ── Initialize models (same as run_eval.py) ──
    print("\nInitializing models...")
    mm = ModelManager()

    # Pre-initialize LLM client with model ID
    model_id = os.getenv("LLAMA33_MODEL_ID")
    mm._models['llm_client'] = create_llm_client(model_id=model_id)

    detector = Detector()
    executor = ProbLogExecutor()
    fact_builder = ProbLogFactBuilder()

    # Initialize Qwen verifier and pass to agent
    qwen_vl = mm.get_qwen_vl()
    qwen_tf = QwenVerifier(qwen_vl=qwen_vl)
    agent = UnifiedAgent(max_iterations=20, extra_verifiers={"qwen_tf": qwen_tf})

    print("All models loaded.\n")

    try:
        # Step 1: Object Detection
        print(f"Step 1: Object Detection...")
        kb = KnowledgeBase(ultimate_question=question)
        for image_id, image_path in image_paths.items():
            detections = detector.detect_from_question(image_path, question)
            kb.add_objects(image_id, detections)
            print(f"  {image_id}: {len(detections)} objects detected")
            for det in detections:
                print(f"    - {det.label} (conf={det.confidence:.3f})")

        # Step 2: Evidence Collection (with Qwen verifier)
        print(f"\nStep 2: Evidence Collection...")
        evidence = agent.collect_evidence(
            question=question,
            images=kb.images,
            image_paths=image_paths
        )
        print(f"  Attributes: {len(evidence.attribute_scores)} scores")
        print(f"  Relationships: {len(evidence.relationship_scores)} scores")
        print(f"  Counts: {len(evidence.counts)} entries")

        # Step 3: Build ProbLog facts
        print(f"\nStep 3: Building ProbLog facts...")
        prob_facts = fact_builder.build_facts(evidence, kb.images)
        facts_data = [
            {"predicate": f.predicate, "arguments": f.arguments,
             "probability": f.probability}
            for f in prob_facts
        ]
        print(f"  Raw facts: {len(facts_data)}")

        # Step 4: Apply scoring config (Qwen scores, agreement, entity_prob)
        print(f"\nStep 4: Applying scoring config ({args.config})...")
        optimized_facts_data = apply_config_to_facts(
            facts_data, evidence.attribute_scores, evidence.relationship_scores,
            llm_cfg['attr_score_type'], llm_cfg['rel_score_type'],
            entity_prob=llm_cfg['entity_prob'],
            agreement_mode=llm_cfg['agreement_mode'])

        optimized_facts = [
            ProbLogFact(probability=f['probability'],
                        predicate=f['predicate'],
                        arguments=f['arguments'])
            for f in optimized_facts_data
        ]
        print(f"  Optimized facts: {len(optimized_facts)}")

        # Compute dynamic threshold: clip(base + slope * ln(n_facts+1), 0, 1)
        import math
        n_facts = len(optimized_facts)
        if args.threshold is not None:
            threshold = args.threshold
        else:
            threshold = max(0.0, min(1.0, thresh_base + thresh_slope * math.log(n_facts + 1)))
        print(f"  Threshold: {threshold:.4f} (n_facts={n_facts})")

        # Print facts for inspection
        print(f"\n  Facts:")
        for f in optimized_facts:
            print(f"    {f.probability:.3f}::{f.predicate}({', '.join(str(a) for a in f.arguments)})")

        # Step 5: ProbLog execution with dampened semiring
        print(f"\nStep 5: ProbLog Reasoning (dampened_alpha={dampened_alpha})...")
        prob_result, det_result = executor.execute_dual(
            question=question,
            evidence=evidence,
            images=kb.images,
            threshold=threshold,
            facts=optimized_facts,
            dampened_alpha=dampened_alpha
        )

        # ── Results ──
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        print(f"\nProbabilistic (PROVE):")
        print(f"  Probability: {prob_result.probability:.4f}")
        print(f"  Answer:      {prob_result.final_answer}")

        print(f"\nDeterministic (DePROVE):")
        print(f"  Probability: {det_result.probability:.4f}")
        print(f"  Answer:      {det_result.final_answer}")

        if example["label"] is not None:
            prob_correct = (prob_result.final_answer == example["label"])
            det_correct = (det_result.final_answer == example["label"])
            print(f"\nGround Truth:  {example['label']}")
            print(f"PROVE:         {'CORRECT' if prob_correct else 'WRONG'}")
            print(f"DePROVE:       {'CORRECT' if det_correct else 'WRONG'}")

        agreement = "AGREE" if prob_result.final_answer == det_result.final_answer else "DISAGREE"
        print(f"\nModes {agreement}")
        print("=" * 60)

        # Save logs if requested
        if args.save_logs:
            from datetime import datetime
            import shutil

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            question_hash = str(hash(question))[:8]
            example_dir = Path("logs") / f"{timestamp}_{question_hash}"
            example_dir.mkdir(parents=True, exist_ok=True)

            # Copy images
            images_dir = example_dir / "images"
            images_dir.mkdir(exist_ok=True)
            for img_id, img_path in image_paths.items():
                ext = Path(img_path).suffix or ".jpg"
                shutil.copy(img_path, images_dir / f"{img_id}{ext}")

            # Save ProbLog programs
            with open(example_dir / "probabilistic.pl", 'w') as f:
                f.write(prob_result.problog_program or "")
            with open(example_dir / "deterministic.pl", 'w') as f:
                f.write(det_result.problog_program or "")

            # Save results JSON
            results_json = {
                "identifier": example["identifier"],
                "question": question,
                "label": example["label"],
                "config": args.config,
                "llm": args.llm,
                "threshold": threshold,
                "dampened_alpha": dampened_alpha,
                "scoring": {
                    "attr_score_type": llm_cfg['attr_score_type'],
                    "rel_score_type": llm_cfg['rel_score_type'],
                    "entity_prob": llm_cfg['entity_prob'],
                    "agreement_mode": llm_cfg['agreement_mode'],
                },
                "detections": {
                    image_id: [
                        {"label": obj.label, "bbox": obj.bbox,
                         "confidence": obj.confidence, "object_id": obj.object_id}
                        for obj in image_data.objects
                    ]
                    for image_id, image_data in kb.images.items()
                },
                "evidence": {
                    "attributes": evidence.attribute_scores,
                    "relationships": evidence.relationship_scores,
                },
                "facts": optimized_facts_data,
                "results": {
                    "prove_answer": prob_result.final_answer,
                    "deprove_answer": det_result.final_answer,
                    "prove_prob": prob_result.probability,
                    "deprove_prob": det_result.probability,
                },
            }
            with open(example_dir / "results.json", 'w') as f:
                json.dump(results_json, f, indent=2, default=str)

            print(f"\nLogs saved to: {example_dir}")

    except Exception as e:
        print(f"\nError: PROVE pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
