#!/usr/bin/env python3
"""
Comprehensive PROVE evaluation with dual-model scoring.

Runs the new pipeline on NLVR2 balanced test set, collecting BOTH
BLIP-ITM and Qwen logit-based scores for every verification action.
Saves all intermediate data for post-hoc experiments.
"""

import os
import sys
import json
import gc
import time
import argparse
import traceback
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '/home/huan2073/PROVE')

from src.core.model_manager import ModelManager
from src.core.knowledge_base import KnowledgeBase
from src.pipeline.detector import Detector
from src.pipeline.unified_agent import UnifiedAgent, EvidenceCollection, EntityCandidate
from src.pipeline.problog_executor import ProbLogExecutor
from src.pipeline.problog_builder import ProbLogFactBuilder
from src.vision.qwen_verifier import QwenVerifier
from src.vision.blip_verifier import BLIPVerifier


def identifier_to_image_paths(identifier, img_dir, directory=None):
    """Map NLVR2 identifier to image file paths.

    identifier format: "test1-X-Y-Z" → images "test1-X-Y-img0.png" and "test1-X-Y-img1.png"
    Also checks nested train directory structure: images/train/{directory}/train-X-Y-img0.png
    """
    parts = identifier.rsplit("-", 1)
    img_base = parts[0]
    img_a = os.path.join(img_dir, f"{img_base}-img0.png")
    img_b = os.path.join(img_dir, f"{img_base}-img1.png")

    # If flat paths don't exist, check nested directory structure (train data)
    if not os.path.exists(img_a) and directory is not None:
        split = identifier.split("-")[0]  # "train", "test1", "dev"
        nested_a = os.path.join(img_dir, "images", split, str(directory), f"{img_base}-img0.png")
        nested_b = os.path.join(img_dir, "images", split, str(directory), f"{img_base}-img1.png")
        if os.path.exists(nested_a):
            return nested_a, nested_b

    return img_a, img_b


def load_gqa_samples(json_path, img_dir):
    """Load GQA dataset samples.

    GQA format: dict of {question_id: {question, imageId, answer, ...}}
    Returns list of dicts with unified keys matching NLVR2 format.
    """
    with open(json_path) as f:
        data = json.load(f)

    samples = []
    for qid, entry in data.items():
        image_id = entry["imageId"]
        answer = entry["answer"].lower()
        # Map yes/no to True/False for consistency with NLVR2
        label = answer == "yes"

        # Try .jpg first, then .png
        img_path = os.path.join(img_dir, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, f"{image_id}.png")

        samples.append({
            "identifier": qid,
            "sentence": entry["question"],
            "label": label,
            "image_path": img_path,
            "gqa_answer": answer,
        })

    return samples


def rescore_with_qwen(evidence, candidates, image_paths, qwen_yn, qwen_tf):
    """Re-score all attribute and relationship evidence with Qwen verifiers.

    Returns dicts mapping (entity_id, attribute, value) → scores for attrs,
    and (subject_id, object_id, relation) → scores for rels.
    """
    attr_scores = []
    rel_scores = []

    # Re-score attributes
    # New pipeline: 3-tuple (entity_id, attribute_value, probability)
    for entity_id, attr_value, blip_score in evidence.attributes:
        # Find entity bbox
        entity = None
        for c in candidates:
            if c.entity_id == entity_id:
                entity = c
                break

        if entity is None:
            attr_scores.append({
                "entity_id": entity_id, "value": attr_value,
                "blip_score": blip_score,
                "image_id": None, "bbox": None, "object_class": None,
                "qwen_yn_score": None, "qwen_yn_response": None,
                "qwen_tf_score": None, "qwen_tf_response": None,
            })
            continue

        image_path = image_paths.get(entity.image_id)
        if not image_path or not os.path.exists(image_path):
            attr_scores.append({
                "entity_id": entity_id, "value": attr_value,
                "blip_score": blip_score,
                "image_id": entity.image_id, "bbox": entity.bbox,
                "object_class": entity.object_class,
                "qwen_yn_score": None, "qwen_yn_response": None,
                "qwen_tf_score": None, "qwen_tf_response": None,
            })
            continue

        try:
            yn_score, yn_resp = qwen_yn.verify_attribute(
                image_path, entity.bbox, entity.object_class, attr_value, use_logits=True
            )
            tf_score, tf_resp = qwen_tf.verify_attribute(
                image_path, entity.bbox, entity.object_class, attr_value, use_logits=True
            )
        except Exception as e:
            yn_score, yn_resp = None, f"ERROR: {e}"
            tf_score, tf_resp = None, f"ERROR: {e}"

        attr_scores.append({
            "entity_id": entity_id, "value": attr_value,
            "blip_score": blip_score,
            "image_id": entity.image_id, "bbox": entity.bbox,
            "object_class": entity.object_class,
            "qwen_yn_score": yn_score, "qwen_yn_response": yn_resp,
            "qwen_tf_score": tf_score, "qwen_tf_response": tf_resp,
        })

    # Re-score relationships
    for subject_id, object_id, relation, blip_score in evidence.relationships:
        subject = None
        obj = None
        for c in candidates:
            if c.entity_id == subject_id:
                subject = c
            if c.entity_id == object_id:
                obj = c

        if subject is None or obj is None:
            rel_scores.append({
                "subject_id": subject_id, "object_id": object_id, "relation": relation,
                "blip_score": blip_score,
                "image_id": None, "bbox1": None, "bbox2": None,
                "obj1_class": None, "obj2_class": None,
                "qwen_yn_score": None, "qwen_yn_response": None,
                "qwen_tf_score": None, "qwen_tf_response": None,
            })
            continue

        image_path = image_paths.get(subject.image_id)
        if not image_path or not os.path.exists(image_path):
            rel_scores.append({
                "subject_id": subject_id, "object_id": object_id, "relation": relation,
                "blip_score": blip_score,
                "image_id": subject.image_id,
                "bbox1": subject.bbox, "bbox2": obj.bbox,
                "obj1_class": subject.object_class, "obj2_class": obj.object_class,
                "qwen_yn_score": None, "qwen_yn_response": None,
                "qwen_tf_score": None, "qwen_tf_response": None,
            })
            continue

        try:
            yn_score, yn_resp = qwen_yn.verify_relationship(
                image_path, subject.bbox, obj.bbox,
                subject.object_class, obj.object_class, relation, use_logits=True
            )
            tf_score, tf_resp = qwen_tf.verify_relationship(
                image_path, subject.bbox, obj.bbox,
                subject.object_class, obj.object_class, relation, use_logits=True
            )
        except Exception as e:
            yn_score, yn_resp = None, f"ERROR: {e}"
            tf_score, tf_resp = None, f"ERROR: {e}"

        rel_scores.append({
            "subject_id": subject_id, "object_id": object_id, "relation": relation,
            "blip_score": blip_score,
            "image_id": subject.image_id,
            "bbox1": subject.bbox, "bbox2": obj.bbox,
            "obj1_class": subject.object_class, "obj2_class": obj.object_class,
            "qwen_yn_score": yn_score, "qwen_yn_response": yn_resp,
            "qwen_tf_score": tf_score, "qwen_tf_response": tf_resp,
        })

    # Count evidence (no alternative scoring — based on detection confidence)
    count_data = []
    for ce in evidence.counts:
        count_data.append({
            "query_type": ce.query_type,
            "object_class": ce.object_class,
            "probability": ce.probability,
            "image_id": ce.image_id,
            "image_id_a": ce.image_id_a,
            "image_id_b": ce.image_id_b,
            "value": ce.value,
        })

    return attr_scores, rel_scores, count_data


def main():
    parser = argparse.ArgumentParser(description="PROVE evaluation with dual scoring")
    parser.add_argument("--dataset", choices=["nlvr2", "gqa"], default="nlvr2",
                        help="Dataset type: nlvr2 (image pairs) or gqa (single image)")
    parser.add_argument("--test_json",
                        default=None,
                        help="Path to test data (default depends on --dataset)")
    parser.add_argument("--img_dir",
                        default=None,
                        help="Path to images directory (default depends on --dataset)")
    parser.add_argument("--output_dir",
                        default="/home/huan2073/PROVE/eval/dual_eval")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to process (0 = all)")
    parser.add_argument("--resume_from", type=int, default=0,
                        help="Resume from this sample index")
    parser.add_argument("--end_at", type=int, default=0,
                        help="Stop at this sample index (0 = process all)")
    parser.add_argument("--z_filter", type=int, default=None, choices=[0, 1],
                        help="Only keep samples with this Z value (0 or 1) to avoid duplicates (NLVR2 only)")
    parser.add_argument("--thinking_budget", type=int, default=None,
                        help="Enable Claude extended thinking with this token budget (e.g. 4096)")
    parser.add_argument("--cot", action="store_true",
                        help="Enable prompt-level chain-of-thought reasoning")
    parser.add_argument("--blip_lora_path", type=str, default=None,
                        help="Path to fine-tuned BLIP LoRA adapter (e.g. eval/vqa_finetune/blip_lora_best)")
    parser.add_argument("--retry_failed", action="store_true",
                        help="Remove failed entries from results and retry them")
    parser.add_argument("--ice_file", type=str, default=None,
                        help="Path to JSON file with in-context examples for ProbLog generation")
    parser.add_argument("--ice_embeddings", type=str, default=None,
                        help="Path to .npy file with pre-computed ICE embeddings (enables dynamic retrieval)")
    parser.add_argument("--ice_k", type=int, default=3,
                        help="Number of ICEs to retrieve per question (default: 3)")
    args = parser.parse_args()

    # Set dataset-specific defaults
    if args.dataset == "gqa":
        if args.test_json is None:
            args.test_json = "/scratch/gautschi/huan2073/gqa_data/testdev_balanced_yn.json"
        if args.img_dir is None:
            args.img_dir = "/scratch/gautschi/huan2073/gqa_data/images"
    else:
        if args.test_json is None:
            args.test_json = "/scratch/gautschi/huan2073/nlvr2_data/balanced_test1.json"
        if args.img_dir is None:
            args.img_dir = "/scratch/gautschi/huan2073/nlvr2_data/images"

    os.makedirs(args.output_dir, exist_ok=True)

    # Load in-context examples if provided
    ices = None
    ice_retriever = None
    if args.ice_file:
        if args.ice_embeddings:
            from src.pipeline.ice_retriever import ICERetriever
            ice_retriever = ICERetriever(args.ice_file, args.ice_embeddings, k=args.ice_k)
            print(f"Dynamic ICE retrieval enabled: {args.ice_k} per question from {args.ice_file}")
        else:
            with open(args.ice_file) as f:
                ices = json.load(f)
            print(f"Loaded {len(ices)} in-context examples (fixed set) from {args.ice_file}")

    # Load test data
    is_gqa = args.dataset == "gqa"
    print(f"Loading test data ({args.dataset})...")

    if is_gqa:
        samples = load_gqa_samples(args.test_json, args.img_dir)
    else:
        # NLVR2: JSONL format
        samples = []
        with open(args.test_json) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        # Filter by Z value to avoid duplicate image pairs
        if args.z_filter is not None:
            before = len(samples)
            samples = [s for s in samples if s['identifier'].endswith(f'-{args.z_filter}')]
            print(f"Filtered Z={args.z_filter}: {before} → {len(samples)} samples")

    # Pre-filter to samples with valid images, then apply max_samples
    if args.max_samples > 0:
        valid_samples = []
        skipped = 0
        for s in samples:
            if is_gqa:
                valid = os.path.exists(s["image_path"])
            else:
                img_a, img_b = identifier_to_image_paths(
                    s["identifier"], args.img_dir, directory=s.get("directory"))
                valid = os.path.exists(img_a) and os.path.exists(img_b)
            if valid:
                valid_samples.append(s)
                if len(valid_samples) >= args.max_samples:
                    break
            else:
                skipped += 1
        print(f"Skipped {skipped} samples with missing images")
        samples = valid_samples
    print(f"Loaded {len(samples)} samples")

    # Initialize models
    print("Initializing models...")
    mm = ModelManager()

    # Pre-initialize LLM client with model ID and thinking/CoT settings
    # (singleton ensures all pipeline components use the same instance)
    from src.language.llm_client import LLMClient
    model_id = os.getenv("LLAMA33_MODEL_ID")
    if args.thinking_budget or args.cot or model_id:
        if args.thinking_budget:
            print(f"Enabling extended thinking with budget={args.thinking_budget} tokens")
        if args.cot:
            print(f"Enabling prompt-level chain-of-thought")
        print(f"Model ID: {model_id}")
        mm._models['llm_client'] = LLMClient(
            model_id=model_id,
            thinking_budget=args.thinking_budget,
            cot_enabled=args.cot
        )

    # Load fine-tuned BLIP if LoRA path provided
    if args.blip_lora_path:
        print(f"Using fine-tuned BLIP from {args.blip_lora_path}")
        mm._models['blip_verifier'] = BLIPVerifier(device="auto", lora_path=args.blip_lora_path)

    detector = Detector()
    agent = UnifiedAgent(max_iterations=20)
    executor = ProbLogExecutor()
    fact_builder = ProbLogFactBuilder()

    # Initialize Qwen verifiers (share underlying QwenVL model)
    qwen_vl = mm.get_qwen_vl()
    qwen_yn = QwenVerifier(qwen_vl=qwen_vl, prompt_style="yes_no")
    qwen_tf = QwenVerifier(qwen_vl=qwen_vl, prompt_style="true_false")

    print("All models loaded.\n")

    # Process samples
    results_file = os.path.join(args.output_dir, "all_results.json")

    # Always load existing results to support preemption recovery
    all_results = []
    done_identifiers = set()
    if os.path.exists(results_file):
        try:
            with open(results_file) as f:
                all_results = json.load(f)
            if args.retry_failed:
                # Only skip successful samples — retry failed ones
                failed_ids = {r.get('identifier') for r in all_results if not r.get('success') and r.get('identifier')}
                all_results = [r for r in all_results if r.get('success')]
                done_identifiers = {r.get('identifier') for r in all_results if r.get('identifier')}
                print(f"Loaded {len(all_results)} successful results, removed {len(failed_ids)} failed for retry")
            else:
                done_identifiers = {r.get('identifier') for r in all_results if r.get('identifier')}
                print(f"Loaded {len(all_results)} existing results from {results_file} "
                      f"({len(done_identifiers)} unique identifiers)")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: could not load existing results: {e}")
            all_results = []

    n_success = sum(1 for r in all_results if r.get("success"))
    n_fail = sum(1 for r in all_results if not r.get("success"))
    start_time = time.time()

    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        if idx < args.resume_from:
            continue
        if args.end_at > 0 and idx >= args.end_at:
            break

        identifier = sample["identifier"]

        # Skip already-processed samples (preemption recovery)
        if identifier in done_identifiers:
            continue
        sentence = sample["sentence"]
        label_str = sample.get("label", "False")
        label = label_str if isinstance(label_str, bool) else label_str.lower() == "true"

        # Build image_paths dict based on dataset type
        if is_gqa:
            img_path = sample["image_path"]
            image_paths = {"image_a": img_path}
            missing_images = not os.path.exists(img_path)
        else:
            img_a_path, img_b_path = identifier_to_image_paths(
                identifier, args.img_dir, directory=sample.get("directory"))
            image_paths = {"image_a": img_a_path, "image_b": img_b_path}
            missing_images = not os.path.exists(img_a_path) or not os.path.exists(img_b_path)

        result = {
            "identifier": identifier,
            "sentence": sentence,
            "label": label,
            "success": False,
        }

        if missing_images:
            result["error"] = "Image files not found"
            all_results.append(result)
            n_fail += 1
            continue

        try:

            # Step 1: Detection
            kb = KnowledgeBase(ultimate_question=sentence)
            for image_id, image_path in image_paths.items():
                detections = detector.detect_from_question(image_path, sentence)
                kb.add_objects(image_id, detections)

            # Save detection info
            det_info = {}
            for image_id, image_data in kb.images.items():
                det_info[image_id] = [
                    {"label": obj.label, "bbox": obj.bbox,
                     "confidence": obj.confidence, "object_id": obj.object_id}
                    for obj in image_data.objects
                ]
            result["detections"] = det_info

            # Step 2: Evidence collection (uses BLIP for verification)
            evidence = agent.collect_evidence(
                question=sentence,
                images=kb.images,
                image_paths=image_paths
            )

            # Build candidates list for Qwen re-scoring
            candidates = []
            for image_id, image_data in kb.images.items():
                letter = image_id.replace("image_", "")
                for obj in image_data.objects:
                    candidates.append(EntityCandidate(
                        entity_id=f"{obj.label}_{letter}_{obj.object_id}",
                        image_id=image_id,
                        object_class=obj.label,
                        bbox=obj.bbox,
                        confidence=obj.confidence
                    ))

            # Step 3: Re-score with Qwen
            attr_scores, rel_scores, count_data = rescore_with_qwen(
                evidence, candidates, image_paths, qwen_yn, qwen_tf
            )

            result["evidence"] = {
                "attributes": attr_scores,
                "relationships": rel_scores,
                "counts": count_data,
                "action_history": evidence.action_history,
            }

            # Step 4: ProbLog execution
            prob_facts = fact_builder.build_facts(evidence, kb.images)

            # Save facts for post-hoc re-execution
            facts_data = [
                {"predicate": f.predicate, "arguments": f.arguments,
                 "probability": f.probability}
                for f in prob_facts
            ]

            # Select ICEs: dynamic retrieval if available, else fixed set
            sample_ices = ice_retriever.retrieve(sentence) if ice_retriever else ices

            # Run ProbLog with BLIP scores (default)
            prob_result, det_result = executor.execute_dual(
                question=sentence,
                evidence=evidence,
                images=kb.images,
                threshold=0.5,
                ices=sample_ices
            )

            result["problog"] = {
                "rules": "",  # Will be filled from problog_program
                "query": "",
                "facts": facts_data,
            }

            # Extract rules and query from problog program
            program = prob_result.problog_program
            if program:
                # Parse rules and query from the program string
                lines = program.split("\n")
                rule_lines = []
                query_lines = []
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("query("):
                        query_lines.append(stripped)
                    elif stripped and not stripped.startswith("%") and "::" not in stripped:
                        rule_lines.append(stripped)
                result["problog"]["rules"] = "\n".join(rule_lines)
                result["problog"]["query"] = "\n".join(query_lines)

            result["results"] = {
                "prove_answer": prob_result.final_answer,
                "deprove_answer": det_result.final_answer,
                "prove_prob": prob_result.probability,
                "deprove_prob": det_result.probability,
                "prove_program": prob_result.problog_program,
                "deprove_program": det_result.problog_program,
            }

            result["success"] = True
            n_success += 1

        except Exception as e:
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            n_fail += 1
            print(f"  FAILED: {identifier}: {e}")

        all_results.append(result)

        # Periodic save and cleanup
        if (idx + 1) % 25 == 0:
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            elapsed = time.time() - start_time
            rate = (idx + 1 - args.resume_from) / elapsed * 3600
            print(f"\n  Progress: {idx+1}/{len(samples)}, "
                  f"success={n_success}, fail={n_fail}, "
                  f"rate={rate:.0f}/hr, "
                  f"elapsed={elapsed/60:.1f}min")

    # Final save
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {len(all_results)}, Success: {n_success}, Failed: {n_fail}")
    print(f"Time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")

    # Quick accuracy check
    successful = [r for r in all_results if r.get("success")]
    if successful:
        prove_correct = sum(1 for r in successful
                           if (r["results"]["prove_answer"] == "True") == r["label"])
        deprove_correct = sum(1 for r in successful
                             if (r["results"]["deprove_answer"] == "True") == r["label"])
        total = len(successful)
        print(f"\nPROVE accuracy:   {prove_correct}/{total} = {prove_correct/total*100:.1f}%")
        print(f"DePROVE accuracy: {deprove_correct}/{total} = {deprove_correct/total*100:.1f}%")

        # McNemar's test
        pw = sum(1 for r in successful
                 if (r["results"]["prove_answer"] == "True") == r["label"]
                 and (r["results"]["deprove_answer"] == "True") != r["label"])
        dw = sum(1 for r in successful
                 if (r["results"]["deprove_answer"] == "True") == r["label"]
                 and (r["results"]["prove_answer"] == "True") != r["label"])
        chi2 = (abs(pw - dw) - 1)**2 / (pw + dw) if (pw + dw) > 0 else 0
        sig = "***" if chi2 > 3.84 and pw > dw else ""
        print(f"McNemar: PW={pw}, DW={dw}, chi²={chi2:.2f} {sig}")

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
