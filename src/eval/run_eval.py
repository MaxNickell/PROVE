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
from tqdm import tqdm

# Add project root (two levels up from src/eval/) to sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

from src.core.model_manager import ModelManager
from src.core.knowledge_base import KnowledgeBase
from src.pipeline.detector import Detector
from src.pipeline.unified_agent import UnifiedAgent
from src.pipeline.problog_executor import ProbLogExecutor
from src.pipeline.problog_builder import ProbLogFactBuilder
from src.core.types import ProbLogFact
from src.vision.qwen_verifier import QwenVerifier

# ── LLM model registry ──────────────────────────────────────────────────────
LLM_MODELS = {
    "llama":          "us.meta.llama3-3-70b-instruct-v1:0",
    "maverick":       "us.meta.llama4-maverick-17b-instruct-v1:0",
    "mistral_large":  "mistral.mistral-large-3-675b-instruct",
    "nova_pro":       "us.amazon.nova-pro-v1:0",
    "nova_premier":   "us.amazon.nova-premier-v1:0",
    "sonnet":         "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
}

# ── Dataset presets ──────────────────────────────────────────────────────────
DATASET_PRESETS = {
    "test1": {
        "type": "paired",       # NLVR2: JSONL, two images per sample
        "test_json": "nlvr2_data/balanced_test1.json",
        "img_dir":   "nlvr2_data/images",
        "z_filter":  0,
    },
    "test2": {
        "type": "paired",
        "test_json": "nlvr2_data/balanced_test2.json",
        "img_dir":   "nlvr2_data/images",
        "z_filter":  0,
    },
    "gqa": {
        "type": "single",       # Single image, yes/no question
        "test_json": "gqa_data/testdev_balanced_yn.json",
        "img_dir":   "gqa_data/images",
        "z_filter":  None,
    },
    "vqav2": {
        "type": "single",
        "test_json": "vqav2_data/val_balanced_yn.json",
        "img_dir":   "vqav2_data/images",
        "z_filter":  None,
    },
    "vsr": {
        "type": "single",
        "test_json": "vsr_data/test_balanced.json",
        "img_dir":   "vsr_data/images",
        "z_filter":  None,
    },
}

# ── Scoring configs (loaded from configs.json) ──────────────────────────
_CONFIGS_PATH = os.path.join(os.path.dirname(__file__), "configs.json")
with open(_CONFIGS_PATH) as _f:
    _ALL_CONFIGS = json.load(_f)


def nlvr2_id_to_image_paths(identifier, img_dir, directory=None):
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


def load_single_image_samples(json_path, img_dir):
    """Load single-image yes/no dataset (GQA, VQAv2, etc.).

    Expected JSON format: dict of {question_id: {question, imageId, answer, ...}}
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



def _eval_sample_worker(args_tuple):
    """Worker function for parallel post-hoc ProbLog evaluation (picklable)."""
    from src.eval.problog_utils import rebuild_facts, execute_problog_direct, SemiringDampened

    llm, ident, sample, cfg = args_tuple
    facts = rebuild_facts(sample, cfg['attr_score_type'], cfg['rel_score_type'],
                          entity_prob=cfg['entity_prob'], agreement_mode=cfg['agreement_mode'])
    rules = sample.get('problog', {}).get('rules', '')
    query = sample.get('problog', {}).get('query', '')

    if facts is None or not rules or not query:
        return llm, ident, None, None, 0

    da = cfg['dampened_alpha']
    sr = SemiringDampened(alpha=da) if da != 1.0 else None
    prove_prob = execute_problog_direct(facts, rules, query, semiring=sr)

    dep_facts = [{**f, 'probability': 1.0 if f['probability'] >= 0.5 else 0.0} for f in facts]
    dep_prob = execute_problog_direct(dep_facts, rules, query, semiring=None)

    return llm, ident, prove_prob, dep_prob, len(facts)


def run_multi_llm(args):
    """Orchestrate parallel data collection for multiple LLMs, then evaluate ensemble."""
    import subprocess as sp

    llm_list = args.llm
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"\nMulti-LLM mode: {len(llm_list)} LLMs, {n_gpus} GPU(s) available")

    # Check which LLMs already have complete results
    to_run = []
    for llm in llm_list:
        output_dir = f"eval/{args.name}_{args.dataset}_{llm}"
        result_file = os.path.join(output_dir, "all_results.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            n_success = sum(1 for s in data if s.get('success'))
            print(f"  {llm}: {n_success} results exist in {output_dir}")
        else:
            to_run.append(llm)

    if to_run:
        print(f"\nLaunching data collection for: {', '.join(to_run)}")

        if n_gpus >= len(to_run):
            # Parallel: one LLM per GPU
            processes = []
            for i, llm in enumerate(to_run):
                gpu_id = i % n_gpus
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

                cmd = [sys.executable, os.path.abspath(__file__),
                       '--llm', llm,
                       '--dataset', args.dataset,
                       '--name', args.name,
                       '--config', args.config]
                if args.resume_from:
                    cmd.extend(['--resume_from', str(args.resume_from)])
                if args.end_at:
                    cmd.extend(['--end_at', str(args.end_at)])
                if args.max_samples:
                    cmd.extend(['--max_samples', str(args.max_samples)])
                if args.retry_failed:
                    cmd.append('--retry_failed')
                if args.data_root:
                    cmd.extend(['--data_root', args.data_root])
                if args.test_json:
                    cmd.extend(['--test_json', args.test_json])
                if args.img_dir:
                    cmd.extend(['--img_dir', args.img_dir])

                print(f"  {llm} → GPU {gpu_id}")
                p = sp.Popen(cmd, env=env)
                processes.append((llm, p))

            for llm, p in processes:
                rc = p.wait()
                if rc != 0:
                    print(f"  WARNING: {llm} exited with code {rc}")
                else:
                    print(f"  {llm} completed successfully")
        else:
            # Sequential: share GPUs (avoids OOM)
            for llm in to_run:
                print(f"\n  Running {llm}...")
                cmd = [sys.executable, os.path.abspath(__file__),
                       '--llm', llm,
                       '--dataset', args.dataset,
                       '--name', args.name,
                       '--config', args.config]
                if args.resume_from:
                    cmd.extend(['--resume_from', str(args.resume_from)])
                if args.end_at:
                    cmd.extend(['--end_at', str(args.end_at)])
                if args.max_samples:
                    cmd.extend(['--max_samples', str(args.max_samples)])
                if args.retry_failed:
                    cmd.append('--retry_failed')
                if args.data_root:
                    cmd.extend(['--data_root', args.data_root])
                if args.test_json:
                    cmd.extend(['--test_json', args.test_json])
                if args.img_dir:
                    cmd.extend(['--img_dir', args.img_dir])

                rc = sp.call(cmd)
                if rc != 0:
                    print(f"  WARNING: {llm} exited with code {rc}")
                else:
                    print(f"  {llm} completed successfully")
    else:
        print("\nAll LLM results already exist. Skipping data collection.")

    # Post-hoc ensemble evaluation with the specified scoring config
    evaluate_with_config(args)


def evaluate_with_config(args):
    """Apply scoring config to collected results and report ensemble accuracy."""
    import math
    import multiprocessing as mp
    from src.eval.problog_utils import threshold_fn

    llm_list = args.llm if isinstance(args.llm, list) else [args.llm]
    preset = _ALL_CONFIGS[args.config]
    thresh_base = preset["threshold"]["base"]
    thresh_slope = preset["threshold"]["slope"]

    # Load results for each LLM
    all_data = {}   # llm -> {ident: sample} (success only, for ProbLog)
    labels = {}     # ident -> bool label (all samples, success + fail)
    for llm in llm_list:
        result_path = f"eval/{args.name}_{args.dataset}_{llm}/all_results.json"
        if not os.path.exists(result_path):
            print(f"  WARNING: {result_path} not found, skipping {llm}")
            continue
        with open(result_path) as f:
            data = json.load(f)
        successful = {}
        for s in data:
            ident = s.get('identifier')
            if ident and s.get('label') is not None:
                labels[ident] = s['label']
            if s.get('success') and ident:
                successful[ident] = s
        all_data[llm] = successful
        print(f"  Loaded {llm}: {len(successful)} success / {len(data)} total")

    if not all_data:
        print("No results to evaluate!")
        return

    # n = total dataset size (all samples, not just successful ones)
    all_ids = sorted(labels.keys())
    n = len(all_ids)
    active_llms = sorted(all_data.keys())

    print(f"\n  Evaluating {n} samples, {len(active_llms)} LLMs: {', '.join(active_llms)}")
    print(f"  Config: {args.config}, Threshold: log(b={thresh_base}, s={thresh_slope})")
    llm_cfgs = {}
    for llm in active_llms:
        llm_cfgs[llm] = preset["llm_configs"].get(llm, preset["fallback"]).copy()
        c = llm_cfgs[llm]
        ep = "orig" if c["entity_prob"] is None else c["entity_prob"]
        ag = c["agreement_mode"] or "none"
        print(f"  {llm}: attr={c['attr_score_type']}, rel={c['rel_score_type']}, "
              f"ep={ep}, da={c['dampened_alpha']}, ag={ag}")

    # Check if results are pre-optimized with the matching config
    pre_optimized = all(
        sample.get('scoring_config') == args.config
        for llm_data in all_data.values()
        for sample in llm_data.values()
    ) if all_data else False

    t0 = time.time()
    results_map = {}  # (llm, ident) -> (prove_prob, dep_prob, n_facts)

    if pre_optimized:
        # Fast path: read stored optimized probs directly (no ProbLog re-execution)
        print(f"  Results pre-optimized with config '{args.config}' — reading stored probs")
        for llm in active_llms:
            for ident, sample in all_data[llm].items():
                r = sample.get('results', {})
                prove_prob = r.get('prove_prob')
                dep_prob = r.get('deprove_prob')
                nf = len(sample.get('problog', {}).get('facts', []))
                if prove_prob is not None or dep_prob is not None:
                    results_map[(llm, ident)] = (prove_prob, dep_prob, nf)
        print(f"  Loaded {len(results_map)} results in {time.time()-t0:.1f}s")
    else:
        # Legacy path: rebuild facts and re-run ProbLog with scoring config
        work_items = []
        for llm in active_llms:
            cfg = llm_cfgs[llm]
            for ident, sample in all_data[llm].items():
                work_items.append((llm, ident, sample, cfg))

        print(f"  {len(work_items)} ProbLog executions across {len(active_llms)} LLMs...")

        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
        n_workers = max(1, int(slurm_cpus) - 2) if slurm_cpus else min(mp.cpu_count() - 1, 12)

        if n_workers > 1:
            with mp.Pool(n_workers, maxtasksperchild=200) as pool:
                for i, result in enumerate(pool.imap_unordered(_eval_sample_worker, work_items, chunksize=10)):
                    llm, ident, prove_prob, dep_prob, nf = result
                    if prove_prob is not None or dep_prob is not None:
                        results_map[(llm, ident)] = (prove_prob, dep_prob, nf)
                    if (i + 1) % 500 == 0:
                        print(f"    {i+1}/{len(work_items)} done ({time.time()-t0:.0f}s)")
        else:
            for i, item in enumerate(work_items):
                result = _eval_sample_worker(item)
                llm, ident, prove_prob, dep_prob, nf = result
                if prove_prob is not None or dep_prob is not None:
                    results_map[(llm, ident)] = (prove_prob, dep_prob, nf)
                if (i + 1) % 200 == 0:
                    print(f"    {i+1}/{len(work_items)} done ({time.time()-t0:.0f}s)")

        print(f"  ProbLog done: {len(results_map)} results in {time.time()-t0:.1f}s")

    # Compute ensemble accuracy
    prove_correct = 0
    prove_all_missing = 0
    deprove_correct = {llm: 0 for llm in active_llms}

    for ident in all_ids:
        label = labels[ident]
        llm_prove_probs = []
        llm_nfacts = []

        for llm in active_llms:
            entry = results_map.get((llm, ident))
            if entry is None:
                # Missing = wrong for DePROVE
                continue
            prove_prob, dep_prob, nf = entry

            if prove_prob is not None:
                llm_prove_probs.append(prove_prob)
                llm_nfacts.append(nf)

            if dep_prob is not None and (dep_prob >= 0.5) == label:
                deprove_correct[llm] += 1

        # PROVE: perlm_soft (avg probs over valid LLMs)
        if llm_prove_probs:
            avg_prob = sum(llm_prove_probs) / len(llm_prove_probs)
            avg_nf = sum(llm_nfacts) / len(llm_nfacts)
            thresh = threshold_fn(avg_nf, thresh_base, thresh_slope)
            if (avg_prob >= thresh) == label:
                prove_correct += 1
        else:
            prove_all_missing += 1

    # Report
    print(f"\n{'='*60}")
    print(f"Results on {args.dataset} (n={n})")
    print(f"{'='*60}")
    prove_acc = prove_correct / n * 100
    print(f"PROVE (perlm_soft): {prove_correct}/{n} = {prove_acc:.2f}%")
    if prove_all_missing:
        print(f"  ({prove_all_missing} samples missing all LLMs → wrong)")
    for llm in active_llms:
        dep_acc = deprove_correct[llm] / n * 100
        print(f"DePROVE ({llm}): {deprove_correct[llm]}/{n} = {dep_acc:.2f}%")

    # DePROVE majority vote (if 3+ LLMs)
    if len(active_llms) >= 3:
        maj_correct = 0
        for ident in all_ids:
            label = labels[ident]
            votes = []
            for llm in active_llms:
                entry = results_map.get((llm, ident))
                if entry is None:
                    votes.append(not label)  # missing = wrong
                else:
                    _, dep_prob, _ = entry
                    votes.append(dep_prob >= 0.5 if dep_prob is not None else not label)

            pred = sum(votes) > len(votes) / 2
            if pred == label:
                maj_correct += 1
        print(f"DePROVE (majority): {maj_correct}/{n} = {maj_correct/n*100:.2f}%")

    # Write post-hoc PROVE/DePROVE probs back into each LLM's results file
    # (skip for pre-optimized results — probs are already in the main results)
    if not pre_optimized:
        for llm in active_llms:
            result_path = f"eval/{args.name}_{args.dataset}_{llm}/all_results.json"
            with open(result_path) as f:
                data = json.load(f)
            updated = 0
            for s in data:
                ident = s.get('identifier')
                if not ident:
                    continue
                entry = results_map.get((llm, ident))
                if entry is not None:
                    prove_prob, dep_prob, nf = entry
                    thresh = threshold_fn(nf, thresh_base, thresh_slope)
                    s['optimized_prove_prob'] = prove_prob
                    s['optimized_prove_pred'] = prove_prob >= thresh if prove_prob is not None else None
                    s['optimized_deprove_prob'] = dep_prob
                    s['optimized_deprove_pred'] = dep_prob >= 0.5 if dep_prob is not None else None
                    s['optimized_n_facts'] = nf
                    updated += 1
            with open(result_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  Wrote optimized predictions to {result_path} ({updated} samples)")


def main():
    parser = argparse.ArgumentParser(description="PROVE evaluation with dual scoring")
    parser.add_argument("--llm", nargs='+', choices=list(LLM_MODELS.keys()),
                        help=f"LLM(s) for ProbLog generation. Single LLM runs pipeline; "
                             f"multiple LLMs run in parallel then evaluate ensemble. "
                             f"Choices: {', '.join(LLM_MODELS.keys())}")
    parser.add_argument("--dataset", default="test1",
                        help=f"Dataset preset ({', '.join(DATASET_PRESETS.keys())}) "
                             f"or 'nlvr2'/'gqa' for manual paths")
    parser.add_argument("--name", default="v5",
                        help="Run name prefix for output dirs, e.g. eval/{name}_{dataset}_{llm} (default: v5)")
    parser.add_argument("--data_root", default=None,
                        help="Root directory for datasets (overrides PROVE_DATA_ROOT env var)")
    parser.add_argument("--test_json", default=None,
                        help="Path to test data (overrides dataset preset)")
    parser.add_argument("--img_dir", default=None,
                        help="Path to images directory (overrides dataset preset)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (auto-generated from --name/--dataset/--llm if not set)")
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
    parser.add_argument("--retry_failed", action="store_true",
                        help="Remove failed entries from results and retry them")
    parser.add_argument("--config", type=str, default="v5_perlm",
                        choices=list(_ALL_CONFIGS.keys()),
                        help="Scoring config preset from configs.json (default: v5_perlm)")
    args = parser.parse_args()

    # ── Multi-LLM mode: orchestrate parallel runs + ensemble eval ─────
    if args.llm and len(args.llm) > 1:
        run_multi_llm(args)
        return

    # ── Single-LLM mode: extract from list ────────────────────────────
    if args.llm:
        args.llm = args.llm[0]
        os.environ["LLAMA33_MODEL_ID"] = LLM_MODELS[args.llm]
        print(f"LLM: {args.llm} → {LLM_MODELS[args.llm]}")

    # ── Resolve data_root: CLI > env var > error ─────────────────────────
    data_root = args.data_root or os.environ.get("PROVE_DATA_ROOT", "")

    # ── Resolve dataset preset ───────────────────────────────────────────
    if args.dataset in DATASET_PRESETS:
        preset = DATASET_PRESETS[args.dataset]
        if args.test_json is None:
            args.test_json = os.path.join(data_root, preset["test_json"])
        if args.img_dir is None:
            args.img_dir = os.path.join(data_root, preset["img_dir"])
        if args.z_filter is None and preset["z_filter"] is not None:
            args.z_filter = preset["z_filter"]
        # Map preset type for pipeline logic
        is_single_image = preset["type"] == "single"
    else:
        # Unknown dataset — require --test_json and --img_dir, default to paired
        is_single_image = False

    # ── Auto-generate output_dir ─────────────────────────────────────────
    if args.output_dir is None:
        llm_suffix = f"_{args.llm}" if args.llm else ""
        dataset_name = args.dataset if args.dataset in DATASET_PRESETS else args.dataset
        args.output_dir = f"eval/{args.name}_{dataset_name}{llm_suffix}"
        print(f"Output dir: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data
    print(f"Loading test data ({args.dataset})...")

    if is_single_image:
        samples = load_single_image_samples(args.test_json, args.img_dir)
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
            if is_single_image:
                valid = os.path.exists(s["image_path"])
            else:
                img_a, img_b = nlvr2_id_to_image_paths(
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

    detector = Detector()
    executor = ProbLogExecutor()
    fact_builder = ProbLogFactBuilder()

    # Initialize Qwen verifier and pass to agent as extra verifier
    qwen_vl = mm.get_qwen_vl()
    qwen_tf = QwenVerifier(qwen_vl=qwen_vl)
    agent = UnifiedAgent(max_iterations=20, extra_verifiers={"qwen_tf": qwen_tf})

    print("All models loaded.\n")

    # Load LLM-specific scoring config for Phase 1 optimized ProbLog execution
    from src.eval.problog_utils import apply_config_to_facts
    scoring_preset = _ALL_CONFIGS[args.config]
    llm_cfg = scoring_preset['llm_configs'].get(args.llm, scoring_preset['fallback'])
    dampened_alpha = llm_cfg['dampened_alpha']
    ep = "orig" if llm_cfg["entity_prob"] is None else llm_cfg["entity_prob"]
    ag = llm_cfg["agreement_mode"] or "none"
    print(f"Scoring config ({args.config}): attr={llm_cfg['attr_score_type']}, "
          f"rel={llm_cfg['rel_score_type']}, ep={ep}, da={dampened_alpha}, ag={ag}")

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
        if is_single_image:
            img_path = sample["image_path"]
            image_paths = {"image_a": img_path}
            missing_images = not os.path.exists(img_path)
        else:
            img_a_path, img_b_path = nlvr2_id_to_image_paths(
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

        max_retries = 3
        for attempt in range(max_retries):
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

                # Step 2: Evidence collection (collects both BLIP and Qwen scores)
                evidence = agent.collect_evidence(
                    question=sentence,
                    images=kb.images,
                    image_paths=image_paths
                )

                # Step 3: Build evidence output (scores collected during evidence collection)
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

                result["evidence"] = {
                    "attributes": evidence.attribute_scores,
                    "relationships": evidence.relationship_scores,
                    "counts": count_data,
                    "action_history": evidence.action_history,
                }

                # Step 4: ProbLog execution with optimized scoring config
                prob_facts = fact_builder.build_facts(evidence, kb.images)

                # Save BLIP-only facts (baseline for rebuild_facts if config changes)
                facts_data = [
                    {"predicate": f.predicate, "arguments": f.arguments,
                     "probability": f.probability}
                    for f in prob_facts
                ]

                # Apply scoring config (Qwen scores, agreement mode, entity_prob)
                optimized_facts_data = apply_config_to_facts(
                    facts_data, evidence.attribute_scores, evidence.relationship_scores,
                    llm_cfg['attr_score_type'], llm_cfg['rel_score_type'],
                    entity_prob=llm_cfg['entity_prob'],
                    agreement_mode=llm_cfg['agreement_mode'])

                # Convert to ProbLogFact objects for executor
                optimized_facts = [
                    ProbLogFact(probability=f['probability'],
                                predicate=f['predicate'],
                                arguments=f['arguments'])
                    for f in optimized_facts_data
                ]

                # Run ProbLog with optimized facts + dampened semiring
                prob_result, det_result = executor.execute_dual(
                    question=sentence,
                    evidence=evidence,
                    images=kb.images,
                    threshold=0.5,
                    facts=optimized_facts,
                    dampened_alpha=dampened_alpha
                )

                result["problog"] = {
                    "rules": "",  # Will be filled from problog_program
                    "query": "",
                    "facts": facts_data,  # BLIP-only baseline
                }
                result["scoring_config"] = args.config

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
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  RETRY {attempt+1}/{max_retries}: {identifier}: {e}")
                    time.sleep(2)
                else:
                    result["error"] = str(e)
                    result["traceback"] = traceback.format_exc()
                    result["retries"] = max_retries
                    n_fail += 1
                    print(f"  FAILED after {max_retries} attempts: {identifier}: {e}")

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
