#!/usr/bin/env python3
"""
Ablation: replace all VQA fact probabilities with Gaussian random values.
Tests whether PROVE's advantage comes from actual scores or ProbLog structure.
Runs multiple seeds and reports mean ± std.
"""
import json, math, sys, os, time, signal
import multiprocessing as mp
import numpy as np

from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

from src.eval.problog_utils import (
    SemiringDampened, execute_problog_direct, threshold_fn,
)

import argparse

BASE = _PROJECT_ROOT
LLM_NAMES = ['llama', 'maverick', 'mistral_large']

SCORE_CONFIGS = {
    'llama': {'dampened_alpha': 1.0},
    'maverick': {'dampened_alpha': 0.9},
    'mistral_large': {'dampened_alpha': 0.9},
}

DATASET_PATHS = {
    'test1': {
        'llm_files': {
            'llama': f'{BASE}/eval/v5_baseline_llama/all_results.json',
            'maverick': f'{BASE}/eval/v5_baseline_maverick/all_results.json',
            'mistral_large': f'{BASE}/eval/v5_baseline_mistral_large/all_results.json',
        },
        'extra_label_files': {
            'nova_pro': f'{BASE}/eval/v5_baseline_nova_pro/all_results.json',
            'nova_premier': f'{BASE}/eval/v5_baseline_nova_premier/all_results.json',
        },
    },
    'test2': {
        'llm_files': {
            'llama': f'{BASE}/eval/v5_test2_llama/all_results.json',
            'maverick': f'{BASE}/eval/v5_test2_maverick/all_results.json',
            'mistral_large': f'{BASE}/eval/v5_test2_mistral_large/all_results.json',
        },
        'extra_label_files': {},
    },
    'gqa': {
        'llm_files': {
            'llama': f'{BASE}/eval/v5_flex_gqa_llama/all_results.json',
            'maverick': f'{BASE}/eval/v5_flex_gqa_maverick/all_results.json',
            'mistral_large': f'{BASE}/eval/v5_flex_gqa_mistral_large/all_results.json',
        },
        'extra_label_files': {},
    },
}

THRESH_BASE = 0.45
THRESH_SLOPE = -0.1
NUM_SEEDS = 20
GAUSS_MEAN = 0.5
GAUSS_STD = 0.2




def _alarm_handler(signum, frame):
    raise TimeoutError("ProbLog timed out")


def _worker(args):
    llm, ident, sample, da, seed = args

    stored_facts = sample.get('problog', {}).get('facts', [])
    rules = sample.get('problog', {}).get('rules', '')
    query = sample.get('problog', {}).get('query', '')
    if not stored_facts or not rules or not query:
        return (llm, ident, seed, None, None, 0)

    # Replace non-entity fact probs with Gaussian random values
    rng = np.random.RandomState(hash((ident, llm, seed)) % (2**31))
    facts = []
    for fact in stored_facts:
        pred = fact.get('predicate', '')
        if pred == 'entity':
            # Keep entity prob as-is (from original pipeline)
            prob = fact.get('probability', 0.5)
        else:
            # Random Gaussian probability
            prob = rng.normal(GAUSS_MEAN, GAUSS_STD)
        prob = max(1e-7, min(1 - 1e-7, prob))
        facts.append({**fact, 'probability': prob})

    sr = SemiringDampened(alpha=da) if da != 1.0 else None

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(30)
    try:
        prove_prob = execute_problog_direct(facts, rules, query, semiring=sr)
        dep_facts = [{**f, 'probability': 1.0 if f['probability'] >= 0.5 else 0.0} for f in facts]
        dep_prob = execute_problog_direct(dep_facts, rules, query, semiring=None)
    except TimeoutError:
        prove_prob = None
        dep_prob = None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return (llm, ident, seed, prove_prob, dep_prob, len(facts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='test1', choices=['test1', 'test2', 'gqa'])
    args = parser.parse_args()

    ds = DATASET_PATHS[args.dataset]
    LLM_FILES = ds['llm_files']
    ALL_LLM_FILES = {**LLM_FILES, **ds['extra_label_files']}

    t0 = time.time()
    print("=" * 100)
    print(f"GAUSSIAN ABLATION ({args.dataset.upper()}): N({GAUSS_MEAN}, {GAUSS_STD}), {NUM_SEEDS} seeds")
    print("=" * 100)

    # Load labels
    print("\nLoading labels from all LLMs (union)...", flush=True)
    all_labels = {}
    for llm, path in ALL_LLM_FILES.items():
        try:
            with open(path) as f:
                data = json.load(f)
            for s in data:
                ident = s.get('identifier')
                if ident and s.get('success') and s.get('label') is not None:
                    all_labels[ident] = s['label']
        except FileNotFoundError:
            pass

    # Load raw samples
    llm_raw = {}
    for llm, path in LLM_FILES.items():
        with open(path) as f:
            data = json.load(f)
        idx = {}
        for s in data:
            ident = s.get('identifier')
            if ident and s.get('success') and s.get('label') is not None:
                idx[ident] = s
        llm_raw[llm] = idx

    all_ids = sorted(all_labels.keys())
    n = len(all_ids)
    labels_arr = np.array([all_labels[ident] for ident in all_ids], dtype=bool)
    print(f"  n = {n}")

    # Build work items: all LLMs × all samples × all seeds
    work_items = []
    for seed in range(NUM_SEEDS):
        for llm in LLM_NAMES:
            da = SCORE_CONFIGS[llm]['dampened_alpha']
            for ident in all_ids:
                sample = llm_raw.get(llm, {}).get(ident)
                if sample is not None:
                    work_items.append((llm, ident, sample, da, seed))

    print(f"  {len(work_items)} work items ({NUM_SEEDS} seeds × 3 LLMs × ~{n} samples)")
    print(f"  Using {os.cpu_count()} workers...", flush=True)

    # results[seed][llm][ident] = (prove_prob, dep_prob, n_facts)
    results = {s: {l: {} for l in LLM_NAMES} for s in range(NUM_SEEDS)}

    with mp.Pool(min(os.cpu_count() or 4, 90), maxtasksperchild=50) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker, work_items, chunksize=20)):
            llm, ident, seed, prove_prob, dep_prob, n_facts = result
            if prove_prob is not None:
                results[seed][llm][ident] = (prove_prob, dep_prob, n_facts)
            if (i + 1) % 5000 == 0:
                print(f"    {i+1}/{len(work_items)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  Precomputation done in {time.time()-t0:.0f}s\n")

    # Compute accuracy per seed
    prove_accs = []
    dep_accs_per_llm = {l: [] for l in LLM_NAMES}
    dep_majority_accs = []

    for seed in range(NUM_SEEDS):
        prove_arr = np.full((3, n), np.nan)
        deprove_arr = np.full((3, n), np.nan)
        nfacts_arr = np.zeros((3, n))

        for j, llm in enumerate(LLM_NAMES):
            cache = results[seed][llm]
            for i, ident in enumerate(all_ids):
                entry = cache.get(ident)
                if entry is not None and entry[0] is not None:
                    prove_arr[j, i] = entry[0]
                    deprove_arr[j, i] = entry[1] if entry[1] is not None else np.nan
                    nfacts_arr[j, i] = entry[2]

        valid_mask = ~np.isnan(prove_arr)
        valid_count = valid_mask.sum(axis=0)
        any_valid = valid_count > 0
        vc = np.maximum(valid_count, 1)

        # PROVE (perlm_soft)
        pp_safe = np.where(valid_mask, prove_arr, 0.0)
        avg_pp = pp_safe.sum(axis=0) / vc
        nf_safe = np.where(valid_mask, nfacts_arr, 0.0)
        avg_nf = nf_safe.sum(axis=0) / vc
        thresholds = np.array([threshold_fn(nf, THRESH_BASE, THRESH_SLOPE) for nf in avg_nf])
        prove_pred = avg_pp >= thresholds
        prove_pred[~any_valid] = ~labels_arr[~any_valid]
        prove_acc = (prove_pred == labels_arr).sum() / n
        prove_accs.append(prove_acc)

        # Per-LLM DePROVE
        dep_preds_all = {}
        for j, llm in enumerate(LLM_NAMES):
            vm = valid_mask[j]
            dp = deprove_arr[j] >= 0.5
            dp[~vm] = ~labels_arr[~vm]
            dep_accs_per_llm[llm].append((dp == labels_arr).sum() / n)
            dep_preds_all[llm] = dp

        # Majority vote
        dep_votes = np.stack([dep_preds_all[l] for l in LLM_NAMES], axis=0)
        majority_pred = dep_votes.sum(axis=0) >= 2
        dep_majority_accs.append((majority_pred == labels_arr).sum() / n)

    # Report
    print("=" * 100)
    print(f"  RESULTS: Gaussian N({GAUSS_MEAN}, {GAUSS_STD}), {NUM_SEEDS} seeds")
    print("=" * 100)

    prove_mean = 100 * np.mean(prove_accs)
    prove_std = 100 * np.std(prove_accs)
    print(f"\n  PROVE (perlm_soft):      {prove_mean:.2f}% ± {prove_std:.2f}%")

    for llm in LLM_NAMES:
        m = 100 * np.mean(dep_accs_per_llm[llm])
        s = 100 * np.std(dep_accs_per_llm[llm])
        short = {'llama': 'Llama', 'maverick': 'Maverick', 'mistral_large': 'Mistral Large'}
        print(f"  DePROVE ({short[llm]:14s}): {m:.2f}% ± {s:.2f}%")

    maj_m = 100 * np.mean(dep_majority_accs)
    maj_s = 100 * np.std(dep_majority_accs)
    print(f"  DePROVE (Majority Vote): {maj_m:.2f}% ± {maj_s:.2f}%")

    # Also show per-seed for reference
    print(f"\n  Per-seed PROVE: {['%.1f' % (100*a) for a in prove_accs]}")

    # Compare to real numbers
    real_prove = {'test1': 73.06, 'test2': 73.92, 'gqa': 63.99}
    real_dep = {'test1': 69.95, 'test2': 69.77, 'gqa': 61.02}
    print(f"\n  For reference — real PROVE: {real_prove.get(args.dataset, '?')}%, real best DePROVE: {real_dep.get(args.dataset, '?')}%")

    print(f"\n\nTotal time: {time.time()-t0:.0f}s")
