#!/usr/bin/env python3
"""
Clean PROVE-only sweep — finalized strategies for the paper.

Usage:
    python src/eval/run_prove_sweep.py llama=eval/v5_test1_llama/all_results.json maverick=eval/v5_test1_maverick/all_results.json
    python src/eval/run_prove_sweep.py --output_dir eval/v5_sweep_test2 llama=eval/v5_test2_llama/all_results.json maverick=eval/v5_test2_maverick/all_results.json mistral_large=eval/v5_test2_mistral_large/all_results.json

Dimensions:
1. VQA scoring (attr × rel, with mix-and-match)
2. Entity probability (original, 1.0, 0.99, 0.9)
3. Dampening alpha (1.0, 0.9, 0.8, 0.7)
4. VQA agreement adjustment (none, agree_0.5, agree_0.3)
5. PROVE threshold (fixed + dynamic logarithmic)
6. LLM combos (singles + all ensemble sizes 2..N; soft + hard vote)

All post-hoc — zero API calls.
Results saved to {output_dir}/results.json.
"""

import gc
import heapq
import os
import sys
import json
import time
import math
import pickle
import multiprocessing as mp
import numpy as np
from itertools import combinations
from collections import defaultdict

from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

from problog.evaluator import SemiringProbability

from src.eval.problog_utils import (
    SemiringDampened, quote_arg, build_score_lookups, get_score,
    rebuild_facts, execute_problog_direct,
)


import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='PROVE scoring config sweep (post-hoc, zero API calls)',
        usage='%(prog)s [options] llm=path [llm=path ...]')
    parser.add_argument('files', nargs='+', metavar='name=path',
                        help='LLM result files as name=path (e.g. llama=eval/v5_test1_llama/all_results.json)')
    parser.add_argument('--output_dir', '-o', default='eval/prove_sweep',
                        help='Output directory for results and cache (default: eval/prove_sweep)')
    return parser.parse_args()


# ─── Parallel worker (batch) ────────────────────────────────────────────────

def _worker_batch(batch, result_queue, worker_id):
    """Process a batch of work items in a subprocess. Writes results to queue.
    This runs as a standalone mp.Process so the parent can kill() it on timeout.
    Tags each result with worker_id so the parent can identify timed-out items."""
    for work_item in batch:
        score_key, ident, facts, rules, query, da = work_item
        sr = SemiringDampened(alpha=da) if da != 1.0 else None
        prove_prob = execute_problog_direct(facts, rules, query, semiring=sr)

        dep_facts = [{**f, 'probability': 1.0 if f['probability'] >= 0.5 else 0.0}
                     for f in facts]
        dep_prob = execute_problog_direct(dep_facts, rules, query, semiring=None)

        if prove_prob is None and dep_prob is None:
            result_queue.put((worker_id, ident, None))
        else:
            result_queue.put((worker_id, ident, (score_key, ident, (prove_prob, dep_prob, len(facts)))))


def _run_parallel_with_timeout(gen, n_workers, progress_cb, timed_out_idents,
                                batch_size=50, timeout_per_item=30):
    """Run work items in parallel using mp.Process workers with hard kill timeouts.

    Unlike mp.Pool, each worker is a plain Process that can be kill()'d if it hangs.
    Workers process batches of items; if a worker exceeds its time budget, it's killed.
    When a batch is killed, the first non-completed item's identifier is added to
    timed_out_idents so the generator can skip that sample for all future configs.
    """
    result_queue = mp.Queue()
    active = {}  # pid -> (process, deadline, batch_idents)
    results = []
    next_worker_id = [0]
    completed_by_worker = defaultdict(set)  # worker_id -> set of completed idents

    def _collect_results():
        """Drain the result queue without blocking."""
        collected = 0
        while not result_queue.empty():
            try:
                wid, ident, r = result_queue.get_nowait()
                completed_by_worker[wid].add(ident)
                if r is not None:
                    results.append(r)
                collected += 1
            except:
                break
        return collected

    def _reap_done():
        """Join finished workers."""
        for pid in list(active):
            proc, deadline, wid, batch_idents = active[pid]
            if not proc.is_alive():
                proc.join(timeout=1)
                completed_by_worker.pop(wid, None)
                del active[pid]

    def _kill_expired():
        """Kill workers that exceeded their time budget. Identify hanging samples."""
        now = time.time()
        for pid in list(active):
            proc, deadline, wid, batch_idents = active[pid]
            if now > deadline:
                proc.kill()
                proc.join(timeout=5)
                # Drain any results this worker put on the queue before dying
                _collect_results()
                # Find the first item that didn't complete — that's the one that hung
                completed = completed_by_worker.get(wid, set())
                for ident in batch_idents:
                    if ident not in completed:
                        timed_out_idents.add(ident)
                        print(f"    TIMEOUT: {ident} added to skip list ({len(timed_out_idents)} total)")
                        break
                completed_by_worker.pop(wid, None)
                del active[pid]

    # Buffer work items into batches
    batch = []
    items_submitted = 0

    for item in gen:
        batch.append(item)
        if len(batch) >= batch_size:
            # Wait for a slot
            while len(active) >= n_workers:
                _collect_results()
                _reap_done()
                _kill_expired()
                if len(active) >= n_workers:
                    time.sleep(0.05)

            wid = next_worker_id[0]
            next_worker_id[0] += 1
            batch_idents = [item[1] for item in batch]  # item[1] is ident
            # Deadline: generous time for fast items + one timeout for a hanging item
            # Most items finish in <0.5s; if one hangs, we detect it after timeout_per_item
            deadline = time.time() + len(batch) * 1.0 + timeout_per_item
            p = mp.Process(target=_worker_batch, args=(batch, result_queue, wid))
            p.start()
            active[p.pid] = (p, deadline, wid, batch_idents)
            items_submitted += len(batch)
            batch = []

            # Progress reporting
            _collect_results()
            progress_cb(len(results))

    # Submit final partial batch
    if batch:
        while len(active) >= n_workers:
            _collect_results()
            _reap_done()
            _kill_expired()
            if len(active) >= n_workers:
                time.sleep(0.05)
        wid = next_worker_id[0]
        next_worker_id[0] += 1
        batch_idents = [item[1] for item in batch]
        deadline = time.time() + len(batch) * 1.0 + timeout_per_item
        p = mp.Process(target=_worker_batch, args=(batch, result_queue, wid))
        p.start()
        active[p.pid] = (p, deadline, wid, batch_idents)
        items_submitted += len(batch)

    # Wait for all workers to finish
    while active:
        _collect_results()
        _reap_done()
        _kill_expired()
        if active:
            time.sleep(0.1)
        progress_cb(len(results))

    # Final drain
    _collect_results()
    return results


# ─── Data loading ────────────────────────────────────────────────────────────

def build_llm_index(samples):
    idx = {}
    for s in samples:
        ident = s.get('identifier')
        if ident and s.get('success', False) and s.get('label') is not None:
            idx[ident] = s
    return idx


def mcnemar_chi2(correct_a, correct_b):
    pw = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)
    dw = sum(1 for a, b in zip(correct_a, correct_b) if b and not a)
    chi2 = ((pw - dw) ** 2) / (pw + dw) if (pw + dw) > 0 else 0.0
    return chi2, pw, dw


# ─── Threshold strategies ────────────────────────────────────────────────────

def make_threshold_configs():
    """
    Return list of (name, params) tuples where params is a dict describing
    the threshold type and parameters for vectorized evaluation.
    """
    configs = []

    # Fixed thresholds: fine steps from 0.05 to 0.55
    for t_int in range(5, 56, 1):
        t = t_int / 100.0
        configs.append((f"t={t:.2f}", {'type': 'fixed', 'value': t}))

    # Dynamic logarithmic: threshold = base + slope * log(n_facts + 1)
    for base in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        for slope in [-0.12, -0.10, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10]:
            name = f"log(b={base},s={slope})"
            configs.append((name, {'type': 'log', 'base': base, 'slope': slope}))

    # Dynamic linear: threshold = base + slope * n_facts
    for base in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        for slope in [-0.025, -0.020, -0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015, 0.020]:
            name = f"lin(b={base},s={slope})"
            configs.append((name, {'type': 'linear', 'base': base, 'slope': slope}))

    # Binned: different thresholds for few/medium/many facts
    binned = [
        ("bin5_15(0.3/0.5/0.7)",  {'type': 'binned', 'b1': 5, 'b2': 15, 't1': 0.3, 't2': 0.5, 't3': 0.7}),
        ("bin5_15(0.4/0.5/0.6)",  {'type': 'binned', 'b1': 5, 'b2': 15, 't1': 0.4, 't2': 0.5, 't3': 0.6}),
        ("bin5_15(0.5/0.4/0.3)",  {'type': 'binned', 'b1': 5, 'b2': 15, 't1': 0.5, 't2': 0.4, 't3': 0.3}),
        ("bin5_15(0.6/0.5/0.4)",  {'type': 'binned', 'b1': 5, 'b2': 15, 't1': 0.6, 't2': 0.5, 't3': 0.4}),
        ("bin5_15(0.3/0.35/0.4)", {'type': 'binned', 'b1': 5, 'b2': 15, 't1': 0.3, 't2': 0.35, 't3': 0.4}),
        ("bin5_15(0.4/0.35/0.3)", {'type': 'binned', 'b1': 5, 'b2': 15, 't1': 0.4, 't2': 0.35, 't3': 0.3}),
        ("bin8_20(0.3/0.5/0.7)",  {'type': 'binned', 'b1': 8, 'b2': 20, 't1': 0.3, 't2': 0.5, 't3': 0.7}),
        ("bin8_20(0.4/0.5/0.6)",  {'type': 'binned', 'b1': 8, 'b2': 20, 't1': 0.4, 't2': 0.5, 't3': 0.6}),
        ("bin8_20(0.5/0.4/0.3)",  {'type': 'binned', 'b1': 8, 'b2': 20, 't1': 0.5, 't2': 0.4, 't3': 0.3}),
        ("bin8_20(0.4/0.35/0.3)", {'type': 'binned', 'b1': 8, 'b2': 20, 't1': 0.4, 't2': 0.35, 't3': 0.3}),
        ("bin10_25(0.3/0.4/0.5)", {'type': 'binned', 'b1': 10, 'b2': 25, 't1': 0.3, 't2': 0.4, 't3': 0.5}),
        ("bin10_25(0.5/0.4/0.3)", {'type': 'binned', 'b1': 10, 'b2': 25, 't1': 0.5, 't2': 0.4, 't3': 0.3}),
    ]
    configs.extend(binned)

    return configs


def compute_thresholds_vectorized(nfacts_arr, threshold_configs):
    """
    Compute threshold values for all configs at once.
    nfacts_arr: shape [...] (any shape)
    Returns: [n_thresholds, ...] array of threshold values.
    """
    n_thr = len(threshold_configs)
    result = np.empty((n_thr,) + nfacts_arr.shape)

    for ti, (name, params) in enumerate(threshold_configs):
        ttype = params['type']
        if ttype == 'fixed':
            result[ti] = params['value']
        elif ttype == 'log':
            result[ti] = np.clip(params['base'] + params['slope'] * np.log(nfacts_arr + 1), 0.0, 1.0)
        elif ttype == 'linear':
            result[ti] = np.clip(params['base'] + params['slope'] * nfacts_arr, 0.0, 1.0)
        elif ttype == 'binned':
            b1, b2 = params['b1'], params['b2']
            t1, t2, t3 = params['t1'], params['t2'], params['t3']
            result[ti] = np.where(nfacts_arr <= b1, t1, np.where(nfacts_arr <= b2, t2, t3))

    return result


# ─── Score configs ───────────────────────────────────────────────────────────

def make_score_configs():
    """
    Build all (attr_score, rel_score, entity_prob, dampening, agreement) combos.
    Returns dict: key -> config dict.
    """
    ALL_SCORES = [
        'blip_score', 'qwen_tf_score',
        'avg_blip_qwen_tf',
    ]

    # Same score for both attrs and rels
    scoring_pairs = [(s, s) for s in ALL_SCORES]

    # Mix-and-match: permutations of {blip, qwen_tf, avg_blip_qwen_tf} where attr != rel
    mix_types = ['blip_score', 'qwen_tf_score', 'avg_blip_qwen_tf']
    for a in mix_types:
        for r in mix_types:
            if a != r:
                scoring_pairs.append((a, r))

    entity_probs = [None, 1.0, 0.99, 0.9]
    dampening_alphas = [1.0, 0.9, 0.8, 0.7]
    agreement_modes = [None, 'dampen_0.5', 'dampen_0.3', 'sharpen', 'both']

    configs = {}
    for attr_st, rel_st in scoring_pairs:
        if attr_st == rel_st:
            score_label = attr_st
        else:
            # Shorten names for readability
            short = lambda s: s.replace('_score', '').replace('avg_blip_qwen_', 'abq_')
            score_label = f"A:{short(attr_st)}/R:{short(rel_st)}"

        for ep in entity_probs:
            ep_label = 'orig' if ep is None else str(ep)
            for da in dampening_alphas:
                for am in agreement_modes:
                    am_label = am if am else 'none'
                    key = f"{score_label}|ep={ep_label}|da={da}|ag={am_label}"
                    configs[key] = {
                        'attr_score_type': attr_st,
                        'rel_score_type': rel_st,
                        'entity_prob': ep,
                        'dampened_alpha': da,
                        'agreement_mode': am,
                    }

    return configs


# ═══════════════════════════════════════════════════════════════════════════════
# PRECOMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _work_item_generator(llm_name, idx, all_ids, score_configs, timed_out_idents):
    """Yield work items lazily. Checks timed_out_idents (shared set updated by
    the parallel runner) to skip samples that have already timed out."""
    skipped = 0
    for score_key, score_cfg in score_configs.items():
        ast = score_cfg['attr_score_type']
        rst = score_cfg['rel_score_type']
        ep = score_cfg.get('entity_prob')
        da = score_cfg.get('dampened_alpha', 1.0)
        am = score_cfg.get('agreement_mode')

        for ident in all_ids:
            if ident in timed_out_idents:
                skipped += 1
                continue
            sample = idx.get(ident)
            if sample is None:
                continue

            facts = rebuild_facts(sample, ast, rst, entity_prob=ep, agreement_mode=am)
            if facts is None:
                continue

            rules = sample.get('problog', {}).get('rules')
            query = sample.get('problog', {}).get('query')
            if not rules or not query:
                continue

            yield (score_key, ident, facts, rules, query, da)


def precompute_all(all_llm_indices, all_ids_by_llm, score_configs, output_dir):
    """
    Precompute ProbLog results for all (LLM, score_config, sample) combinations.
    Returns: precomputed[llm][(score_key, ident)] = (prove_prob, deprove_prob, num_facts)

    Caches per-LLM results to disk so completed LLMs can be skipped on restart.
    """
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    default_workers = int(slurm_cpus) - 2 if slurm_cpus else min(mp.cpu_count() - 2, 12)
    n_workers = max(1, int(os.environ.get('PROBLOG_WORKERS', default_workers)))
    # Cap workers — ProbLog uses ~1.7GB per active worker with maxtasksperchild recycling
    n_workers = min(n_workers, 180)

    cache_dir = os.path.join(output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    precomputed = {llm: {} for llm in all_llm_indices}

    for llm_name, idx in all_llm_indices.items():
        cache_path = os.path.join(cache_dir, f'{llm_name}_precomputed.pkl')

        # Check for cached results
        if os.path.exists(cache_path):
            print(f"  {llm_name}: loading from cache {cache_path}")
            with open(cache_path, 'rb') as f:
                precomputed[llm_name] = pickle.load(f)
            print(f"  {llm_name}: {len(precomputed[llm_name])} cached entries loaded")
            continue

        t0 = time.time()
        n_expected = len(score_configs) * len(all_ids_by_llm[llm_name])
        print(f"  {llm_name}: ~{n_expected} work items (generator), {n_workers} workers...")

        computed = 0
        timed_out_idents = set()  # dynamically populated when workers timeout
        gen = _work_item_generator(llm_name, idx, all_ids_by_llm[llm_name],
                                   score_configs, timed_out_idents)

        last_reported = [0]

        def _progress(n_results):
            nonlocal computed
            if n_results // 10000 > last_reported[0]:
                last_reported[0] = n_results // 10000
                elapsed = time.time() - t0
                print(f"    {llm_name}: {n_results} done ({elapsed:.0f}s)")

        if n_workers <= 1:
            for item in gen:
                score_key, ident, facts, rules, query, da = item
                sr = SemiringDampened(alpha=da) if da != 1.0 else None
                prove_prob = execute_problog_direct(facts, rules, query, semiring=sr)
                dep_facts = [{**f, 'probability': 1.0 if f['probability'] >= 0.5 else 0.0}
                             for f in facts]
                dep_prob = execute_problog_direct(dep_facts, rules, query, semiring=None)
                if prove_prob is not None or dep_prob is not None:
                    precomputed[llm_name][(score_key, ident)] = (prove_prob, dep_prob, len(facts))
                    computed += 1
                    _progress(computed)
        else:
            results = _run_parallel_with_timeout(
                gen, n_workers, _progress, timed_out_idents,
                batch_size=50, timeout_per_item=10
            )
            for r in results:
                if r is not None:
                    sk, ident, entry = r
                    precomputed[llm_name][(sk, ident)] = entry
                    computed += 1

        elapsed = time.time() - t0
        print(f"  {llm_name}: {computed} entries in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        if timed_out_idents:
            print(f"  {llm_name}: {len(timed_out_idents)} samples auto-skipped after timeout: {sorted(timed_out_idents)}")

        # Save to cache
        with open(cache_path, 'wb') as f:
            pickle.dump(precomputed[llm_name], f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  {llm_name}: cached to {cache_path}")

    return precomputed


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_all(precomputed, all_llm_indices, all_labels, score_configs, threshold_configs):
    """
    Sweep all configs: score_config × threshold × LLM combo × vote type.
    Fully vectorized: all thresholds processed simultaneously via numpy broadcasting.
    Failed samples are counted as incorrect (not removed).
    Returns list of result dicts.
    """
    llm_names = sorted(all_llm_indices.keys())

    # Build LLM combos: singles + all ensemble sizes (2, 3, ..., N)
    llm_combos = []
    for llm in llm_names:
        llm_combos.append(([llm], f"{llm}"))
    for r in range(2, len(llm_names) + 1):
        for combo in combinations(llm_names, r):
            llm_combos.append((sorted(combo), "+".join(sorted(combo))))

    all_sample_ids = sorted(all_labels.keys())
    n = len(all_sample_ids)
    n_thr = len(threshold_configs)
    thr_names = [name for name, _ in threshold_configs]
    print(f"  Universal sample set: {n} samples, {n_thr} thresholds")

    labels_arr = np.array([all_labels[ident] for ident in all_sample_ids], dtype=bool)

    all_results = []
    t0 = time.time()
    n_score = len(score_configs)

    for si, score_key in enumerate(score_configs):
        if si % 100 == 0:
            print(f"  Evaluating score config {si}/{n_score} ({time.time()-t0:.1f}s)")

        for llms, combo_name in llm_combos:
            num_llms = len(llms)

            # Build numpy arrays: [num_llms, n] with NaN for missing
            prove_arr = np.full((num_llms, n), np.nan)
            deprove_arr = np.full((num_llms, n), np.nan)
            nfacts_arr = np.zeros((num_llms, n))

            for j, llm in enumerate(llms):
                for i, ident in enumerate(all_sample_ids):
                    entry = precomputed[llm].get((score_key, ident))
                    if entry is not None:
                        prove_arr[j, i] = entry[0]
                        deprove_arr[j, i] = entry[1]
                        nfacts_arr[j, i] = entry[2]

            valid_mask = ~np.isnan(prove_arr)  # [num_llms, n]
            valid_count = valid_mask.sum(axis=0)  # [n]
            any_valid = valid_count > 0  # [n]
            vc = np.maximum(valid_count, 1)

            # DePROVE predictions: deprove >= 0.5 (same for all thresholds)
            deprove_preds = deprove_arr >= 0.5  # [num_llms, n]

            is_ensemble = num_llms > 1
            vote_types = ['soft', 'hard', 'conf_weighted', 'perf_weighted',
                          'prove_weighted', 'prove_weighted_sq'] if is_ensemble else ['single']

            if is_ensemble:
                llm_dep_acc = np.zeros(num_llms)
                for j in range(num_llms):
                    vm = valid_mask[j]
                    if vm.sum() > 0:
                        llm_dep_acc[j] = ((deprove_preds[j] == labels_arr) & vm).sum() / n
                    else:
                        llm_dep_acc[j] = 0.5

                # PROVE accuracy at reference threshold (t=0.35) per LLM
                llm_prove_acc = np.zeros(num_llms)
                for j in range(num_llms):
                    vm = valid_mask[j]
                    if vm.sum() > 0:
                        prove_preds_ref = np.where(vm, prove_arr[j] >= 0.35, ~labels_arr)
                        llm_prove_acc[j] = (prove_preds_ref == labels_arr).sum() / n
                    else:
                        llm_prove_acc[j] = 0.5

            # DePROVE baseline: use best single-LLM DePROVE for ensembles
            # (DePROVE outputs are binary 0/1, so soft-averaging creates an OR gate)
            if is_ensemble:
                per_llm_dep_correct = []
                per_llm_dep_acc_vals = []
                for j in range(num_llms):
                    vm = valid_mask[j]
                    dp = deprove_preds[j].copy()
                    dp[~vm] = ~labels_arr[~vm]
                    correct_j = (dp == labels_arr)
                    per_llm_dep_correct.append(correct_j)
                    per_llm_dep_acc_vals.append(correct_j.sum() / n)
                best_j = int(np.argmax(per_llm_dep_acc_vals))
                deprove_correct = per_llm_dep_correct[best_j]
                dep_acc = per_llm_dep_acc_vals[best_j]
            else:
                dp = deprove_preds[0].copy()
                dp[~valid_mask[0]] = ~labels_arr[~valid_mask[0]]
                deprove_correct = (dp == labels_arr)
                dep_acc = deprove_correct.sum() / n

            # Vectorized thresholds: [n_thr, num_llms, n]
            all_thresholds = compute_thresholds_vectorized(nfacts_arr, threshold_configs)
            # prove_arr: [num_llms, n] -> broadcast with [n_thr, num_llms, n]
            prove_preds_all = prove_arr[None, :, :] >= all_thresholds  # [n_thr, num_llms, n]

            # For soft/weighted votes, compute avg nfacts thresholds: [n_thr, n]
            nf_safe = np.where(valid_mask, nfacts_arr, 0.0)
            avg_nf = nf_safe.sum(axis=0) / vc  # [n]
            avg_thresholds = compute_thresholds_vectorized(avg_nf, threshold_configs)  # [n_thr, n]

            # Pre-compute shared quantities for soft/weighted votes
            pp_safe = np.where(valid_mask, prove_arr, 0.0)
            avg_pp = pp_safe.sum(axis=0) / vc  # [n]

            for vote_type in vote_types:
                # deprove_correct and dep_acc are already computed above
                # (best single-LLM for ensembles, single LLM for non-ensemble)

                if vote_type == 'single':
                    # [n_thr, n]
                    prove_pred = prove_preds_all[:, 0, :].copy()
                    prove_pred[:, ~valid_mask[0]] = ~labels_arr[~valid_mask[0]]
                elif vote_type == 'soft':
                    prove_pred = avg_pp[None, :] >= avg_thresholds  # [n_thr, n]
                    prove_pred[:, ~any_valid] = ~labels_arr[~any_valid]
                elif vote_type == 'hard':
                    pp_votes = (prove_preds_all & valid_mask[None, :, :]).sum(axis=1).astype(float)  # [n_thr, n]
                    prove_pred = pp_votes > valid_count[None, :] / 2.0
                    prove_pred[:, ~any_valid] = ~labels_arr[~any_valid]
                elif vote_type == 'conf_weighted':
                    weights = np.abs(prove_arr - 0.5) + 1e-6
                    weights = np.where(valid_mask, weights, 0.0)
                    w_sum = np.maximum(weights.sum(axis=0), 1e-12)
                    w_avg_pp = (weights * pp_safe).sum(axis=0) / w_sum  # [n]
                    prove_pred = w_avg_pp[None, :] >= avg_thresholds  # [n_thr, n]
                    prove_pred[:, ~any_valid] = ~labels_arr[~any_valid]
                elif vote_type == 'perf_weighted':
                    weights = llm_dep_acc[:, None] * valid_mask.astype(float)
                    w_sum = np.maximum(weights.sum(axis=0), 1e-12)
                    w_avg_pp = (weights * pp_safe).sum(axis=0) / w_sum  # [n]
                    prove_pred = w_avg_pp[None, :] >= avg_thresholds  # [n_thr, n]
                    prove_pred[:, ~any_valid] = ~labels_arr[~any_valid]
                elif vote_type == 'prove_weighted':
                    weights = llm_prove_acc[:, None] * valid_mask.astype(float)
                    w_sum = np.maximum(weights.sum(axis=0), 1e-12)
                    w_avg_pp = (weights * pp_safe).sum(axis=0) / w_sum  # [n]
                    prove_pred = w_avg_pp[None, :] >= avg_thresholds  # [n_thr, n]
                    prove_pred[:, ~any_valid] = ~labels_arr[~any_valid]
                elif vote_type == 'prove_weighted_sq':
                    weights = (llm_prove_acc ** 2)[:, None] * valid_mask.astype(float)
                    w_sum = np.maximum(weights.sum(axis=0), 1e-12)
                    w_avg_pp = (weights * pp_safe).sum(axis=0) / w_sum  # [n]
                    prove_pred = w_avg_pp[None, :] >= avg_thresholds  # [n_thr, n]
                    prove_pred[:, ~any_valid] = ~labels_arr[~any_valid]

                # Vectorized accuracy and McNemar across all thresholds at once
                prove_correct = (prove_pred == labels_arr[None, :])  # [n_thr, n]
                prove_acc_arr = prove_correct.sum(axis=1) / n  # [n_thr]

                # Vectorized McNemar: pw = prove_correct & ~deprove_correct, dw = ~prove_correct & deprove_correct
                dep_correct_2d = deprove_correct[None, :]  # [1, n]
                pw_arr = (prove_correct & ~dep_correct_2d).sum(axis=1)  # [n_thr]
                dw_arr = (~prove_correct & dep_correct_2d).sum(axis=1)  # [n_thr]
                pd_sum = pw_arr + dw_arr
                chi2_arr = np.where(pd_sum > 0, (pw_arr - dw_arr) ** 2 / pd_sum, 0.0)
                delta_arr = prove_acc_arr - dep_acc
                sig_arr = (chi2_arr >= 3.84) & (delta_arr > 0)

                for ti in range(n_thr):
                    all_results.append({
                        'score_key': score_key,
                        'combo': combo_name,
                        'vote': vote_type,
                        'threshold': thr_names[ti],
                        'n': int(n),
                        'prove_acc': float(round(float(prove_acc_arr[ti]) * 100, 2)),
                        'deprove_acc': float(round(dep_acc * 100, 2)),
                        'delta': float(round(float(delta_arr[ti]) * 100, 2)),
                        'chi2': float(round(float(chi2_arr[ti]), 2)),
                        'significant': bool(sig_arr[ti]),
                    })

    elapsed = time.time() - t0
    print(f"Evaluation: {len(all_results)} configs in {elapsed:.1f}s")
    return all_results


def evaluate_perlm_configs(precomputed, all_llm_indices, all_labels,
                           score_configs, threshold_configs, top_k=10,
                           max_results=10000):
    """
    Per-LLM config optimization: each LLM uses its own best score config,
    but all share the same threshold. Vectorized with numpy.
    """
    llm_names = sorted(all_llm_indices.keys())
    all_sample_ids = sorted(all_labels.keys())
    n = len(all_sample_ids)
    n_thr = len(threshold_configs)
    thr_names = [name for name, _ in threshold_configs]
    labels_arr = np.array([all_labels[ident] for ident in all_sample_ids], dtype=bool)

    # Step 1: Find top-K score configs per LLM (vectorized)
    print(f"  Finding top-{top_k} score configs per LLM...")
    topk_per_llm = {}

    for llm in llm_names:
        config_accs = []
        for score_key in score_configs:
            prove_probs = np.full(n, np.nan)
            for i, ident in enumerate(all_sample_ids):
                entry = precomputed[llm].get((score_key, ident))
                if entry is not None:
                    prove_probs[i] = entry[0]
            valid = ~np.isnan(prove_probs)
            preds = np.where(valid, prove_probs >= 0.3, ~labels_arr)
            correct = (preds == labels_arr).sum()
            config_accs.append((correct / n, score_key))

        config_accs.sort(reverse=True)
        topk_per_llm[llm] = [sk for _, sk in config_accs[:top_k]]
        print(f"    {llm}: top-{top_k} configs (best single acc={100*config_accs[0][0]:.2f}%)")
        for acc, sk in config_accs[:3]:
            print(f"      {100*acc:.2f}% | {sk}")

    from itertools import product

    llm_combos = []
    for r in range(2, len(llm_names) + 1):
        for combo in combinations(llm_names, r):
            llm_combos.append((sorted(combo), "+".join(sorted(combo))))

    # Use a bounded heap to keep only top results (avoids storing millions of result dicts)
    top_heap = []  # min-heap of (prove_acc, counter, result_dict)
    heap_counter = 0
    min_acc_in_heap = -float('inf')
    total_evaluated = 0
    t0 = time.time()

    for llms, combo_name in llm_combos:
        effective_k = top_k  # Full top_k for ALL ensemble sizes
        per_llm_configs = [topk_per_llm[llm][:effective_k] for llm in llms]
        n_combos = 1
        for c in per_llm_configs:
            n_combos *= len(c)
        print(f"  {combo_name}: {n_combos} score combos × {len(threshold_configs)} thresholds (top_k={effective_k})")
        num_llms = len(llms)

        for score_combo in product(*per_llm_configs):
            score_combo_name = "perlm|" + "|".join(
                f"{llm}:{sk.split('|')[0]}" for llm, sk in zip(llms, score_combo)
            )

            # Build numpy arrays
            prove_arr = np.full((num_llms, n), np.nan)
            deprove_arr = np.full((num_llms, n), np.nan)
            nfacts_arr = np.zeros((num_llms, n))

            for j, llm in enumerate(llms):
                sk = score_combo[j]
                for i, ident in enumerate(all_sample_ids):
                    entry = precomputed[llm].get((sk, ident))
                    if entry is not None:
                        prove_arr[j, i] = entry[0]
                        deprove_arr[j, i] = entry[1]
                        nfacts_arr[j, i] = entry[2]

            valid_mask = ~np.isnan(prove_arr)
            valid_count = valid_mask.sum(axis=0)
            any_valid = valid_count > 0
            vc = np.maximum(valid_count, 1)

            # DePROVE: per-LLM accuracies (deterministic output is binary, so no ensemble averaging)
            per_llm_deprove_acc = {}
            for j, llm in enumerate(llms):
                vm = valid_mask[j]
                dep_pred_j = deprove_arr[j] >= 0.5
                dep_pred_j[~vm] = ~labels_arr[~vm]
                per_llm_deprove_acc[llm] = float((dep_pred_j == labels_arr).sum()) / n

            # For McNemar test, use the best single-LLM DePROVE as baseline
            best_dep_llm = max(per_llm_deprove_acc, key=per_llm_deprove_acc.get)
            dep_pred_best = deprove_arr[llms.index(best_dep_llm)] >= 0.5
            vm_best = valid_mask[llms.index(best_dep_llm)]
            dep_pred_best[~vm_best] = ~labels_arr[~vm_best]
            deprove_correct = (dep_pred_best == labels_arr)
            dep_acc = per_llm_deprove_acc[best_dep_llm]

            # Vectorized across all thresholds at once
            pp_safe = np.where(valid_mask, prove_arr, 0.0)
            avg_pp = pp_safe.sum(axis=0) / vc  # [n]
            nf_safe = np.where(valid_mask, nfacts_arr, 0.0)
            avg_nf = nf_safe.sum(axis=0) / vc  # [n]
            avg_thresholds = compute_thresholds_vectorized(avg_nf, threshold_configs)  # [n_thr, n]

            prove_pred = avg_pp[None, :] >= avg_thresholds  # [n_thr, n]
            prove_pred[:, ~any_valid] = ~labels_arr[~any_valid]

            prove_correct = (prove_pred == labels_arr[None, :])  # [n_thr, n]
            prove_acc_arr = prove_correct.sum(axis=1) / n  # [n_thr]

            dep_correct_2d = deprove_correct[None, :]  # [1, n]
            pw_arr = (prove_correct & ~dep_correct_2d).sum(axis=1)
            dw_arr = (~prove_correct & dep_correct_2d).sum(axis=1)
            pd_sum = pw_arr + dw_arr
            chi2_arr = np.where(pd_sum > 0, (pw_arr - dw_arr) ** 2 / pd_sum, 0.0)
            delta_arr = prove_acc_arr - dep_acc
            sig_arr = (chi2_arr >= 3.84) & (delta_arr > 0)

            # Full per-LLM score config mapping
            score_detail = {llm: sk for llm, sk in zip(llms, score_combo)}

            # Per-LLM DePROVE accuracies rounded
            deprove_per_llm = {llm: round(acc * 100, 2) for llm, acc in per_llm_deprove_acc.items()}

            # Instead of saving ALL n_thr results, find top-5 thresholds for this score combo
            best_ti_indices = np.argsort(prove_acc_arr)[-5:][::-1]
            total_evaluated += n_thr  # Count all thresholds as evaluated

            for ti in best_ti_indices:
                acc_pct = float(round(float(prove_acc_arr[ti]) * 100, 2))

                if len(top_heap) < max_results or acc_pct > min_acc_in_heap:
                    result = {
                        'score_key': score_combo_name,
                        'score_detail': score_detail,
                        'combo': combo_name,
                        'vote': 'perlm_soft',
                        'threshold': thr_names[ti],
                        'n': int(n),
                        'prove_acc': acc_pct,
                        'deprove_acc': float(round(dep_acc * 100, 2)),
                        'deprove_per_llm': deprove_per_llm,
                        'delta': float(round(float(delta_arr[ti]) * 100, 2)),
                        'chi2': float(round(float(chi2_arr[ti]), 2)),
                        'significant': bool(sig_arr[ti]),
                    }
                    if len(top_heap) < max_results:
                        heapq.heappush(top_heap, (acc_pct, heap_counter, result))
                        heap_counter += 1
                        if len(top_heap) == max_results:
                            min_acc_in_heap = top_heap[0][0]
                    else:
                        heapq.heapreplace(top_heap, (acc_pct, heap_counter, result))
                        heap_counter += 1
                        min_acc_in_heap = top_heap[0][0]

    all_results = [item[2] for item in sorted(top_heap, key=lambda x: -x[0])]
    elapsed = time.time() - t0
    print(f"  Per-LLM evaluation: {total_evaluated} threshold evals across all combos, top {len(all_results)} saved, in {elapsed:.1f}s")

    print(f"\n  Top 10 per-LLM optimized configs:")
    print(f"  {'PROVE%':>7} {'DeP%':>7} {'Δ':>7} {'χ²':>6} {'Sig':>4} {'N':>5} | Config")
    for r in all_results[:10]:
        s = '*' if r['significant'] else ''
        print(f"  {r['prove_acc']:>7.2f} {r['deprove_acc']:>7.2f} {r['delta']:>+7.2f} {r['chi2']:>6.1f} {s:>4} {r['n']:>5} | {r['combo']}|{r['vote']}|{r['threshold']}|{r['score_key']}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results):
    sig_results = [r for r in all_results if r['significant']]
    print(f"\n{'='*100}")
    print(f"TOTAL: {len(all_results)} configs evaluated, {len(sig_results)} significant (p<0.05, PROVE > DePROVE)")
    print(f"{'='*100}")

    # Top 30 overall by PROVE accuracy
    sorted_all = sorted(all_results, key=lambda r: (-r['prove_acc'], -r['delta']))
    print(f"\n--- TOP 30 OVERALL (by PROVE accuracy) ---")
    print(f"{'PROVE%':>7} {'DeP%':>7} {'Δ':>7} {'χ²':>6} {'Sig':>4} {'N':>5} | Config")
    for r in sorted_all[:30]:
        s = '*' if r['significant'] else ''
        config = f"{r['combo']}|{r['vote']}|{r['threshold']}|{r['score_key']}"
        print(f"{r['prove_acc']:>7.2f} {r['deprove_acc']:>7.2f} {r['delta']:>+7.2f} {r['chi2']:>6.1f} {s:>4} {r['n']:>5} | {config}")

    # Top 10 significant by delta
    sorted_sig = sorted(sig_results, key=lambda r: -r['delta'])
    if sorted_sig:
        print(f"\n--- TOP 10 SIGNIFICANT (by delta PROVE - DePROVE) ---")
        print(f"{'PROVE%':>7} {'DeP%':>7} {'Δ':>7} {'χ²':>6} {'N':>5} | Config")
        for r in sorted_sig[:10]:
            config = f"{r['combo']}|{r['vote']}|{r['threshold']}|{r['score_key']}"
            print(f"{r['prove_acc']:>7.2f} {r['deprove_acc']:>7.2f} {r['delta']:>+7.2f} {r['chi2']:>6.1f} {r['n']:>5} | {config}")

    # Best per LLM combo
    combos = sorted(set(r['combo'] for r in all_results))
    print(f"\n--- BEST PER LLM COMBO ---")
    print(f"{'Combo':<30} {'PROVE%':>7} {'DeP%':>7} {'Δ':>7} {'χ²':>6} {'Sig':>4} | Config")
    for combo in combos:
        combo_results = [r for r in all_results if r['combo'] == combo]
        best = max(combo_results, key=lambda r: r['prove_acc'])
        s = '*' if best['significant'] else ''
        config = f"{best['vote']}|{best['threshold']}|{best['score_key']}"
        print(f"{combo:<30} {best['prove_acc']:>7.2f} {best['deprove_acc']:>7.2f} {best['delta']:>+7.2f} {best['chi2']:>6.1f} {s:>4} | {config}")

    # Factor analysis: which dimension matters most?
    print(f"\n--- FACTOR ANALYSIS (avg PROVE acc by factor level) ---")

    # By score type (extract from score_key)
    print("\nVQA Score Type:")
    score_accs = defaultdict(list)
    for r in all_results:
        # Extract the score part (before first |)
        score_part = r['score_key'].split('|')[0]
        score_accs[score_part].append(r['prove_acc'])
    for score, accs in sorted(score_accs.items(), key=lambda x: -sum(x[1])/len(x[1])):
        print(f"  {score:<30} avg={sum(accs)/len(accs):.2f}% (n={len(accs)})")

    # By entity prob
    print("\nEntity Probability:")
    ep_accs = defaultdict(list)
    for r in all_results:
        for part in r['score_key'].split('|'):
            if part.startswith('ep='):
                ep_accs[part].append(r['prove_acc'])
    for ep, accs in sorted(ep_accs.items(), key=lambda x: -sum(x[1])/len(x[1])):
        print(f"  {ep:<20} avg={sum(accs)/len(accs):.2f}% (n={len(accs)})")

    # By dampening
    print("\nDampening Alpha:")
    da_accs = defaultdict(list)
    for r in all_results:
        for part in r['score_key'].split('|'):
            if part.startswith('da='):
                da_accs[part].append(r['prove_acc'])
    for da, accs in sorted(da_accs.items(), key=lambda x: -sum(x[1])/len(x[1])):
        print(f"  {da:<20} avg={sum(accs)/len(accs):.2f}% (n={len(accs)})")

    # By agreement mode
    print("\nVQA Agreement:")
    ag_accs = defaultdict(list)
    for r in all_results:
        for part in r['score_key'].split('|'):
            if part.startswith('ag='):
                ag_accs[part].append(r['prove_acc'])
    for ag, accs in sorted(ag_accs.items(), key=lambda x: -sum(x[1])/len(x[1])):
        print(f"  {ag:<20} avg={sum(accs)/len(accs):.2f}% (n={len(accs)})")

    # By threshold type (fixed vs dynamic)
    print("\nThreshold Type:")
    thr_accs = defaultdict(list)
    for r in all_results:
        thr = r['threshold']
        thr_type = 'fixed' if thr.startswith('t=') else 'log' if thr.startswith('log(') else 'linear' if thr.startswith('lin(') else 'binned'
        thr_accs[thr_type].append(r['prove_acc'])
    for tt, accs in sorted(thr_accs.items(), key=lambda x: -sum(x[1])/len(x[1])):
        print(f"  {tt:<20} avg={sum(accs)/len(accs):.2f}% (n={len(accs)})")

    # By vote type
    print("\nVote Type:")
    vote_accs = defaultdict(list)
    for r in all_results:
        vote_accs[r['vote']].append(r['prove_acc'])
    for vt, accs in sorted(vote_accs.items(), key=lambda x: -sum(x[1])/len(x[1])):
        print(f"  {vt:<20} avg={sum(accs)/len(accs):.2f}% (n={len(accs)})")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Parse name=path pairs
    LLM_RESULT_FILES = {}
    for spec in args.files:
        if '=' not in spec:
            print(f"Error: expected name=path, got: {spec}")
            sys.exit(1)
        name, path = spec.split('=', 1)
        LLM_RESULT_FILES[name] = path

    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = time.time()

    # Load all LLM results
    all_llm_indices = {}
    all_ids_by_llm = {}
    all_labels = {}  # universal label lookup: ident → label (includes failed samples)
    for llm_name, path in LLM_RESULT_FILES.items():
        full_path = os.path.join(_PROJECT_ROOT, path)
        if not os.path.exists(full_path):
            print(f"Skipping {llm_name}: {path} not found")
            continue
        with open(full_path) as f:
            data = json.load(f)
        idx = build_llm_index(data)
        all_llm_indices[llm_name] = idx
        all_ids_by_llm[llm_name] = sorted(idx.keys())
        # Build universal label set from VALID samples only (failed excluded from sweep)
        for ident, sample in idx.items():
            label = sample.get('label')
            if label is not None:
                all_labels[ident] = label
        n_failed = sum(1 for s in data if s.get('identifier') and not s.get('success', False) and s.get('label') is not None)
        print(f"Loaded {llm_name}: {len(idx)} valid samples, {n_failed} failed (excluded from sweep)")

    # Build score configs
    score_configs = make_score_configs()
    print(f"\nScore configs: {len(score_configs)}")

    # Build threshold configs
    threshold_configs = make_threshold_configs()
    print(f"Threshold configs: {len(threshold_configs)}")

    # Precompute ProbLog results
    print(f"\n{'='*80}")
    print("PRECOMPUTING ProbLog results...")
    print(f"{'='*80}")
    precomputed = precompute_all(all_llm_indices, all_ids_by_llm, score_configs, OUTPUT_DIR)
    precompute_time = time.time() - start_time
    print(f"\nPrecomputation done in {precompute_time:.1f}s ({precompute_time/60:.1f} min)")

    # Evaluate all configs
    print(f"\n{'='*80}")
    print("EVALUATING all configs...")
    print(f"{'='*80}")
    all_results = evaluate_all(precomputed, all_llm_indices, all_labels, score_configs, threshold_configs)

    # Print summary
    print_summary(all_results)

    # Per-LLM config optimization (shared threshold, different score configs)
    print(f"\n{'='*80}")
    print("PER-LLM CONFIG OPTIMIZATION (shared threshold, per-LLM score config)")
    print(f"{'='*80}")
    perlm_results = evaluate_perlm_configs(precomputed, all_llm_indices, all_labels,
                                            score_configs, threshold_configs)
    all_results.extend(perlm_results)

    # Print summary
    print_summary(all_results)

    # Save results
    output_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
