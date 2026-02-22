#!/usr/bin/env python3
"""
Quick accuracy check using the optimized v5 per-LLM config.
Uses multiprocessing with per-sample timeouts to avoid hangs.

Archive after VQAv2 is done — run_eval.py Phase 2 does the same thing.
"""

import json
import math
import sys
import os
import time
import multiprocessing as mp
import argparse
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

from src.eval.problog_utils import (
    SemiringDampened, rebuild_facts, execute_problog_direct, threshold_fn,
)


# ─── Config (loaded from configs.json) ─────────────────────────────────────

_CONFIGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.json")
with open(_CONFIGS_PATH) as _f:
    _ALL_CONFIGS = json.load(_f)

_parser = argparse.ArgumentParser(description="Quick accuracy check with optimized config")
_parser.add_argument('base_dir', nargs='?', default='/home/huan2073/PROVE', help='Project root')
_parser.add_argument('dataset', nargs='?', default='vqav2', help='Dataset name')
_parser.add_argument('version', nargs='?', default='v5', help='Version prefix')
_parser.add_argument('--config', default='v5_perlm', choices=list(_ALL_CONFIGS.keys()),
                     help='Scoring config preset (default: v5_perlm)')
_args = _parser.parse_args()

_preset = _ALL_CONFIGS[_args.config]
LLM_CONFIGS = {}
for _llm, _cfg in _preset['llm_configs'].items():
    LLM_CONFIGS[_llm] = {
        'attr_score': _cfg['attr_score_type'],
        'rel_score': _cfg['rel_score_type'],
        'entity_prob': _cfg['entity_prob'],
        'dampening': _cfg['dampened_alpha'],
        'agreement': _cfg['agreement_mode'],
    }

THRESHOLD_BASE = _preset['threshold']['base']
THRESHOLD_SLOPE = _preset['threshold']['slope']

BASE_DIR = _args.base_dir
DATASET = _args.dataset
VERSION = _args.version


# ─── Helpers ───────────────────────────────────────────────────────────────

def to_bool_label(label):
    if isinstance(label, bool): return label
    if isinstance(label, str): return label.lower() in ('true', 'yes', '1')
    return bool(label)


# ─── Parallel worker ──────────────────────────────────────────────────────

def _worker(batch, result_queue, worker_id):
    """Process batch of (ident, facts, rules, query, da) tuples."""
    for item in batch:
        ident, facts, rules, query, da = item
        sr = SemiringDampened(alpha=da) if da != 1.0 else None
        prove_prob = execute_problog_direct(facts, rules, query, semiring=sr)
        dep_facts = [{**f, 'probability': 1.0 if f['probability'] >= 0.5 else 0.0} for f in facts]
        dep_prob = execute_problog_direct(dep_facts, rules, query, semiring=None)
        result_queue.put((worker_id, ident, prove_prob, dep_prob, len(facts)))


def run_parallel(work_items, n_workers=30, batch_size=40, timeout_per_item=15):
    """Run ProbLog evaluations in parallel with timeout protection."""
    result_queue = mp.Queue()
    active = {}
    results = {}
    next_wid = [0]
    completed_by_worker = defaultdict(set)

    def drain():
        while not result_queue.empty():
            try:
                wid, ident, pp, dp, nf = result_queue.get_nowait()
                completed_by_worker[wid].add(ident)
                results[ident] = (pp, dp, nf)
            except:
                break

    def reap():
        for pid in list(active):
            proc, deadline, wid, bidents = active[pid]
            if not proc.is_alive():
                proc.join(timeout=1)
                del active[pid]

    def kill_expired():
        now = time.time()
        for pid in list(active):
            proc, deadline, wid, bidents = active[pid]
            if now > deadline:
                proc.kill()
                proc.join(timeout=5)
                drain()
                completed = completed_by_worker.get(wid, set())
                skipped = [i for i in bidents if i not in completed]
                if skipped:
                    print(f"    TIMEOUT: killed worker, skipped {len(skipped)} items (first: {skipped[0]})")
                del active[pid]

    batch = []
    for item in work_items:
        batch.append(item)
        if len(batch) >= batch_size:
            while len(active) >= n_workers:
                drain(); reap(); kill_expired()
                if len(active) >= n_workers: time.sleep(0.05)
            wid = next_wid[0]; next_wid[0] += 1
            bidents = [it[0] for it in batch]
            deadline = time.time() + len(batch) * 1.0 + timeout_per_item
            p = mp.Process(target=_worker, args=(batch, result_queue, wid))
            p.start()
            active[p.pid] = (p, deadline, wid, bidents)
            batch = []
            drain()

    if batch:
        while len(active) >= n_workers:
            drain(); reap(); kill_expired()
            if len(active) >= n_workers: time.sleep(0.05)
        wid = next_wid[0]; next_wid[0] += 1
        bidents = [it[0] for it in batch]
        deadline = time.time() + len(batch) * 1.0 + timeout_per_item
        p = mp.Process(target=_worker, args=(batch, result_queue, wid))
        p.start()
        active[p.pid] = (p, deadline, wid, bidents)

    while active:
        drain(); reap(); kill_expired()
        if active: time.sleep(0.1)
    drain()
    return results


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    # Load data
    llm_data = {}
    for llm in LLM_CONFIGS:
        path = os.path.join(BASE_DIR, 'eval', f'{VERSION}_{DATASET}_{llm}', 'all_results.json')
        if not os.path.exists(path):
            print(f"  {llm}: not found at {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        idx = {}
        for s in data:
            ident = s.get('identifier')
            if ident and s.get('success', False) and s.get('label') is not None:
                idx[ident] = s
        llm_data[llm] = idx
        print(f"  {llm}: {len(idx)} successful samples")

    if not llm_data:
        print("No data!"); return

    all_idents = set()
    for idx in llm_data.values():
        all_idents |= set(idx.keys())
    common = set.intersection(*[set(idx.keys()) for idx in llm_data.values()])
    print(f"\nUnion: {len(all_idents)}, Common: {len(common)}")

    # ── Phase 1: per-LLM evaluation (parallel) ────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 1: Per-LLM ProbLog evaluation (parallel)")
    print(f"{'='*60}")

    # key = (llm, ident), value = (prove_prob, dep_prob, n_facts)
    all_results = {}

    for llm, cfg in LLM_CONFIGS.items():
        if llm not in llm_data:
            continue
        idx = llm_data[llm]
        print(f"\n  {llm}: building {len(idx)} work items...")

        work_items = []
        for ident in idx:
            sample = idx[ident]
            facts = rebuild_facts(sample, cfg['attr_score'], cfg['rel_score'],
                                  entity_prob=cfg['entity_prob'], agreement_mode=cfg['agreement'])
            if facts is None: continue
            rules = sample.get('problog', {}).get('rules', '')
            query = sample.get('problog', {}).get('query', '')
            if not rules or not query: continue
            work_items.append((ident, facts, rules, query, cfg['dampening']))

        print(f"  {llm}: running {len(work_items)} through ProbLog...")
        t0 = time.time()
        results = run_parallel(work_items)
        elapsed = time.time() - t0
        print(f"  {llm}: done in {elapsed:.1f}s ({len(results)} results)")

        for ident, (pp, dp, nf) in results.items():
            all_results[(llm, ident)] = (pp, dp, nf)

    # ── Phase 2: compute accuracy ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 2: Accuracy with v5 optimized config")
    print(f"Threshold: log(b={THRESHOLD_BASE}, s={THRESHOLD_SLOPE})")
    print(f"{'='*60}")

    # Per-LLM
    for llm in LLM_CONFIGS:
        if llm not in llm_data: continue
        idx = llm_data[llm]
        n_prove = n_dep = n_eval = 0

        for ident in idx:
            label = to_bool_label(idx[ident]['label'])
            key = (llm, ident)
            if key not in all_results: continue
            pp, dp, nf = all_results[key]
            if pp is None and dp is None: continue
            n_eval += 1

            thr = threshold_fn(nf, THRESHOLD_BASE, THRESHOLD_SLOPE)
            if (pp is not None and (pp >= thr) == label):
                n_prove += 1
            elif pp is None and not label:
                n_prove += 1  # None → predict False

            if dp is not None and ((dp >= 0.5) == label):
                n_dep += 1

        print(f"\n  {llm} ({n_eval} evaluated):")
        if n_eval:
            print(f"    PROVE:   {n_prove}/{n_eval} = {n_prove/n_eval*100:.2f}%")
            print(f"    DePROVE: {n_dep}/{n_eval} = {n_dep/n_eval*100:.2f}%")

    # Ensemble on union
    n_total = len(all_idents)
    n_prove_ens = n_dep_ens = 0

    for ident in all_idents:
        label = None
        for idx in llm_data.values():
            if ident in idx:
                label = to_bool_label(idx[ident]['label'])
                break

        prove_probs = []
        dep_preds = []
        nfacts_list = []

        for llm in LLM_CONFIGS:
            key = (llm, ident)
            if key in all_results:
                pp, dp, nf = all_results[key]
                if pp is not None:
                    prove_probs.append(pp)
                    nfacts_list.append(nf)
                dep_preds.append((dp >= 0.5) if dp is not None else False)
            else:
                dep_preds.append(False)

        # PROVE perlm_soft
        if prove_probs:
            avg_p = sum(prove_probs) / len(prove_probs)
            avg_nf = sum(nfacts_list) / len(nfacts_list)
            thr = threshold_fn(avg_nf, THRESHOLD_BASE, THRESHOLD_SLOPE)
            if (avg_p >= thr) == label:
                n_prove_ens += 1
        else:
            if not label:  # all missing → predict False
                n_prove_ens += 1

        # DePROVE majority
        n_true = sum(1 for p in dep_preds if p)
        if (n_true > len(dep_preds) / 2) == label:
            n_dep_ens += 1

    print(f"\n  ENSEMBLE (n={n_total}, union):")
    print(f"    PROVE perlm_soft: {n_prove_ens}/{n_total} = {n_prove_ens/n_total*100:.2f}%")
    print(f"    DePROVE majority: {n_dep_ens}/{n_total} = {n_dep_ens/n_total*100:.2f}%")

    # Ensemble on common
    n_common = len(common)
    n_prove_com = n_dep_com = 0

    for ident in common:
        label = to_bool_label(list(llm_data.values())[0][ident]['label'])

        prove_probs = []
        dep_preds = []
        nfacts_list = []

        for llm in LLM_CONFIGS:
            key = (llm, ident)
            if key in all_results:
                pp, dp, nf = all_results[key]
                if pp is not None:
                    prove_probs.append(pp)
                    nfacts_list.append(nf)
                dep_preds.append((dp >= 0.5) if dp is not None else False)
            else:
                dep_preds.append(False)

        if prove_probs:
            avg_p = sum(prove_probs) / len(prove_probs)
            avg_nf = sum(nfacts_list) / len(nfacts_list)
            thr = threshold_fn(avg_nf, THRESHOLD_BASE, THRESHOLD_SLOPE)
            if (avg_p >= thr) == label:
                n_prove_com += 1
        else:
            if not label:
                n_prove_com += 1

        n_true = sum(1 for p in dep_preds if p)
        if (n_true > len(dep_preds) / 2) == label:
            n_dep_com += 1

    print(f"\n  COMMON SUBSET (n={n_common}):")
    print(f"    PROVE perlm_soft: {n_prove_com}/{n_common} = {n_prove_com/n_common*100:.2f}%")
    print(f"    DePROVE majority: {n_dep_com}/{n_common} = {n_dep_com/n_common*100:.2f}%")

    print(f"\nDone!")


if __name__ == '__main__':
    main()
