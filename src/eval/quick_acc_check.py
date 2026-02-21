#!/usr/bin/env python3
"""
Quick accuracy check using the optimized v5 per-LLM config.
Standalone — only imports problog, json, math (no src.*).
Uses multiprocessing with per-sample timeouts to avoid hangs.
"""

import json
import math
import sys
import os
import time
import multiprocessing as mp
from collections import defaultdict

from problog.program import PrologString
from problog import get_evaluatable
from problog.evaluator import Semiring


# ─── Config (loaded from configs.json) ─────────────────────────────────────

_CONFIGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs.json")
with open(_CONFIGS_PATH) as _f:
    _ALL_CONFIGS = json.load(_f)

import argparse
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

def quote_arg(arg):
    if not isinstance(arg, str):
        return str(arg)
    if "'" in arg or " " in arg or "-" in arg or (arg and arg[0].isupper()):
        escaped = str(arg).replace("'", "\\'")
        return f"'{escaped}'"
    return arg


def get_score(ev, score_type):
    if score_type == 'avg_blip_qwen_tf':
        b = ev.get('blip_score')
        q = ev.get('qwen_tf_score')
        if b is not None and q is not None:
            return (b + q) / 2.0
        return b if b is not None else q
    return ev.get(score_type)


def build_score_lookups(sample):
    attr_lookup = {}
    for attr in sample.get('evidence', {}).get('attributes', []):
        key = (attr.get('entity_id'), attr.get('value'))
        attr_lookup[key] = attr
    rel_lookup = {}
    for rel in sample.get('evidence', {}).get('relationships', []):
        key = (rel.get('subject_id'), rel.get('object_id'), rel.get('relation'))
        rel_lookup[key] = rel
    return attr_lookup, rel_lookup


def rebuild_facts(sample, attr_score_type, rel_score_type, entity_prob=None, agreement_mode=None):
    stored_facts = sample.get('problog', {}).get('facts', [])
    if not stored_facts:
        return None

    attr_lookup, rel_lookup = build_score_lookups(sample)
    new_facts = []

    dampen_factor = sharpen_factor = None
    if agreement_mode == 'dampen_0.5': dampen_factor = 0.5
    elif agreement_mode == 'dampen_0.3': dampen_factor = 0.3
    elif agreement_mode == 'sharpen': sharpen_factor = 1.5
    elif agreement_mode == 'both': dampen_factor, sharpen_factor = 0.5, 1.5

    for fact in stored_facts:
        pred = fact.get('predicate', '')
        args = fact.get('arguments', [])
        prob = fact.get('probability', 0.5)

        if pred == 'entity':
            if entity_prob is not None:
                prob = entity_prob
        elif pred == 'attribute' and len(args) >= 3:
            ev = attr_lookup.get((args[1], args[2]))
            if ev:
                raw = get_score(ev, attr_score_type)
                if raw is not None: prob = raw
                if dampen_factor is not None or sharpen_factor is not None:
                    blip, qwen = ev.get('blip_score'), ev.get('qwen_tf_score')
                    if blip is not None and qwen is not None:
                        disagree = (blip >= 0.5) != (qwen >= 0.5)
                        if disagree and dampen_factor is not None:
                            prob = 0.5 + (prob - 0.5) * dampen_factor
                        elif not disagree and sharpen_factor is not None:
                            prob = 0.5 + (prob - 0.5) * sharpen_factor
        elif pred == 'relation' and len(args) >= 4:
            ev = rel_lookup.get((args[1], args[2], args[3]))
            if ev:
                raw = get_score(ev, rel_score_type)
                if raw is not None: prob = raw
                if dampen_factor is not None or sharpen_factor is not None:
                    blip, qwen = ev.get('blip_score'), ev.get('qwen_tf_score')
                    if blip is not None and qwen is not None:
                        disagree = (blip >= 0.5) != (qwen >= 0.5)
                        if disagree and dampen_factor is not None:
                            prob = 0.5 + (prob - 0.5) * dampen_factor
                        elif not disagree and sharpen_factor is not None:
                            prob = 0.5 + (prob - 0.5) * sharpen_factor

        new_facts.append({**fact, 'probability': max(1e-7, min(1 - 1e-7, prob))})
    return new_facts


def execute_problog_direct(facts, rules, query, semiring=None):
    lines = []
    for f in facts:
        args_str = ", ".join(quote_arg(a) for a in f['arguments'])
        p = max(1e-7, min(1 - 1e-7, f['probability']))
        lines.append(f"{p}::{f['predicate']}({args_str}).")
    program_str = "\n".join(lines) + f"\n\n{rules}\n\n{query}"
    try:
        if semiring is not None:
            kc_class = get_evaluatable(semiring=semiring)
            program = PrologString(program_str)
            kc = kc_class.create_from(program, semiring=semiring)
            result = kc.evaluate(semiring=semiring)
        else:
            result = get_evaluatable().create_from(PrologString(program_str)).evaluate()
        for atom, prob in result.items():
            if 'answer' in str(atom):
                return float(prob)
        return 0.0
    except:
        return None


def to_bool_label(label):
    if isinstance(label, bool): return label
    if isinstance(label, str): return label.lower() in ('true', 'yes', '1')
    return bool(label)


# ─── Dampened semiring ─────────────────────────────────────────────────────

class SemiringDampened(Semiring):
    def __init__(self, alpha=0.8):
        Semiring.__init__(self)
        self.alpha = alpha
    def one(self): return 1.0
    def zero(self): return 0.0
    def is_one(self, v): return v == 1.0
    def is_zero(self, v): return v == 0.0
    def plus(self, a, b): return 1.0 - (max(0, (1-a)*(1-b)))**self.alpha
    def times(self, a, b): return (max(0, a*b))**self.alpha
    def negate(self, a): return 1.0 - a
    def value(self, a): return float(a)
    def normalize(self, a, z): return a
    def result(self, a, formula=None): return float(a)
    def ad_negate(self, s, pos_only): return self.negate(s)
    def true(self, a=None): return 1.0
    def false(self, a=None): return 0.0
    def is_nsp(self): return True
    def pos_value(self, a, key=None): return float(a)
    def neg_value(self, a, key=None): return 1.0 - float(a)


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

            threshold = max(0, min(1, THRESHOLD_BASE + THRESHOLD_SLOPE * math.log(nf + 1)))
            if (pp is not None and (pp >= threshold) == label):
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
            thr = max(0, min(1, THRESHOLD_BASE + THRESHOLD_SLOPE * math.log(avg_nf + 1)))
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
            thr = max(0, min(1, THRESHOLD_BASE + THRESHOLD_SLOPE * math.log(avg_nf + 1)))
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
