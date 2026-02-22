#!/usr/bin/env python3
"""
Ablation: replace all VQA fact probabilities with Gaussian random values.
Tests whether PROVE's advantage comes from actual scores or ProbLog structure.
Runs multiple seeds and reports mean +/- std.
"""
import json, math, sys, os, time, signal, argparse
import multiprocessing as mp
import numpy as np

from pathlib import Path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

from src.eval.problog_utils import (
    SemiringDampened, execute_problog_direct, threshold_fn,
)

# ─── Config (loaded from configs.json) ────────────────────────────────────────

_CONFIGS_PATH = os.path.join(os.path.dirname(__file__), "configs.json")
with open(_CONFIGS_PATH) as _f:
    _ALL_CONFIGS = json.load(_f)

# ─── Parse arguments ─────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Gaussian ablation experiment')
parser.add_argument('--config', default='v5_perlm', choices=list(_ALL_CONFIGS.keys()),
                    help='Config preset (default: v5_perlm)')
parser.add_argument('--seeds', type=int, default=20, help='Number of random seeds (default: 20)')
parser.add_argument('--gauss_mean', type=float, default=0.5, help='Gaussian mean (default: 0.5)')
parser.add_argument('--gauss_std', type=float, default=0.2, help='Gaussian std (default: 0.2)')
parser.add_argument('--output', '-o', type=str, default=None,
                    help='Save results to JSON file')
parser.add_argument('files', nargs='+', metavar='name=path',
                    help='LLM result files as name=path')
args = parser.parse_args()

# Parse name=path pairs
LLM_FILES = {}
for spec in args.files:
    if '=' not in spec:
        parser.error(f"Expected name=path, got: {spec}")
    name, path = spec.split('=', 1)
    LLM_FILES[name] = path

LLM_NAMES = list(LLM_FILES.keys())

preset = _ALL_CONFIGS[args.config]
THRESH_BASE = preset['threshold']['base']
THRESH_SLOPE = preset['threshold']['slope']

# Build per-LLM dampening from config
DAMPENING = {}
for llm in LLM_NAMES:
    if llm in preset['llm_configs']:
        DAMPENING[llm] = preset['llm_configs'][llm]['dampened_alpha']
    else:
        DAMPENING[llm] = preset['fallback']['dampened_alpha']


def _alarm_handler(signum, frame):
    raise TimeoutError("ProbLog timed out")


GAUSS_MEAN = args.gauss_mean
GAUSS_STD = args.gauss_std


def _worker(work_args):
    """Process one (llm, sample, seed) combination: replace non-entity fact
    probabilities with Gaussian random values, then run ProbLog to get
    PROVE (soft) and DePROVE (binarized) probabilities."""
    llm, ident, sample, da, seed = work_args

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
            prob = fact.get('probability', 0.5)
        else:
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
    NUM_SEEDS = args.seeds

    t0 = time.time()
    print("=" * 100)
    print(f"GAUSSIAN ABLATION: N({GAUSS_MEAN}, {GAUSS_STD}), {NUM_SEEDS} seeds")
    print(f"Config: {args.config}, LLMs: {', '.join(LLM_NAMES)}")
    print("=" * 100)

    def to_bool_label(label):
        if isinstance(label, bool): return label
        if isinstance(label, str): return label.lower() in ('true', 'yes', '1')
        return bool(label)

    # Load labels and raw samples
    print("\nLoading data...", flush=True)
    all_labels = {}
    llm_raw = {}
    for llm, path in LLM_FILES.items():
        with open(path) as f:
            data = json.load(f)
        idx = {}
        for s in data:
            ident = s.get('identifier')
            if not ident or s.get('label') is None:
                continue
            all_labels[ident] = to_bool_label(s['label'])
            if s.get('success'):
                idx[ident] = s
        llm_raw[llm] = idx
        print(f"  {llm}: {len(idx)} success / {len(data)} total")

    all_ids = sorted(all_labels.keys())
    n = len(all_ids)
    labels_arr = np.array([all_labels[ident] for ident in all_ids], dtype=bool)
    print(f"  n = {n}")

    # Build work items: all LLMs x all samples x all seeds
    work_items = []
    for seed in range(NUM_SEEDS):
        for llm in LLM_NAMES:
            da = DAMPENING[llm]
            for ident in all_ids:
                sample = llm_raw.get(llm, {}).get(ident)
                if sample is not None:
                    work_items.append((llm, ident, sample, da, seed))

    print(f"  {len(work_items)} work items ({NUM_SEEDS} seeds x {len(LLM_NAMES)} LLMs x ~{n} samples)")
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
        n_llms = len(LLM_NAMES)
        prove_arr = np.full((n_llms, n), np.nan)
        deprove_arr = np.full((n_llms, n), np.nan)
        nfacts_arr = np.zeros((n_llms, n))

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
        majority_pred = dep_votes.sum(axis=0) >= len(LLM_NAMES) / 2
        dep_majority_accs.append((majority_pred == labels_arr).sum() / n)

    # Report
    print("=" * 100)
    print(f"  RESULTS: Gaussian N({GAUSS_MEAN}, {GAUSS_STD}), {NUM_SEEDS} seeds")
    print("=" * 100)

    prove_mean = 100 * np.mean(prove_accs)
    prove_std = 100 * np.std(prove_accs)
    print(f"\n  PROVE (perlm_soft):      {prove_mean:.2f}% +/- {prove_std:.2f}%")

    for llm in LLM_NAMES:
        m = 100 * np.mean(dep_accs_per_llm[llm])
        s = 100 * np.std(dep_accs_per_llm[llm])
        print(f"  DePROVE ({llm:14s}): {m:.2f}% +/- {s:.2f}%")

    maj_m = 100 * np.mean(dep_majority_accs)
    maj_s = 100 * np.std(dep_majority_accs)
    print(f"  DePROVE (Majority Vote): {maj_m:.2f}% +/- {maj_s:.2f}%")

    print(f"\n  Per-seed PROVE: {['%.1f' % (100*a) for a in prove_accs]}")
    print(f"\nTotal time: {time.time()-t0:.0f}s")

    # Save to JSON
    if args.output:
        out = {
            'gaussian': {'mean': GAUSS_MEAN, 'std': GAUSS_STD},
            'config': args.config,
            'llms': LLM_NAMES,
            'n': n,
            'num_seeds': NUM_SEEDS,
            'prove': {'mean': round(prove_mean, 2), 'std': round(prove_std, 2),
                      'per_seed': [round(100 * a, 2) for a in prove_accs]},
            'deprove_per_llm': {
                llm: {'mean': round(100 * np.mean(dep_accs_per_llm[llm]), 2),
                       'std': round(100 * np.std(dep_accs_per_llm[llm]), 2)}
                for llm in LLM_NAMES
            },
            'deprove_majority': {'mean': round(maj_m, 2), 'std': round(maj_s, 2)},
        }
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output}")
