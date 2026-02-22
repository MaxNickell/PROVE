#!/usr/bin/env python3
"""
Subset analysis — reads pre-computed PROVE/DePROVE predictions from all_results.json.
Shows PROVE + per-LLM DePROVE for each subset.

Usage:
  python analyze_subsets.py llama=eval/v5_test1_llama/all_results.json \
                            maverick=eval/v5_test1_maverick/all_results.json \
                            mistral_large=eval/v5_test1_mistral_large/all_results.json
"""
import json, sys, argparse

# ─── Parse arguments ─────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description='Subset analysis from result files.',
    epilog='Positional args: name=path pairs.')
parser.add_argument('files', nargs='+', metavar='name=path',
                    help='LLM result files as name=path')
parser.add_argument('--output', '-o', type=str, default=None,
                    help='Save results to JSON file')
args = parser.parse_args()

LLM_FILES = {}
for spec in args.files:
    if '=' not in spec:
        parser.error(f"Expected name=path, got: {spec}")
    name, path = spec.split('=', 1)
    LLM_FILES[name] = path

LLM_NAMES = list(LLM_FILES.keys())


def to_bool_label(label):
    if isinstance(label, bool): return label
    if isinstance(label, str): return label.lower() in ('true', 'yes', '1')
    return bool(label)


# ─── Load data ────────────────────────────────────────────────────────────────

print(f"LLMs: {', '.join(LLM_NAMES)}")
print(f"Loading data from {len(LLM_FILES)} LLMs...", flush=True)

all_labels = {}   # ident -> bool label
llm_data = {}     # llm -> {ident: sample} (successful with optimized predictions)

for llm, path in LLM_FILES.items():
    with open(path) as f:
        data = json.load(f)
    n_optimized = 0
    idx = {}
    for s in data:
        ident = s.get('identifier')
        if not ident or s.get('label') is None:
            continue
        all_labels[ident] = to_bool_label(s['label'])
        if s.get('success') and s.get('optimized_prove_pred') is not None:
            s['label'] = to_bool_label(s['label'])
            idx[ident] = s
            n_optimized += 1
    llm_data[llm] = idx
    print(f"  {llm}: {n_optimized} with optimized predictions / {len(data)} total")

all_ids = sorted(all_labels.keys())
n = len(all_ids)
print(f"  n = {n} (total samples in dataset)")

# ─── Build per-sample data ────────────────────────────────────────────────────

print("Building per-sample data...", flush=True)
samples = []
n_missing = 0

for ident in all_ids:
    label = all_labels[ident]

    # Get per-LLM predictions from stored optimized values
    llm_prove_pred = {}
    llm_prove_prob = {}
    llm_dep_pred = {}
    for llm in LLM_NAMES:
        s = llm_data.get(llm, {}).get(ident)
        if s is not None:
            llm_prove_pred[llm] = s['optimized_prove_pred']
            llm_prove_prob[llm] = s.get('optimized_prove_prob')
            llm_dep_pred[llm] = s['optimized_deprove_pred']
        else:
            llm_prove_pred[llm] = None
            llm_prove_prob[llm] = None
            llm_dep_pred[llm] = None

    n_valid = sum(1 for l in LLM_NAMES if llm_prove_pred[l] is not None)
    if n_valid < len(LLM_NAMES):
        n_missing += 1

    # Get a raw sample for feature extraction (from any available LLM)
    raw = None
    for llm in LLM_NAMES:
        raw = llm_data.get(llm, {}).get(ident)
        if raw is not None:
            break
    if raw is None:
        continue

    # PROVE ensemble: perlm_soft — avg probs over valid LLMs
    valid_probs = [llm_prove_prob[l] for l in LLM_NAMES if llm_prove_prob[l] is not None]
    valid_preds = [llm_prove_pred[l] for l in LLM_NAMES if llm_prove_pred[l] is not None]
    if valid_preds:
        # Majority of per-LLM predictions
        prove_pred = sum(valid_preds) > len(valid_preds) / 2
    else:
        prove_pred = not label  # all missing = wrong

    # Per-LLM DePROVE predictions (missing = wrong)
    dep_preds = {l: llm_dep_pred[l] if llm_dep_pred[l] is not None else (not label) for l in LLM_NAMES}
    dep_majority = sum(dep_preds[l] for l in LLM_NAMES) > len(LLM_NAMES) / 2

    # Extract features from raw sample
    rules = raw.get('problog', {}).get('rules', '')
    evidence = raw.get('evidence', {})
    facts = raw.get('problog', {}).get('facts', [])

    n_attr = sum(1 for f in facts if f.get('predicate') == 'attribute')
    n_rel = sum(1 for f in facts if f.get('predicate') == 'relation')
    n_total_facts = len(facts)
    n_vqa_facts = n_attr + n_rel

    fact_probs = [f.get('probability', 0.5) for f in facts if f.get('predicate') != 'entity']
    avg_fact_prob = sum(fact_probs) / max(len(fact_probs), 1) if fact_probs else 0.5

    high_conf = sum(1 for p in fact_probs if p > 0.7 or p < 0.3) if fact_probs else 0
    high_conf_ratio = high_conf / max(len(fact_probs), 1)

    uncertain_count = sum(1 for p in fact_probs if 0.3 <= p <= 0.7) if fact_probs else 0

    has_negation = '\\+' in rules
    has_disjunction = ';' in rules
    n_helper_rules = max(0, rules.count(':-') - 1) if rules else 0

    # VQA agreement: BLIP vs Qwen TF
    agree_tf = 0
    disagree_tf = 0
    for ev_list in [evidence.get('attributes', []), evidence.get('relationships', [])]:
        for ev in ev_list:
            b = ev.get('blip_score')
            q = ev.get('qwen_tf_score')
            if b is not None and q is not None:
                if (b >= 0.5) == (q >= 0.5):
                    agree_tf += 1
                else:
                    disagree_tf += 1
    agreement_tf_ratio = agree_tf / max(agree_tf + disagree_tf, 1)

    # LLM PROVE probability gap
    prove_vals = [llm_prove_prob[l] for l in LLM_NAMES if llm_prove_prob[l] is not None]
    prove_gap = (max(prove_vals) - min(prove_vals)) if len(prove_vals) >= 2 else 0.0

    # LLMs disagree on PROVE direction (using stored per-LLM predictions)
    llm_dirs = [llm_prove_pred[l] for l in LLM_NAMES if llm_prove_pred[l] is not None]
    llms_disagree = bool(llm_dirs) and not (all(llm_dirs) or not any(llm_dirs))

    all_dep_wrong = all(dep_preds[l] != label for l in LLM_NAMES)

    samples.append({
        'ident': ident,
        'label': label,
        'prove_pred': prove_pred,
        'dep_preds': dep_preds,
        'dep_majority': dep_majority,
        'n_total_facts': n_total_facts,
        'has_negation': has_negation,
        'has_disjunction': has_disjunction,
        'n_helper_rules': n_helper_rules,
        'prove_gap': prove_gap,
        'llms_disagree': llms_disagree,
        'all_dep_wrong': all_dep_wrong,
        'avg_fact_prob': avg_fact_prob,
        'high_conf_ratio': high_conf_ratio,
        'uncertain_count': uncertain_count,
        'agreement_tf_ratio': agreement_tf_ratio,
        'n_attr': n_attr,
        'n_rel': n_rel,
        'n_vqa_facts': n_vqa_facts,
    })

print(f"  {len(samples)} samples ({n_missing} partial)\n")

# ─── Subset definitions ──────────────────────────────────────────────────────

SUBSETS = {
    'Label': {
        'True': lambda s: s['label'] == True,
        'False': lambda s: s['label'] == False,
    },
    'Fact Count': {
        'Few (1-5)': lambda s: s['n_total_facts'] <= 5,
        'Medium (6-10)': lambda s: 5 < s['n_total_facts'] <= 10,
        'Many (11+)': lambda s: s['n_total_facts'] > 10,
    },
    'VQA Fact Count': {
        '<=2 VQA facts': lambda s: s['n_vqa_facts'] <= 2,
        '3-5 VQA facts': lambda s: 3 <= s['n_vqa_facts'] <= 5,
        '>=6 VQA facts': lambda s: s['n_vqa_facts'] >= 6,
    },
    'Fact Type': {
        'Attrs only': lambda s: s['n_attr'] > 0 and s['n_rel'] == 0 and s['n_vqa_facts'] > 0,
        'Rels only': lambda s: s['n_rel'] > 0 and s['n_attr'] == 0,
        'Attrs + Rels': lambda s: s['n_attr'] > 0 and s['n_rel'] > 0,
        'Count only': lambda s: s['n_vqa_facts'] == 0,
    },
    'Relation Count': {
        '0 relations': lambda s: s['n_rel'] == 0,
        '1 relation': lambda s: s['n_rel'] == 1,
        '2+ relations': lambda s: s['n_rel'] >= 2,
    },
    'ProbLog Negation': {
        'Contains negation': lambda s: s['has_negation'],
        'No negation': lambda s: not s['has_negation'],
    },
    'ProbLog Disjunction': {
        'Has disjunction': lambda s: s['has_disjunction'],
        'Conjunction only': lambda s: not s['has_disjunction'],
    },
    'Helper Rules': {
        '0 helpers': lambda s: s['n_helper_rules'] == 0,
        '1 helper': lambda s: s['n_helper_rules'] == 1,
        '2+ helpers': lambda s: s['n_helper_rules'] >= 2,
    },
    'Confidence (decisive facts)': {
        'High conf (>70% decisive)': lambda s: s['high_conf_ratio'] > 0.7,
        'Mixed conf': lambda s: 0.3 <= s['high_conf_ratio'] <= 0.7,
        'Low conf (<30% decisive)': lambda s: s['high_conf_ratio'] < 0.3,
    },
    'Avg Fact Probability': {
        'High (>0.7)': lambda s: s['avg_fact_prob'] > 0.7,
        'Medium (0.4-0.7)': lambda s: 0.4 <= s['avg_fact_prob'] <= 0.7,
        'Low (<0.4)': lambda s: s['avg_fact_prob'] < 0.4,
    },
    'Uncertain Facts (0.3-0.7)': {
        '0 uncertain': lambda s: s['uncertain_count'] == 0,
        '1-2 uncertain': lambda s: 1 <= s['uncertain_count'] <= 2,
        '3+ uncertain': lambda s: s['uncertain_count'] >= 3,
    },
    'VQA Agreement (BLIP vs Qwen TF)': {
        'High agree TF (>80%)': lambda s: s['agreement_tf_ratio'] > 0.8,
        'Mod agree TF (50-80%)': lambda s: 0.5 < s['agreement_tf_ratio'] <= 0.8,
        'Low agree TF (<=50%)': lambda s: s['agreement_tf_ratio'] <= 0.5,
    },
    'LLM Probability Gap': {
        'Large gap (>=0.5)': lambda s: s['prove_gap'] >= 0.5,
        'Medium gap (0.2-0.5)': lambda s: 0.2 <= s['prove_gap'] < 0.5,
        'Small gap (<0.2)': lambda s: s['prove_gap'] < 0.2,
    },
    'LLM PROVE Direction': {
        'LLMs disagree (mixed dirs)': lambda s: s['llms_disagree'],
        'LLMs agree': lambda s: not s['llms_disagree'],
    },
    'DePROVE Correctness': {
        'All 3 DePROVE wrong': lambda s: s['all_dep_wrong'],
        'At least 1 DePROVE right': lambda s: not s['all_dep_wrong'],
    },
}


# ─── Compute results ─────────────────────────────────────────────────────────

def compute_subset(subset_samples):
    ns = len(subset_samples)
    if ns == 0:
        return None

    prove_correct = sum(1 for s in subset_samples if s['prove_pred'] == s['label'])
    prove_acc = 100 * prove_correct / ns

    dep_accs = {}
    for llm in LLM_NAMES:
        dep_correct = sum(1 for s in subset_samples if s['dep_preds'][llm] == s['label'])
        dep_accs[llm] = 100 * dep_correct / ns

    maj_correct = sum(1 for s in subset_samples if s['dep_majority'] == s['label'])
    maj_acc = 100 * maj_correct / ns

    best_dep = max(dep_accs.values())

    return {
        'n': ns,
        'prove_acc': round(prove_acc, 2),
        'deprove_per_llm': {llm: round(acc, 2) for llm, acc in dep_accs.items()},
        'deprove_majority_acc': round(maj_acc, 2),
        'deprove_best_acc': round(best_dep, 2),
        'delta_best': round(prove_acc - best_dep, 2),
        'delta_majority': round(prove_acc - maj_acc, 2),
    }


# Compute all results
results = {'llms': LLM_NAMES, 'n_total': n, 'n_samples': len(samples), 'n_partial': n_missing}
results['overall'] = compute_subset(samples)
results['subsets'] = {}

for group_name, subsets in SUBSETS.items():
    results['subsets'][group_name] = {}
    for subset_name, filter_fn in subsets.items():
        filtered = [s for s in samples if filter_fn(s)]
        r = compute_subset(filtered)
        if r is not None:
            results['subsets'][group_name][subset_name] = r

# ─── Print analysis ──────────────────────────────────────────────────────────

def print_row(name, r):
    dep_parts = "  ".join(f"DeP({llm[:3].title()})={r['deprove_per_llm'][llm]:5.1f}%" for llm in LLM_NAMES)
    print(f"  {name:40s} n={r['n']:4d} | PROVE={r['prove_acc']:5.1f}%  {dep_parts}  DeP(Maj)={r['deprove_majority_acc']:5.1f}%")

print(f"{'='*120}")
print(f"SUBSET ANALYSIS")
print(f"  LLMs: {', '.join(LLM_NAMES)}")
print(f"{'='*120}")
print()
print_row("OVERALL", results['overall'])

for group_name, group in results['subsets'].items():
    print(f"\n  --- {group_name} ---")
    for subset_name, r in group.items():
        print_row(subset_name, r)

# ─── PROVE vs best-DePROVE delta per subset ──────────────────────────────────

print(f"\n\n{'='*120}")
print(f"PROVE vs BEST SINGLE-LLM DePROVE (delta)")
print(f"{'='*120}")

def print_delta(name, r):
    print(f"  {name:40s} n={r['n']:4d} | PROVE={r['prove_acc']:5.1f}%  BestDeP={r['deprove_best_acc']:5.1f}%  Maj={r['deprove_majority_acc']:5.1f}%  d(best)={r['delta_best']:+5.1f}%  d(maj)={r['delta_majority']:+5.1f}%")

print()
print_delta("OVERALL", results['overall'])
for group_name, group in results['subsets'].items():
    print(f"\n  --- {group_name} ---")
    for subset_name, r in group.items():
        print_delta(subset_name, r)

# ─── Save to JSON ────────────────────────────────────────────────────────────

if args.output:
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
