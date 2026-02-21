#!/usr/bin/env python3
"""
Comprehensive results printer for the PROVE project.
Reads all experiment result JSON files and prints formatted ASCII tables.
"""

import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE = _PROJECT_ROOT / "eval"

# ============================================================================
# Utility functions
# ============================================================================

def load_json(path):
    """Load a JSON file, return None if it doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def hline(width=100, char="="):
    return char * width


def section(title, width=100):
    print()
    print(hline(width))
    print(f"  {title}")
    print(hline(width))


def subsection(title, width=100):
    print()
    print(f"  --- {title} ---")


def fmt_pct(val, width=7):
    """Format a float as percentage string."""
    if val is None:
        return "   N/A ".rjust(width)
    return f"{val*100:.2f}%".rjust(width)


def fmt_float(val, width=8):
    if val is None:
        return "    N/A ".rjust(width)
    return f"{val:.4f}".rjust(width)


def fmt_int(val, width=6):
    if val is None:
        return "  N/A ".rjust(width)
    return str(val).rjust(width)


# ============================================================================
# 1. Baselines: compute from raw samples
# ============================================================================

def compute_baseline(data, name):
    """Compute PROVE/DePROVE accuracy from raw sample results."""
    n_total = len(data)
    n_success = 0
    n_valid = 0
    prove_correct = 0
    deprove_correct = 0
    # For McNemar: count discordant pairs
    prove_right_deprove_wrong = 0
    prove_wrong_deprove_right = 0

    for item in data:
        if not item.get("success"):
            continue
        n_success += 1

        results = item.get("results", {})
        if results.get("prove_prob") is None:
            continue
        n_valid += 1

        label = item["label"]  # bool
        prove_ans = results.get("prove_answer", "")
        deprove_ans = results.get("deprove_answer", "")

        # Convert string answers to bool
        prove_pred = prove_ans.strip().lower() == "true" if isinstance(prove_ans, str) else bool(prove_ans)
        deprove_pred = deprove_ans.strip().lower() == "true" if isinstance(deprove_ans, str) else bool(deprove_ans)

        p_correct = (prove_pred == label)
        d_correct = (deprove_pred == label)

        if p_correct:
            prove_correct += 1
        if d_correct:
            deprove_correct += 1

        if p_correct and not d_correct:
            prove_right_deprove_wrong += 1
        elif not p_correct and d_correct:
            prove_wrong_deprove_right += 1

    prove_acc = prove_correct / n_valid if n_valid > 0 else None
    deprove_acc = deprove_correct / n_valid if n_valid > 0 else None

    # McNemar chi2 (without continuity correction)
    b = prove_right_deprove_wrong
    c = prove_wrong_deprove_right
    if (b + c) > 0:
        chi2 = (b - c) ** 2 / (b + c)
    else:
        chi2 = 0.0

    return {
        "name": name,
        "n_total": n_total,
        "n_success": n_success,
        "n_valid": n_valid,
        "prove_acc": prove_acc,
        "deprove_acc": deprove_acc,
        "delta": (prove_acc - deprove_acc) if prove_acc is not None and deprove_acc is not None else None,
        "chi2": chi2,
        "prove_wins": b,
        "deprove_wins": c,
    }


def print_baselines():
    section("1. BASELINE RESULTS (raw sample computation)")

    baselines = [
        (BASE / "v3_baseline_maverick" / "all_results.json", "Maverick (v3)"),
        (BASE / "v3_baseline_llama" / "all_results.json", "Llama 3.3 70B (v3)"),
        (BASE / "nova_pro_cot" / "all_results.json", "Nova Pro CoT"),
    ]

    results = []
    for path, name in baselines:
        data = load_json(path)
        if data is None:
            print(f"  [MISSING] {path}")
            continue
        r = compute_baseline(data, name)
        results.append(r)

    if not results:
        print("  No baseline files found.")
        return

    # Print table
    header = f"  {'Model':<22s} {'N':>6s} {'PROVE':>8s} {'DePROVE':>8s} {'Delta':>8s} {'Chi2':>8s} {'Sig?':>5s} {'P>D':>5s} {'D>P':>5s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        sig = "Yes" if r["chi2"] > 3.84 else "No"
        delta_str = fmt_pct(r["delta"])
        print(f"  {r['name']:<22s} {r['n_valid']:>6d} {fmt_pct(r['prove_acc'])} {fmt_pct(r['deprove_acc'])} {delta_str} {r['chi2']:>8.3f} {sig:>5s} {r['prove_wins']:>5d} {r['deprove_wins']:>5d}")

    print()
    print("  Legend: N=valid samples, Delta=PROVE-DePROVE, Chi2=McNemar, Sig=Chi2>3.84, P>D=prove-only-correct, D>P=deprove-only-correct")


# ============================================================================
# Config analysis helpers
# ============================================================================

def analyze_configs(data, name, chi2_key="chi2"):
    """Analyze a configurations dict. Returns summary and top configs."""
    configs = data.get("configurations", {})
    n_total = len(configs)
    n_sig_prove = 0  # chi2 > 3.84 AND prove_acc > deprove_acc
    n_sig_deprove = 0  # chi2 > 3.84 AND deprove_acc > prove_acc

    all_configs = []
    for key, c in configs.items():
        chi2 = c.get(chi2_key, c.get("chi2", 0))
        prove_acc = c.get("prove_acc", 0)
        deprove_acc = c.get("deprove_acc", 0)
        delta = prove_acc - deprove_acc
        n = c.get("n_samples", 0)

        if chi2 > 3.84:
            if prove_acc > deprove_acc:
                n_sig_prove += 1
            elif deprove_acc > prove_acc:
                n_sig_deprove += 1

        all_configs.append({
            "key": key,
            "prove_acc": prove_acc,
            "deprove_acc": deprove_acc,
            "delta": delta,
            "chi2": chi2,
            "n": n,
        })

    # Sort by delta (PROVE > DePROVE)
    by_delta = sorted(all_configs, key=lambda x: x["delta"], reverse=True)
    # Sort by chi2 where PROVE > DePROVE
    prove_better = [c for c in all_configs if c["delta"] > 0]
    by_chi2 = sorted(prove_better, key=lambda x: x["chi2"], reverse=True)

    return {
        "name": name,
        "n_total": n_total,
        "n_sig_prove": n_sig_prove,
        "n_sig_deprove": n_sig_deprove,
        "top_by_delta": by_delta[:5],
        "top_by_chi2": by_chi2[:5],
    }


def print_config_table(analysis, show_top=True):
    """Print analysis for a configurations experiment."""
    a = analysis
    subsection(f"{a['name']} ({a['n_total']} configs)")
    print(f"  Total configs: {a['n_total']}")
    print(f"  Significant PROVE > DePROVE (chi2>3.84): {a['n_sig_prove']}")
    print(f"  Significant DePROVE > PROVE (chi2>3.84): {a['n_sig_deprove']}")

    if show_top and a["top_by_delta"]:
        print()
        print(f"  Top 5 by gap (PROVE - DePROVE):")
        header = f"    {'Config':<72s} {'PROVE':>7s} {'DePROVE':>8s} {'Delta':>7s} {'Chi2':>8s} {'N':>5s}"
        print(header)
        print("    " + "-" * (len(header) - 4))
        for c in a["top_by_delta"]:
            key = c["key"][:70]
            print(f"    {key:<72s} {fmt_pct(c['prove_acc'])} {fmt_pct(c['deprove_acc'])} {fmt_pct(c['delta'])} {c['chi2']:>8.3f} {c['n']:>5d}")

    if show_top and a["top_by_chi2"]:
        print()
        print(f"  Top 5 by chi2 (PROVE > DePROVE only):")
        header = f"    {'Config':<72s} {'PROVE':>7s} {'DePROVE':>8s} {'Delta':>7s} {'Chi2':>8s} {'N':>5s}"
        print(header)
        print("    " + "-" * (len(header) - 4))
        for c in a["top_by_chi2"]:
            key = c["key"][:70]
            print(f"    {key:<72s} {fmt_pct(c['prove_acc'])} {fmt_pct(c['deprove_acc'])} {fmt_pct(c['delta'])} {c['chi2']:>8.3f} {c['n']:>5d}")


# ============================================================================
# 2. Threshold sweep
# ============================================================================

def print_threshold_sweep():
    section("2. v3_threshold_sweep RESULTS")

    files = [
        (BASE / "v3_threshold_sweep" / "maverick_results.json", "Maverick"),
        (BASE / "v3_threshold_sweep" / "llama_results.json", "Llama 3.3 70B"),
    ]

    for path, name in files:
        data = load_json(path)
        if data is None:
            print(f"  [MISSING] {path}")
            continue

        thresholds = data.get("thresholds", [])
        subsection(f"{name} ({len(thresholds)} thresholds)")

        header = f"  {'Thresh':>7s} {'PROVE':>8s} {'DePROVE':>8s} {'Delta':>8s} {'Chi2':>8s} {'Sig?':>5s} {'P>D':>5s} {'D>P':>5s} {'N':>5s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for t in thresholds:
            thr = t.get("threshold", t.get("prove_threshold", "?"))
            prove = t.get("prove_acc", 0)
            deprove = t.get("deprove_acc", 0)
            delta = prove - deprove
            chi2 = t.get("chi2", 0)
            sig = "Yes" if chi2 > 3.84 else "No"
            pw = t.get("prove_wins", 0)
            dw = t.get("deprove_wins", 0)
            n = t.get("n_samples", 0)
            print(f"  {thr:>7.2f} {fmt_pct(prove)} {fmt_pct(deprove)} {fmt_pct(delta)} {chi2:>8.3f} {sig:>5s} {pw:>5d} {dw:>5d} {n:>5d}")


# ============================================================================
# 3. Dynamic threshold (BUGGY)
# ============================================================================

def print_dynamic_threshold():
    section("3. v3_dynamic_threshold RESULTS [BUGGY -- DePROVE uses dynamic threshold instead of fixed 0.5]")

    files = [
        (BASE / "v3_dynamic_threshold" / "maverick_results.json", "Maverick"),
        (BASE / "v3_dynamic_threshold" / "llama_results.json", "Llama 3.3 70B"),
    ]

    for path, name in files:
        data = load_json(path)
        if data is None:
            print(f"  [MISSING] {path}")
            continue

        strategies = data.get("strategies", [])
        subsection(f"{name} [BUGGY] ({len(strategies)} strategies)")

        header = f"  {'Strategy':<50s} {'PROVE':>8s} {'DePROVE':>8s} {'Delta':>8s} {'Chi2':>8s} {'Sig?':>5s} {'N':>5s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for s in strategies:
            strat = s.get("strategy", "?")[:48]
            prove = s.get("prove_acc", 0)
            deprove = s.get("deprove_acc", 0)
            delta = prove - deprove
            chi2 = s.get("chi2", 0)
            sig = "Yes" if chi2 > 3.84 else "No"
            n = s.get("n_samples", 0)
            print(f"  {strat:<50s} {fmt_pct(prove)} {fmt_pct(deprove)} {fmt_pct(delta)} {chi2:>8.3f} {sig:>5s} {n:>5d}")


# ============================================================================
# 4. Combined
# ============================================================================

def print_combined():
    section("4. v3_combined RESULTS (3645 configs, score + threshold sweep)")

    files = [
        (BASE / "v3_combined" / "maverick_results.json", "Maverick"),
        (BASE / "v3_combined" / "llama_results.json", "Llama 3.3 70B"),
        (BASE / "v3_combined" / "nova_pro_cot_results.json", "Nova Pro CoT"),
    ]

    for path, name in files:
        data = load_json(path)
        if data is None:
            print(f"  [MISSING] {path}")
            continue
        a = analyze_configs(data, name)
        print_config_table(a)


# ============================================================================
# 5. Power entity
# ============================================================================

def print_power_entity():
    section("5. v3_power_entity RESULTS (4800 configs, power-scaled + entity filtering)")

    files = [
        (BASE / "v3_power_entity" / "maverick_results.json", "Maverick"),
        (BASE / "v3_power_entity" / "llama_results.json", "Llama 3.3 70B"),
        (BASE / "v3_power_entity" / "nova_pro_cot_results.json", "Nova Pro CoT"),
    ]

    for path, name in files:
        data = load_json(path)
        if data is None:
            print(f"  [MISSING] {path}")
            continue
        a = analyze_configs(data, name)
        print_config_table(a)


# ============================================================================
# Grand summary
# ============================================================================

def print_grand_summary():
    section("GRAND SUMMARY: Best Results Across All Experiments")

    print()
    print("  Collecting best PROVE > DePROVE configs across all experiments...")
    print()

    all_best = []

    # Baselines
    baselines = [
        (BASE / "v3_baseline_maverick" / "all_results.json", "Baseline/Maverick"),
        (BASE / "v3_baseline_llama" / "all_results.json", "Baseline/Llama"),
        (BASE / "nova_pro_cot" / "all_results.json", "Baseline/NovaPro"),
    ]
    for path, name in baselines:
        data = load_json(path)
        if data is None:
            continue
        r = compute_baseline(data, name)
        all_best.append({
            "experiment": name,
            "config": "(default threshold=0.5)",
            "prove_acc": r["prove_acc"],
            "deprove_acc": r["deprove_acc"],
            "delta": r["delta"],
            "chi2": r["chi2"],
            "n": r["n_valid"],
        })

    # Config-based experiments
    config_files = [
        (BASE / "v3_combined" / "maverick_results.json", "Combined/Maverick"),
        (BASE / "v3_combined" / "llama_results.json", "Combined/Llama"),
        (BASE / "v3_combined" / "nova_pro_cot_results.json", "Combined/NovaPro"),
        (BASE / "v3_power_entity" / "maverick_results.json", "PowerEntity/Maverick"),
        (BASE / "v3_power_entity" / "llama_results.json", "PowerEntity/Llama"),
        (BASE / "v3_power_entity" / "nova_pro_cot_results.json", "PowerEntity/NovaPro"),
    ]
    for path, exp_name in config_files:
        data = load_json(path)
        if data is None:
            continue
        configs = data.get("configurations", {})
        # Find best by delta where delta > 0
        best_delta = None
        best_chi2 = None
        for key, c in configs.items():
            prove = c.get("prove_acc", 0)
            deprove = c.get("deprove_acc", 0)
            delta = prove - deprove
            chi2 = c.get("chi2", 0)
            n = c.get("n_samples", 0)

            if delta > 0:
                if best_delta is None or delta > best_delta["delta"]:
                    best_delta = {"config": key, "prove_acc": prove, "deprove_acc": deprove, "delta": delta, "chi2": chi2, "n": n}
                if chi2 > 3.84 and (best_chi2 is None or chi2 > best_chi2["chi2"]):
                    best_chi2 = {"config": key, "prove_acc": prove, "deprove_acc": deprove, "delta": delta, "chi2": chi2, "n": n}

        if best_delta:
            all_best.append({"experiment": exp_name + " (best gap)", **best_delta})
        if best_chi2:
            all_best.append({"experiment": exp_name + " (best chi2)", **best_chi2})

    # Threshold sweep best
    sweep_files = [
        (BASE / "v3_threshold_sweep" / "maverick_results.json", "ThreshSweep/Maverick"),
        (BASE / "v3_threshold_sweep" / "llama_results.json", "ThreshSweep/Llama"),
    ]
    for path, exp_name in sweep_files:
        data = load_json(path)
        if data is None:
            continue
        thresholds = data.get("thresholds", [])
        best = None
        for t in thresholds:
            prove = t.get("prove_acc", 0)
            deprove = t.get("deprove_acc", 0)
            delta = prove - deprove
            chi2 = t.get("chi2", 0)
            thr = t.get("threshold", t.get("prove_threshold", "?"))
            n = t.get("n_samples", 0)
            if delta > 0 and (best is None or delta > best["delta"]):
                best = {"config": f"t={thr}", "prove_acc": prove, "deprove_acc": deprove, "delta": delta, "chi2": chi2, "n": n}
        if best:
            all_best.append({"experiment": exp_name, **best})

    # Sort by delta
    all_best.sort(key=lambda x: x.get("delta", 0) or 0, reverse=True)

    header = f"  {'Experiment':<35s} {'Config':<45s} {'PROVE':>7s} {'DePROVE':>8s} {'Delta':>7s} {'Chi2':>8s} {'Sig':>4s} {'N':>5s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in all_best[:20]:
        sig = "*" if r.get("chi2", 0) > 3.84 else ""
        config = r.get("config", "")[:43]
        print(f"  {r['experiment']:<35s} {config:<45s} {fmt_pct(r['prove_acc'])} {fmt_pct(r['deprove_acc'])} {fmt_pct(r.get('delta'))} {r.get('chi2',0):>8.3f} {sig:>4s} {r.get('n',0):>5d}")

    print()
    print("  * = statistically significant (McNemar chi2 > 3.84, p < 0.05)")


# ============================================================================
# Main
# ============================================================================

def main():
    print(hline(120, "="))
    print("  PROVE PROJECT -- COMPREHENSIVE EXPERIMENT RESULTS")
    print(f"  Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(hline(120, "="))

    print_baselines()
    print_threshold_sweep()
    print_dynamic_threshold()
    print_combined()
    print_power_entity()
    print_grand_summary()

    print()
    print(hline(120, "="))
    print("  END OF RESULTS")
    print(hline(120, "="))


if __name__ == "__main__":
    main()
