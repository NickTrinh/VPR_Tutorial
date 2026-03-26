"""
Scaling analysis: How does recognition performance change with reference set size?

Tests filter_n, mean_bad, baseline, and Vysotska with increasing fractions of the
GardensPoint reference set (25%, 50%, 75%, 100%) in both closed-set and open-set.

Usage:
    python experiments/scaling_analysis.py
"""

import os
import sys
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from experiments.experiment_utils import (
    load_cached_descriptors, discover_places, compute_thresholds,
    evaluate, evaluate_vysotska,
)

OUTPUT_DIR = "results/paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

print("Loading data...")
all_ref = load_cached_descriptors("cache/GardensPoint/day_left/eigenplaces", 200)
all_query = load_cached_descriptors("cache/GardensPoint/day_right/eigenplaces", 200)
sfu_descs = load_cached_descriptors("cache/SFU/dry/eigenplaces",
    len(glob("cache/SFU/dry/eigenplaces/img_*_descriptor.pkl")))

fractions = [0.25, 0.50, 0.75, 1.00]
ref_sizes = [int(200 * f) for f in fractions]
methods = ["Baseline", "mean_bad", "filter_n", "Vysotska"]

closed_results = {m: [] for m in methods}
open_results = {m: [] for m in methods}
n_places_list = []

for n_ref in ref_sizes:
    ref_descs = all_ref[:n_ref]
    query_descs = all_query[:n_ref]  # match query range to ref range

    places = discover_places(ref_descs)
    n_places_list.append(len(places))

    thresh_none = {p: -np.inf for p in range(len(places))}
    thresh_mean_bad = compute_thresholds(ref_descs, places, "mean_bad")
    thresh_filter_n = compute_thresholds(ref_descs, places, "filter_n")

    print(f"\n=== Reference size: {n_ref} ({len(places)} places) ===")

    # Closed-set
    print(f"  Closed-set ({n_ref} genuine queries):")
    for method_name, thresh in [("Baseline", thresh_none),
                                 ("mean_bad", thresh_mean_bad),
                                 ("filter_n", thresh_filter_n)]:
        r = evaluate(query_descs, ref_descs, places, thresh, n_ref)
        closed_results[method_name].append(r["F1"])
        print(f"    {method_name:<12} F1={r['F1']:>5.1f}%")

    r = evaluate_vysotska(query_descs, ref_descs, places, n_ref)
    closed_results["Vysotska"].append(r["F1"])
    print(f"    {'Vysotska':<12} F1={r['F1']:>5.1f}%")

    # Open-set: half genuine + SFU distractors
    n_genuine = n_ref // 2
    n_dist = min(n_genuine, len(sfu_descs))
    mixed = np.vstack([query_descs[:n_genuine], sfu_descs[:n_dist]])

    print(f"  Open-set ({n_genuine} genuine + {n_dist} distractors):")
    for method_name, thresh in [("Baseline", thresh_none),
                                 ("mean_bad", thresh_mean_bad),
                                 ("filter_n", thresh_filter_n)]:
        r = evaluate(mixed, ref_descs, places, thresh, n_genuine)
        open_results[method_name].append(r["F1"])
        print(f"    {method_name:<12} F1={r['F1']:>5.1f}%  Rej={r['dist_rej']:>5.1f}%")

    r = evaluate_vysotska(mixed, ref_descs, places, n_genuine)
    open_results["Vysotska"].append(r["F1"])
    print(f"    {'Vysotska':<12} F1={r['F1']:>5.1f}%  Rej={r['dist_rej']:>5.1f}%")

# ─── Visualization ────────────────────────────────────────────────────────────

print("\nGenerating figure...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

method_styles = {
    "Baseline": ("#95a5a6", "o", "--"),
    "mean_bad": ("#3498db", "s", "-"),
    "filter_n": ("#e74c3c", "D", "-"),
    "Vysotska": ("#f39c12", "^", "-"),
}

for method in methods:
    color, marker, ls = method_styles[method]
    ax1.plot(ref_sizes, closed_results[method], color=color, marker=marker,
             linestyle=ls, linewidth=2, markersize=8, label=method)
    ax2.plot(ref_sizes, open_results[method], color=color, marker=marker,
             linestyle=ls, linewidth=2, markersize=8, label=method)

for ax, title in [(ax1, "Closed-Set"), (ax2, "Open-Set (50% distractors)")]:
    ax.set_xlabel("Reference Set Size", fontsize=11)
    ax.set_ylabel("F1 Score (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(ref_sizes)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

# Add place counts as secondary info
for i, (n_ref, n_pl) in enumerate(zip(ref_sizes, n_places_list)):
    ax1.annotate(f"{n_pl}p", (n_ref, 5), ha="center", fontsize=7, color="#7f8c8d")

plt.suptitle("Scaling: Performance vs Reference Set Size",
             fontsize=13, fontweight="bold")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig13_scaling_analysis.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
plt.close()
print(f"Saved: {path}")
