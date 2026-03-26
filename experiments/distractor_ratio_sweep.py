"""
Distractor ratio sweep: How does performance degrade as distractors increase?

Fixes 100 genuine queries and varies the number of SFU distractors from 0 to 400.
Shows filter_n maintains high F1 even with 4:1 distractor:genuine ratio.

Usage:
    python experiments/distractor_ratio_sweep.py
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
ref_descs = load_cached_descriptors("cache/GardensPoint/day_left/eigenplaces", 200)
query_descs = load_cached_descriptors("cache/GardensPoint/day_right/eigenplaces", 200)
sfu_descs = load_cached_descriptors("cache/SFU/dry/eigenplaces",
    len(glob("cache/SFU/dry/eigenplaces/img_*_descriptor.pkl")))

places = discover_places(ref_descs)
print(f"  {len(places)} places")

thresh_none = {p: -np.inf for p in range(len(places))}
thresh_mean_bad = compute_thresholds(ref_descs, places, "mean_bad")
thresh_filter_n = compute_thresholds(ref_descs, places, "filter_n")

n_genuine = 100
genuine = query_descs[:n_genuine]

n_distractors = [0, 25, 50, 100, 150, 200, 300, 400]
methods_data = {
    "Baseline": {"f1": [], "rej": []},
    "mean_bad": {"f1": [], "rej": []},
    "filter_n": {"f1": [], "rej": []},
    "Vysotska": {"f1": [], "rej": []},
}

for n_dist in n_distractors:
    if n_dist == 0:
        mixed = genuine
    else:
        # Cycle SFU if we need more than available
        dist = sfu_descs
        while len(dist) < n_dist:
            dist = np.vstack([dist, sfu_descs])
        mixed = np.vstack([genuine, dist[:n_dist]])

    ratio = n_dist / n_genuine
    print(f"\n--- {n_dist} distractors (ratio {ratio:.1f}:1) ---")

    for method_name, thresh in [("Baseline", thresh_none),
                                 ("mean_bad", thresh_mean_bad),
                                 ("filter_n", thresh_filter_n)]:
        r = evaluate(mixed, ref_descs, places, thresh, n_genuine)
        methods_data[method_name]["f1"].append(r["F1"])
        methods_data[method_name]["rej"].append(r["dist_rej"])
        rej_str = f"Rej={r['dist_rej']:>5.1f}%" if n_dist > 0 else "N/A"
        print(f"  {method_name:<12} F1={r['F1']:>5.1f}%  {rej_str}")

    if n_dist > 0:
        r = evaluate_vysotska(mixed, ref_descs, places, n_genuine)
    else:
        r = evaluate_vysotska(genuine, ref_descs, places, n_genuine)
    methods_data["Vysotska"]["f1"].append(r["F1"])
    methods_data["Vysotska"]["rej"].append(r["dist_rej"])
    rej_str = f"Rej={r['dist_rej']:>5.1f}%" if n_dist > 0 else "N/A"
    print(f"  {'Vysotska':<12} F1={r['F1']:>5.1f}%  {rej_str}")

# ─── Visualization ────────────────────────────────────────────────────────────

print("\nGenerating figure...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

ratios = [d / n_genuine for d in n_distractors]
style = {
    "Baseline": ("#95a5a6", "o", "--"),
    "mean_bad": ("#3498db", "s", "-"),
    "filter_n": ("#e74c3c", "D", "-"),
    "Vysotska": ("#f39c12", "^", "-"),
}

for method in ["Baseline", "mean_bad", "filter_n", "Vysotska"]:
    color, marker, ls = style[method]
    ax1.plot(ratios, methods_data[method]["f1"], color=color, marker=marker,
             linestyle=ls, linewidth=2, markersize=7, label=method)
    # Only plot rejection for non-zero distractors
    ax2.plot(ratios[1:], methods_data[method]["rej"][1:], color=color, marker=marker,
             linestyle=ls, linewidth=2, markersize=7, label=method)

ax1.set_xlabel("Distractor:Genuine Ratio", fontsize=11)
ax1.set_ylabel("F1 Score (%)", fontsize=11)
ax1.set_title("F1 vs Distractor Ratio", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 105)

ax2.set_xlabel("Distractor:Genuine Ratio", fontsize=11)
ax2.set_ylabel("Distractor Rejection Rate (%)", fontsize=11)
ax2.set_title("Rejection Rate vs Distractor Ratio", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 105)

plt.suptitle("Stress Test: Increasing Distractor Ratio",
             fontsize=13, fontweight="bold")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig16_distractor_ratio_sweep.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
plt.close()
print(f"Saved: {path}")
