"""
Test filter_n with multiple distractor sources to show robustness.

Distractor sources:
  1. SFU dry (campus, completely different)
  2. Nordland summer (outdoor/railway, more visually similar to GardensPoint)

Reference: GardensPoint day_left (200 images)
Genuine: GardensPoint day_right (100 images)

Usage:
    python experiments/multi_distractor_test.py
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
nordland_descs = load_cached_descriptors("cache/Nordland_filtered/summer/eigenplaces", 100)

print(f"  Ref: {len(ref_descs)}, Query: {len(query_descs)}, SFU: {len(sfu_descs)}, Nordland: {len(nordland_descs)}")

print("Discovering places...")
places = discover_places(ref_descs)
print(f"  {len(places)} places")

thresh_none = {p: -np.inf for p in range(len(places))}
thresh_mean_bad = compute_thresholds(ref_descs, places, "mean_bad")
thresh_filter_n = compute_thresholds(ref_descs, places, "filter_n")

n_genuine = 100
genuine = query_descs[:n_genuine]

distractor_sources = {
    "SFU (different campus)": sfu_descs[:100],
    "Nordland (different country)": nordland_descs[:100],
    "SFU + Nordland (mixed)": np.vstack([sfu_descs[:50], nordland_descs[:50]]),
}

all_results = {}

for dist_name, dist_descs in distractor_sources.items():
    n_dist = len(dist_descs)
    mixed = np.vstack([genuine, dist_descs])
    print(f"\n--- Distractors: {dist_name} ({n_dist} images) ---")

    for method_name, thresh in [("Baseline", thresh_none),
                                 ("mean_bad", thresh_mean_bad),
                                 ("filter_n", thresh_filter_n)]:
        r = evaluate(mixed, ref_descs, places, thresh, n_genuine)
        key = f"{dist_name}|{method_name}"
        all_results[key] = r
        print(f"  {method_name:<12} P={r['P']:>5.1f}%  R={r['R']:>5.1f}%  F1={r['F1']:>5.1f}%  "
              f"Dist.Rej={r['dist_rej']:>5.1f}%")

    r = evaluate_vysotska(mixed, ref_descs, places, n_genuine)
    all_results[f"{dist_name}|Vysotska"] = r
    print(f"  {'Vysotska':<12} P={r['P']:>5.1f}%  R={r['R']:>5.1f}%  F1={r['F1']:>5.1f}%  "
          f"Dist.Rej={r['dist_rej']:>5.1f}%")

# ─── Visualization ────────────────────────────────────────────────────────────

print("\nGenerating figure...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
methods = ["Baseline", "mean_bad", "filter_n", "Vysotska"]
method_colors = ["#95a5a6", "#3498db", "#e74c3c", "#f39c12"]

for ax_idx, (dist_name, _) in enumerate(distractor_sources.items()):
    ax = axes[ax_idx]
    f1_vals = [all_results[f"{dist_name}|{m}"]["F1"] for m in methods]
    rej_vals = [all_results[f"{dist_name}|{m}"]["dist_rej"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35
    bars1 = ax.bar(x - width/2, f1_vals, width, color=method_colors, alpha=0.85,
                    edgecolor="white", label="F1" if ax_idx == 0 else None)
    bars2 = ax.bar(x + width/2, rej_vals, width, color=method_colors, alpha=0.4,
                    edgecolor=method_colors, linewidth=1.5, hatch="//",
                    label="Dist. Rejection" if ax_idx == 0 else None)

    for bar, val in zip(bars1, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.0f}",
                ha="center", fontsize=8, fontweight="bold")
    for bar, val in zip(bars2, rej_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.0f}",
                ha="center", fontsize=7, color="#7f8c8d")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_title(f"Distractors: {dist_name}", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    if ax_idx == 0:
        ax.set_ylabel("Percentage (%)", fontsize=11)

axes[0].legend(fontsize=8, loc="lower right")

plt.suptitle("Robustness Across Distractor Sources: F1 and Distractor Rejection Rate",
             fontsize=13, fontweight="bold")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig11_multi_distractor.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
plt.close()
print(f"Saved: {path}")
