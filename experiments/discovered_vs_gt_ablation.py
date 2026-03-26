"""
Ablation: Discovered places vs Ground Truth places.

Compare recognition performance using:
  1. Our auto-discovered places (from online discovery)
  2. Uniform GT places (evenly sized, group-and-step)

This isolates how much the discovery step helps or hurts.

Usage:
    python experiments/discovered_vs_gt_ablation.py
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
    load_cached_descriptors, compute_thresholds, evaluate,
)
from experiments.online_place_discovery import OnlinePlaceDiscovery

OUTPUT_DIR = "results/paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

print("Loading data...")
ref_descs = load_cached_descriptors("cache/GardensPoint/day_left/eigenplaces", 200)
query_descs = load_cached_descriptors("cache/GardensPoint/day_right/eigenplaces", 200)
sfu_descs = load_cached_descriptors("cache/SFU/dry/eigenplaces",
    len(glob("cache/SFU/dry/eigenplaces/img_*_descriptor.pkl")))

# ── Place definitions ──

# 1. Auto-discovered
discoverer = OnlinePlaceDiscovery(min_place_size=3, hysteresis=2, filter_n_cap=10)
for i in range(len(ref_descs)):
    discoverer.process_frame(ref_descs[i], i, verbose=False)
discovered_places = discoverer.places

# 2. Uniform GT: group=10, step=0 (contiguous, no gaps)
uniform_10 = [list(range(i, min(i + 10, 200))) for i in range(0, 200, 10)]

# 3. Uniform GT: group=5, step=0
uniform_5 = [list(range(i, min(i + 5, 200))) for i in range(0, 200, 5)]

# 4. Uniform GT: group=20, step=0
uniform_20 = [list(range(i, min(i + 20, 200))) for i in range(0, 200, 20)]

# 5. RCC-style: group=3, step=10 (sparse, well-separated)
rcc_places = []
idx = 0
while idx + 3 <= 200:
    rcc_places.append(list(range(idx, idx + 3)))
    idx += 3 + 10

place_configs = [
    ("Discovered\n(online, variable)", discovered_places),
    ("Uniform-5\n(40 places)", uniform_5),
    ("Uniform-10\n(20 places)", uniform_10),
    ("Uniform-20\n(10 places)", uniform_20),
    ("RCC g3s10\n(16 places, sparse)", rcc_places),
]

print(f"\nPlace configurations:")
for name, places in place_configs:
    clean = name.replace('\n', ' ')
    total_frames = sum(len(p) for p in places)
    print(f"  {clean:<35} {len(places):>3} places, {total_frames:>4} frames covered")

# ── Evaluate: Closed-set ──
print("\n=== Closed-Set (200 genuine queries) ===")
print(f"  {'Config':<35} {'P':>6} {'R':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4}")
print(f"  {'-'*70}")

closed_results = {}
for name, places in place_configs:
    thresh = compute_thresholds(ref_descs, places, method="filter_n")
    r = evaluate(query_descs, ref_descs, places, thresh)
    closed_results[name] = r
    clean = name.replace('\n', ' ')
    print(f"  {clean:<35} {r['P']:>5.1f}% {r['R']:>5.1f}% {r['F1']:>5.1f}% "
          f"{r['TP']:>4} {r['FP']:>4} {r['FN']:>4}")

# ── Evaluate: Open-set with distractors ──
print("\n=== Open-Set (100 genuine + 100 SFU distractors) ===")
mixed = np.vstack([query_descs[:100], sfu_descs[:100]])
print(f"  {'Config':<35} {'P':>6} {'R':>6} {'F1':>6} {'Dist.Rej':>9}")
print(f"  {'-'*70}")

open_results = {}
for name, places in place_configs:
    thresh = compute_thresholds(ref_descs, places, method="filter_n")
    r = evaluate(mixed, ref_descs, places, thresh, n_genuine=100)
    open_results[name] = r
    clean = name.replace('\n', ' ')
    print(f"  {clean:<35} {r['P']:>5.1f}% {r['R']:>5.1f}% {r['F1']:>5.1f}% "
          f"{r['dist_rej']:>8.1f}%")

# ── Visualization ──
print("\nGenerating figure...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

configs = [name for name, _ in place_configs]
colors = ["#e74c3c", "#3498db", "#27ae60", "#f39c12", "#9b59b6"]

# Left: Closed-set F1
closed_f1 = [closed_results[n]["F1"] for n in configs]
bars1 = ax1.bar(range(len(configs)), closed_f1, color=colors, alpha=0.85, edgecolor="white")
for bar, val in zip(bars1, closed_f1):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}%",
             ha="center", fontsize=9, fontweight="bold")
ax1.set_xticks(range(len(configs)))
ax1.set_xticklabels(configs, fontsize=8)
ax1.set_ylim(0, 110)
ax1.set_ylabel("F1 Score (%)", fontsize=11)
ax1.set_title("Closed-Set (200 genuine queries)", fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.2, axis="y")

# Right: Open-set F1
open_f1 = [open_results[n]["F1"] for n in configs]
bars2 = ax2.bar(range(len(configs)), open_f1, color=colors, alpha=0.85, edgecolor="white")
for bar, val in zip(bars2, open_f1):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}%",
             ha="center", fontsize=9, fontweight="bold")
ax2.set_xticks(range(len(configs)))
ax2.set_xticklabels(configs, fontsize=8)
ax2.set_ylim(0, 110)
ax2.set_ylabel("F1 Score (%)", fontsize=11)
ax2.set_title("Open-Set (100 genuine + 100 SFU distractors)", fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.2, axis="y")

plt.suptitle("Ablation: How Does Place Definition Affect Recognition?",
             fontsize=13, fontweight="bold")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig12_place_ablation.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
plt.close()
print(f"Saved: {path}")
