"""
Visualize per-place threshold statistics.

Shows for each discovered place:
  - mean_good (intra-place similarity)
  - mean_bad (inter-place similarity)
  - computed threshold (mean_bad + filter_n * std_bad)
  - filter_n value
  - place size

Usage:
    python experiments/per_place_threshold_viz.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from experiments.experiment_utils import (
    load_cached_descriptors, discover_places, compute_place_stats,
)

OUTPUT_DIR = "results/paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

print("Loading data...")
ref_descs = load_cached_descriptors("cache/GardensPoint/day_left/eigenplaces", 200)
places = discover_places(ref_descs)
stats = compute_place_stats(ref_descs, places)

print(f"\n{'Place':>5} {'Frames':>10} {'Size':>4} {'mean_good':>10} {'mean_bad':>9} "
      f"{'std_bad':>8} {'filter_n':>8} {'threshold':>10}")
print("-" * 75)
for i, s in enumerate(stats):
    print(f"  {i:>3}  {s.get('frames','?'):>10}  {s['size']:>3}  {s['mean_good']:>9.4f}  "
          f"{s['mean_bad']:>8.4f}  {s['std_bad']:>7.4f}  {s['filter_n']:>7}  {s['threshold']:>9.4f}")

# ─── Visualization ────────────────────────────────────────────────────────────

print("\nGenerating figure...")
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
n = len(places)
x = np.arange(n)

# Top-left: mean_good vs mean_bad vs threshold
ax = axes[0, 0]
ax.bar(x - 0.25, [s["mean_good"] for s in stats], 0.25, color="#27ae60", alpha=0.8, label="mean_good (intra)")
ax.bar(x, [s["mean_bad"] for s in stats], 0.25, color="#e74c3c", alpha=0.8, label="mean_bad (inter)")
ax.bar(x + 0.25, [s["threshold"] for s in stats], 0.25, color="#3498db", alpha=0.8, label="threshold")
ax.set_xlabel("Place Index")
ax.set_ylabel("Cosine Similarity")
ax.set_title("Per-Place: Intra vs Inter Similarity & Threshold", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis="y")
ax.set_xticks(x)

# Top-right: filter_n values
ax = axes[0, 1]
bars = ax.bar(x, [s["filter_n"] for s in stats], color="#9b59b6", alpha=0.85, edgecolor="white")
for bar, s in zip(bars, stats):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            str(s["filter_n"]), ha="center", fontsize=8, fontweight="bold")
ax.set_xlabel("Place Index")
ax.set_ylabel("filter_n")
ax.set_title("Per-Place: Computed filter_n Values", fontweight="bold")
ax.grid(True, alpha=0.2, axis="y")
ax.set_xticks(x)
ax.axhline(y=10, color="#e74c3c", linestyle="--", alpha=0.5, label="cap=10")
ax.legend(fontsize=8)

# Bottom-left: place sizes
ax = axes[1, 0]
sizes = [s["size"] for s in stats]
bars = ax.bar(x, sizes, color="#f39c12", alpha=0.85, edgecolor="white")
for bar, sz in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            str(sz), ha="center", fontsize=8, fontweight="bold")
ax.set_xlabel("Place Index")
ax.set_ylabel("Number of Frames")
ax.set_title("Per-Place: Place Size (# frames)", fontweight="bold")
ax.grid(True, alpha=0.2, axis="y")
ax.set_xticks(x)

# Bottom-right: separation gap (mean_good - threshold) — positive = safe margin
ax = axes[1, 1]
gaps = [s["mean_good"] - s["threshold"] for s in stats]
colors = ["#27ae60" if g > 0 else "#e74c3c" for g in gaps]
bars = ax.bar(x, gaps, color=colors, alpha=0.85, edgecolor="white")
ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_xlabel("Place Index")
ax.set_ylabel("Gap (mean_good − threshold)")
ax.set_title("Per-Place: Safety Margin (positive = good)", fontweight="bold")
ax.grid(True, alpha=0.2, axis="y")
ax.set_xticks(x)

plt.suptitle("Per-Place Threshold Analysis: GardensPoint day_left (19 Discovered Places)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig15_per_place_thresholds.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
plt.close()
print(f"Saved: {path}")
