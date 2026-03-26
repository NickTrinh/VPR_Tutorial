"""
Sensitivity analysis: How does the filter_n cap affect performance?

Sweeps filter_n_cap from 1 to 15 and measures closed-set and open-set F1.
Shows that the cap=10 choice is robust (performance plateaus).

Usage:
    python experiments/filter_n_sensitivity.py
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
    load_cached_descriptors, discover_places, evaluate,
)

OUTPUT_DIR = "results/paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_thresholds_with_cap(ref_descs, places, cap):
    thresholds = {}
    filter_ns = {}
    for p_idx in range(len(places)):
        target = places[p_idx]
        other = [f for i, p in enumerate(places) if i != p_idx for f in p]
        if not other or len(target) < 2:
            thresholds[p_idx] = 0.0
            filter_ns[p_idx] = 0
            continue
        neg_sims = ref_descs[target] @ ref_descs[other].T
        per_img = neg_sims.mean(axis=1)
        mean_bad = float(per_img.mean())
        std_bad = float(per_img.std()) if len(per_img) > 1 else 0.1
        pos_sims = ref_descs[target] @ ref_descs[target].T
        np.fill_diagonal(pos_sims, 0)
        mean_good = float(pos_sims.sum(axis=1).mean() / max(len(target) - 1, 1))
        raw_fn = (mean_good - mean_bad) / max(std_bad, 1e-8)
        fn = max(0, min(np.floor(raw_fn), cap))
        thresholds[p_idx] = mean_bad + fn * std_bad
        filter_ns[p_idx] = fn
    return thresholds, filter_ns




# ─── Main ─────────────────────────────────────────────────────────────────────

print("Loading data...")
ref_descs = load_cached_descriptors("cache/GardensPoint/day_left/eigenplaces", 200)
query_descs = load_cached_descriptors("cache/GardensPoint/day_right/eigenplaces", 200)
sfu_descs = load_cached_descriptors("cache/SFU/dry/eigenplaces",
    len(glob("cache/SFU/dry/eigenplaces/img_*_descriptor.pkl")))

places = discover_places(ref_descs)
print(f"  {len(places)} places discovered")

caps = list(range(0, 16))
closed_f1, closed_p, closed_r = [], [], []
open_f1, open_rej = [], []

n_genuine = 100
mixed = np.vstack([query_descs[:n_genuine], sfu_descs[:100]])

for cap in caps:
    thresh, fns = compute_thresholds_with_cap(ref_descs, places, cap)

    # Closed-set
    r = evaluate(query_descs, ref_descs, places, thresh, 200)
    closed_f1.append(r["F1"])
    closed_p.append(r["P"])
    closed_r.append(r["R"])

    # Open-set
    r = evaluate(mixed, ref_descs, places, thresh, n_genuine)
    open_f1.append(r["F1"])
    open_rej.append(r["dist_rej"])

    fn_vals = list(fns.values())
    mean_fn = np.mean(fn_vals)
    print(f"  cap={cap:>2}: closed F1={closed_f1[-1]:>5.1f}%  open F1={open_f1[-1]:>5.1f}%  "
          f"rej={open_rej[-1]:>5.1f}%  avg_fn={mean_fn:.1f}")

# ─── Visualization ────────────────────────────────────────────────────────────

print("\nGenerating figure...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: Closed-set P/R/F1
ax1.plot(caps, closed_f1, "D-", color="#e74c3c", linewidth=2, markersize=6, label="F1")
ax1.plot(caps, closed_p, "s--", color="#3498db", linewidth=1.5, markersize=5, label="Precision")
ax1.plot(caps, closed_r, "o--", color="#27ae60", linewidth=1.5, markersize=5, label="Recall")
ax1.axvline(x=10, color="#7f8c8d", linestyle=":", alpha=0.5, label="cap=10 (used)")
ax1.set_xlabel("filter_n cap", fontsize=11)
ax1.set_ylabel("Score (%)", fontsize=11)
ax1.set_title("Closed-Set: P/R/F1 vs filter_n Cap", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(60, 105)

# Right: Open-set F1 + rejection
ax2r = ax2.twinx()
ax2.plot(caps, open_f1, "D-", color="#e74c3c", linewidth=2, markersize=6, label="F1")
ax2r.plot(caps, open_rej, "^--", color="#9b59b6", linewidth=1.5, markersize=5, label="Dist. Rejection")
ax2.axvline(x=10, color="#7f8c8d", linestyle=":", alpha=0.5)
ax2.set_xlabel("filter_n cap", fontsize=11)
ax2.set_ylabel("F1 Score (%)", fontsize=11, color="#e74c3c")
ax2r.set_ylabel("Distractor Rejection (%)", fontsize=11, color="#9b59b6")
ax2.set_title("Open-Set: F1 & Rejection vs filter_n Cap", fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(60, 105)
ax2r.set_ylim(0, 105)

# Combine legends
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2r.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")

plt.suptitle("Sensitivity: How Does filter_n Cap Affect Performance?",
             fontsize=13, fontweight="bold")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "fig14_filter_n_sensitivity.png")
plt.savefig(path, dpi=200, bbox_inches="tight")
plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
plt.close()
print(f"Saved: {path}")
