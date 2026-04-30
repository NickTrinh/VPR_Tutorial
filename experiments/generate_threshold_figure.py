"""
Per-place similarity histograms with threshold visualization.

For one dataset (default GardensPoint) we run the online discovery, then for
three representative places — low / mid / high separability — plot:
  - within-place positive-similarity histogram (green)
  - cross-place negative-similarity histogram (red)
  - vertical line at θ_k = μ_k⁻ + k · σ_k⁻
  - annotation of sep_k and adaptive k

Visualises why per-place k matters: low-sep places get k≈1, high-sep get k=2.

Usage:
    python -m experiments.generate_threshold_figure
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.final_all_datasets import load_descs_pkl, discover_places  # noqa: E402

CACHE_DIR = "cache/GardensPoint/day_left/dinov2-salad"
OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "IEEE_RAL_VPR",
    "fig_threshold_mechanism.png",
)


def per_place_stats(ref_S, places, pidx):
    pos, neg = [], []
    for i in places[pidx]:
        for j in places[pidx]:
            if i != j:
                pos.append(ref_S[i, j])
        for op, other in enumerate(places):
            if op == pidx:
                continue
            for j in other:
                neg.append(ref_S[i, j])
    pos = np.array(pos) if pos else np.array([0.5])
    neg = np.array(neg)
    mu_pos, mu_neg, sigma_neg = pos.mean(), neg.mean(), neg.std()
    sep = (mu_pos - mu_neg) / max(sigma_neg, 1e-8)
    k = max(1.0, min(sep / 2.0, 2.0))
    theta = mu_neg + k * sigma_neg
    return pos, neg, mu_pos, mu_neg, sigma_neg, sep, k, theta


def main():
    print(f"Loading descriptors from {CACHE_DIR}...")
    ref = load_descs_pkl(CACHE_DIR)
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    print(f"  {len(ref)} descriptors")

    places = discover_places(ref, min_place_size=2)
    print(f"Discovered {len(places)} places (sizes: {min(len(p) for p in places)}-{max(len(p) for p in places)})")

    ref_S = ref @ ref.T

    # Compute sep_k for every place; pick low / mid / high
    seps = []
    for pidx in range(len(places)):
        _, _, _, _, _, sep, _, _ = per_place_stats(ref_S, places, pidx)
        seps.append(sep)
    seps = np.array(seps)
    order = np.argsort(seps)
    low, mid, high = order[1], order[len(order) // 2], order[-2]
    chosen = [(low, "low sep"), (mid, "median sep"), (high, "high sep")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=160, sharey=True)
    for ax, (pidx, label) in zip(axes, chosen):
        pos, neg, mu_pos, mu_neg, sigma_neg, sep, k, theta = per_place_stats(ref_S, places, pidx)
        ax.hist(neg, bins=40, density=True, alpha=0.55, color="#D9534F",
                label=f"cross-place ($\\mu^- = {mu_neg:.2f}$, $\\sigma^- = {sigma_neg:.2f}$)")
        ax.hist(pos, bins=20, density=True, alpha=0.65, color="#5CB85C",
                label=f"within-place ($\\mu^+ = {mu_pos:.2f}$, $n = {len(pos)}$)")
        ax.axvline(theta, color="black", linestyle="--", linewidth=2.0,
                   label=f"$\\theta_k = {theta:.2f}$")
        ax.set_title(f"Place {pidx}, |$P_k$|={len(places[pidx])} ({label})\n"
                     f"sep$_k$ = {sep:.2f}, $k$ = {k:.2f}",
                     fontsize=12)
        ax.set_xlabel("cosine similarity", fontsize=11)
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
        ax.set_xlim(0.0, 1.0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("density", fontsize=11)

    fig.suptitle("Per-place threshold sits between the negative and positive distributions; "
                 "$k$ adapts to local separability",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
