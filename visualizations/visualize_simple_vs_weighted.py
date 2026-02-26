"""
Visualize Simple Average vs Weighted Average thresholds per place.

Shows that the two threshold methods produce nearly identical values,
supporting the paper's claim that variance weighting adds no practical
benefit over the simpler unweighted mean.

Usage (from project root):
    python visualizations/visualize_simple_vs_weighted.py \
        --results_dirs results/GardensPoint_Mini/eigenplaces \
                       results/Nordland_Mini_g3s3/eigenplaces \
        --output_dir results/visualizations
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scatter plot: simple vs weighted average thresholds"
    )
    parser.add_argument("--results_dirs", type=str, nargs="+",
                        default=[
                            "results/GardensPoint_Mini/eigenplaces",
                            "results/Nordland_Mini_g3s3/eigenplaces",
                            "results/Nordland_Mini_g3s3/cosplace",
                            "results/Nordland_Mini_g3s3/alexnet",
                        ],
                        help="One or more results/<Dataset>/<descriptor>/ paths")
    parser.add_argument("--output_dir", type=str, default="results/visualizations",
                        help="Output directory for PNGs")
    return parser.parse_args()


def load_place_averages(results_dir):
    """Load place_averages.csv -> arrays of (simple, weighted, std_dev) per place."""
    path = os.path.join(results_dir, "place_averages.csv")
    simple, weighted, std_devs = [], [], []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            simple.append(float(row["simple_avg_threshold"]))
            weighted.append(float(row["weighted_avg_threshold"]))
            std_devs.append(float(row["std_dev_of_thresholds"]))
    return np.array(simple), np.array(weighted), np.array(std_devs)


def label_from_path(path):
    """Derive a short label from results path, e.g. 'GardensPoint / eigenplaces'."""
    parts = os.path.normpath(path).split(os.sep)
    return " / ".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def plot_scatter_grid(datasets, output_dir):
    """One subplot per dataset: scatter of simple vs weighted thresholds."""
    n = len(datasets)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for i, (label, simple, weighted, std_devs) in enumerate(datasets):
        ax = axes[i // cols][i % cols]

        # Scatter, colored by std_dev (uncertainty in threshold estimate)
        sc = ax.scatter(simple, weighted, c=std_devs, cmap="YlOrRd",
                        s=12, alpha=0.7, edgecolors="none")
        fig.colorbar(sc, ax=ax, label="$\\sigma_{\\theta}$ (threshold std)")

        # Identity line
        lo = min(simple.min(), weighted.min()) - 0.002
        hi = max(simple.max(), weighted.max()) + 0.002
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, alpha=0.6,
                label="y = x (identical)")

        # Stats
        r, _ = pearsonr(simple, weighted)
        diff = weighted - simple
        mae = np.abs(diff).mean()
        max_diff = np.abs(diff).max()

        ax.set_xlabel("Simple Average $\\theta_p$")
        ax.set_ylabel("Weighted Average $\\theta_p$")
        ax.set_title(
            f"{label}\n"
            f"r={r:.6f}  |  MAE={mae:.6f}  |  max|diff|={max_diff:.6f}",
            fontsize=10
        )
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    # Hide unused subplots
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle("Simple vs Weighted Average Thresholds per Place",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "simple_vs_weighted_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_difference_distribution(datasets, output_dir):
    """Distribution of (weighted - simple) differences across all datasets."""
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(5 * len(datasets), 4), squeeze=False)

    for i, (label, simple, weighted, _) in enumerate(datasets):
        ax = axes[0][i]
        diff = weighted - simple

        ax.hist(diff, bins=min(50, max(10, len(diff) // 10)),
                color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1.2,
                   label="zero difference")
        ax.axvline(x=diff.mean(), color="green", linestyle="--", linewidth=1.2,
                   label=f"mean = {diff.mean():.2e}")

        ax.set_xlabel("Weighted $-$ Simple")
        ax.set_ylabel("Number of Places" if i == 0 else "")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Distribution of (Weighted − Simple) Threshold Differences",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "simple_vs_weighted_diff.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def print_summary(datasets):
    print(f"\n{'=' * 65}")
    print(f"{'Dataset':<40} {'r':>8} {'MAE':>10} {'max|diff|':>12}")
    print(f"{'-' * 65}")
    for label, simple, weighted, _ in datasets:
        r, _ = pearsonr(simple, weighted)
        diff = weighted - simple
        print(f"{label:<40} {r:>8.6f} {np.abs(diff).mean():>10.6f} "
              f"{np.abs(diff).max():>12.6f}")
    print(f"{'=' * 65}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    datasets = []
    for rdir in args.results_dirs:
        if not os.path.exists(os.path.join(rdir, "place_averages.csv")):
            print(f"  Skipping {rdir} — place_averages.csv not found")
            continue
        simple, weighted, std_devs = load_place_averages(rdir)
        label = label_from_path(rdir)
        datasets.append((label, simple, weighted, std_devs))
        print(f"  Loaded {len(simple)} places from {label}")

    if not datasets:
        print("No valid results directories found.")
        return

    print_summary(datasets)

    output_files = []
    output_files.append(plot_scatter_grid(datasets, args.output_dir))
    output_files.append(plot_difference_distribution(datasets, args.output_dir))

    print(f"\nDone. {len(output_files)} figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
