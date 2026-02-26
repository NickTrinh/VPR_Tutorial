"""
Visualize cross-validation stability of per-place thresholds across runs.

Loads all per-run CSVs, computes per-place threshold for each run,
and shows how stable the thresholds are across different train/test splits.

Produces:
  1. cv_stability_histogram.png  — distribution of per-place threshold std across all places
  2. cv_stability_boxplots.png   — box plots of threshold distributions for example places
  3. cv_stability_heatmap.png    — heatmap of threshold values (run × place)

Usage (from project root):
    python visualizations/visualize_cv_stability.py \
        --results_dir results/Nordland_Mini_g3s3/eigenplaces \
        --num_places 20 --output_dir results/visualizations
"""

import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize cross-validation stability of per-place thresholds"
    )
    parser.add_argument("--results_dir", type=str,
                        default="results/Nordland_Mini_g3s3/eigenplaces",
                        help="Path to results/<Dataset>/<descriptor>/")
    parser.add_argument("--num_places", type=int, default=20,
                        help="Number of example places for box plots (default: 20)")
    parser.add_argument("--output_dir", type=str, default="results/visualizations",
                        help="Output directory for PNGs")
    return parser.parse_args()


def load_all_runs(results_dir):
    """Load all test_results_run_N.csv files.

    Returns:
        run_place_thresholds: dict {run_number: {place_key: mean_bad_score_mean}}
        all_places: sorted list of place keys
    """
    pattern = os.path.join(results_dir, "test_results_run_*.csv")
    run_files = sorted(glob.glob(pattern),
                       key=lambda p: int(re.search(r"run_(\d+)", p).group(1)))

    if not run_files:
        raise FileNotFoundError(f"No run CSV files found in {results_dir}")

    print(f"  Found {len(run_files)} run files")

    run_place_thresholds = {}
    all_places_set = set()

    for fpath in run_files:
        run_num = int(re.search(r"run_(\d+)", fpath).group(1))
        place_scores = {}  # place -> list of mean_bad_scores for images in that place

        with open(fpath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                place = row["Image"].split("/")[0]
                score = float(row["Mean Bad Scores"])
                place_scores.setdefault(place, []).append(score)
                all_places_set.add(place)

        # Per-place threshold = mean of image thresholds
        run_place_thresholds[run_num] = {
            p: float(np.mean(scores)) for p, scores in place_scores.items()
        }

    all_places = sorted(all_places_set, key=lambda x: int(x[1:]))
    return run_place_thresholds, all_places


def build_threshold_matrix(run_place_thresholds, all_places):
    """Build (num_runs, num_places) matrix of threshold values.

    Missing values (place not present in a run) filled with NaN.
    """
    runs = sorted(run_place_thresholds.keys())
    matrix = np.full((len(runs), len(all_places)), np.nan, dtype=np.float32)

    place_to_idx = {p: i for i, p in enumerate(all_places)}
    for r_idx, run in enumerate(runs):
        for place, val in run_place_thresholds[run].items():
            if place in place_to_idx:
                matrix[r_idx, place_to_idx[place]] = val

    return matrix, runs


def plot_stability_histogram(matrix, all_places, output_dir, dataset_label):
    """Histogram of per-place threshold standard deviation across runs."""
    # Std across runs for each place (ignore NaN)
    place_stds = np.nanstd(matrix, axis=0)
    place_means = np.nanmean(matrix, axis=0)
    cv = place_stds / (place_means + 1e-9)  # coefficient of variation

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: histogram of std devs
    ax1.hist(place_stds, bins=min(60, max(15, len(place_stds) // 20)),
             color="steelblue", edgecolor="white", alpha=0.85)
    ax1.axvline(x=place_stds.mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"mean $\\sigma$ = {place_stds.mean():.4f}")
    ax1.axvline(x=np.percentile(place_stds, 95), color="orange",
                linestyle="--", linewidth=1.2,
                label=f"95th pct = {np.percentile(place_stds, 95):.4f}")
    ax1.set_xlabel("Threshold Std Dev across Runs $\\sigma_{\\theta_p}$")
    ax1.set_ylabel("Number of Places")
    ax1.set_title("Per-Place Threshold Stability")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Right: histogram of coefficient of variation (std/mean)
    ax2.hist(cv * 100, bins=min(60, max(15, len(cv) // 20)),
             color="darkorange", edgecolor="white", alpha=0.85)
    ax2.axvline(x=(cv * 100).mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"mean CV = {cv.mean() * 100:.2f}%")
    ax2.set_xlabel("Coefficient of Variation (%)")
    ax2.set_ylabel("Number of Places")
    ax2.set_title("Relative Threshold Variability (CV = $\\sigma$ / $\\mu$)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    fig.suptitle(f"Cross-Validation Threshold Stability — {dataset_label}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "cv_stability_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_stability_boxplots(matrix, all_places, num_places, output_dir, dataset_label):
    """Box plots of threshold distributions for evenly sampled example places."""
    # Select evenly spaced places
    n = min(num_places, len(all_places))
    indices = np.linspace(0, len(all_places) - 1, n, dtype=int)
    selected_indices = list(indices)
    selected_places = [all_places[i] for i in selected_indices]
    selected_data = [matrix[:, i] for i in selected_indices]
    # Remove NaN per column
    selected_data = [col[~np.isnan(col)] for col in selected_data]

    fig, ax = plt.subplots(figsize=(max(12, n * 0.65), 5))

    bp = ax.boxplot(selected_data, patch_artist=True, notch=False,
                    medianprops=dict(color="red", linewidth=1.5),
                    boxprops=dict(facecolor="steelblue", alpha=0.6),
                    whiskerprops=dict(linewidth=0.8),
                    flierprops=dict(marker=".", markersize=2, alpha=0.4))

    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels([p[1:] for p in selected_places],
                       fontsize=max(6, 9 - n // 10), rotation=45)
    ax.set_xlabel("Place Index")
    ax.set_ylabel("Threshold $\\theta_p$")
    ax.set_title(
        f"Per-Place Threshold Distribution across {matrix.shape[0]} Runs — {dataset_label}",
        fontsize=12, fontweight="bold"
    )
    ax.grid(True, axis="y", alpha=0.2)

    # Annotate mean std
    overall_std = np.nanstd(matrix, axis=0).mean()
    ax.text(0.02, 0.97, f"Mean $\\sigma$ across all places = {overall_std:.4f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, "cv_stability_boxplots.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_stability_heatmap(matrix, all_places, runs, output_dir, dataset_label):
    """Heatmap: run (y) x place (x), color = threshold value.

    Subsample places for readability if dataset is large.
    """
    max_places_shown = 200
    if len(all_places) > max_places_shown:
        step = len(all_places) // max_places_shown
        col_indices = list(range(0, len(all_places), step))[:max_places_shown]
        sub_matrix = matrix[:, col_indices]
        sub_places = [all_places[i] for i in col_indices]
        note = f" (showing {len(col_indices)} of {len(all_places)} places)"
    else:
        sub_matrix = matrix
        sub_places = all_places
        note = ""

    fig, ax = plt.subplots(figsize=(14, max(4, len(runs) * 0.12)))

    im = ax.imshow(sub_matrix, aspect="auto", cmap="viridis",
                   interpolation="nearest")

    # Y axis: run numbers (subsample if many)
    max_yticks = 20
    if len(runs) > max_yticks:
        y_step = len(runs) // max_yticks
        ytick_pos = list(range(0, len(runs), y_step))
    else:
        ytick_pos = list(range(len(runs)))
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels([runs[i] for i in ytick_pos], fontsize=7)

    # X axis: place indices (subsample)
    max_xticks = 20
    if len(sub_places) > max_xticks:
        x_step = len(sub_places) // max_xticks
        xtick_pos = list(range(0, len(sub_places), x_step))
    else:
        xtick_pos = list(range(len(sub_places)))
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels([sub_places[i][1:] for i in xtick_pos], fontsize=7)

    ax.set_xlabel(f"Place Index{note}")
    ax.set_ylabel("Run Number")
    ax.set_title(f"Threshold Values per Run × Place — {dataset_label}",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Threshold $\\theta_p$", shrink=0.8)

    plt.tight_layout()
    path = os.path.join(output_dir, "cv_stability_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def print_summary(matrix, all_places, runs):
    place_stds = np.nanstd(matrix, axis=0)
    place_means = np.nanmean(matrix, axis=0)
    cv = place_stds / (place_means + 1e-9)
    print(f"\n{'=' * 55}")
    print(f"  Runs:              {len(runs)}")
    print(f"  Places:            {len(all_places)}")
    print(f"  Mean threshold:    {place_means.mean():.4f}")
    print(f"  Mean σ (runs):     {place_stds.mean():.4f}")
    print(f"  Mean CV:           {cv.mean() * 100:.2f}%")
    print(f"  Max σ:             {place_stds.max():.4f}  (place {all_places[place_stds.argmax()]})")
    print(f"  Min σ:             {place_stds.min():.4f}  (place {all_places[place_stds.argmin()]})")
    print(f"{'=' * 55}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    parts = os.path.normpath(args.results_dir).split(os.sep)
    dataset_label = " / ".join(parts[-2:]) if len(parts) >= 2 else parts[-1]

    print(f"Loading run data from: {args.results_dir}")
    run_place_thresholds, all_places = load_all_runs(args.results_dir)

    print(f"Building threshold matrix ({len(run_place_thresholds)} runs × {len(all_places)} places) ...")
    matrix, runs = build_threshold_matrix(run_place_thresholds, all_places)

    print_summary(matrix, all_places, runs)

    output_files = []
    output_files.append(plot_stability_histogram(matrix, all_places,
                                                  args.output_dir, dataset_label))
    output_files.append(plot_stability_boxplots(matrix, all_places, args.num_places,
                                                 args.output_dir, dataset_label))
    output_files.append(plot_stability_heatmap(matrix, all_places, runs,
                                                args.output_dir, dataset_label))

    print(f"\nDone. {len(output_files)} figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
