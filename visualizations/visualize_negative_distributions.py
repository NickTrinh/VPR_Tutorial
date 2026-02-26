"""
Visualize negative Gaussian mixture distributions per place.

Reads image_averages.csv and place_averages.csv to reconstruct
the per-place Gaussian mixture M(p) = Σ w_i N(μ_i, σ²_i) from
the paper (Eq. 2), showing individual components and the threshold.
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize negative Gaussian mixture distributions per place"
    )
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to results/<Dataset>/<descriptor>/ containing CSVs")
    parser.add_argument("--num_places", type=int, default=4,
                        help="Number of example places to show (default: 4)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as results_dir)")
    parser.add_argument("--selection", type=str, default="spread",
                        choices=["spread", "first", "extreme"],
                        help="How to select example places: "
                             "spread=evenly spaced, first=first N, "
                             "extreme=min/max threshold contrast (default: spread)")
    return parser.parse_args()


def load_image_averages(results_dir):
    """Load per-image statistics from image_averages.csv.

    Returns dict: place_id -> list of (mu, sigma, filter_n)
    """
    path = os.path.join(results_dir, "image_averages.csv")
    place_images = defaultdict(list)

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            place = row["Image"].split("/")[0]
            mu = float(row["Mean Bad Scores"])
            sigma = float(row["Std Deviation Bad Scores"])
            filter_n = float(row["Filter N"])
            place_images[place].append((mu, sigma, filter_n))

    return dict(place_images)


def load_place_averages(results_dir):
    """Load per-place thresholds from place_averages.csv.

    Returns dict: place_id -> {simple_avg_threshold, weighted_avg_threshold, ...}
    """
    path = os.path.join(results_dir, "place_averages.csv")
    place_thresholds = {}

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            place_thresholds[row["place"]] = {
                "simple_avg": float(row["simple_avg_threshold"]),
                "weighted_avg": float(row["weighted_avg_threshold"]),
                "std_dev": float(row["std_dev_of_thresholds"]),
            }

    return place_thresholds


def select_places(place_images, place_thresholds, num_places, selection):
    """Select which places to visualize."""
    all_places = sorted(place_thresholds.keys(), key=lambda x: int(x[1:]))

    if selection == "first":
        return all_places[:num_places]

    if selection == "extreme":
        # Pick places with lowest and highest thresholds for contrast
        by_threshold = sorted(all_places,
                              key=lambda p: place_thresholds[p]["simple_avg"])
        half = num_places // 2
        selected = by_threshold[:half] + by_threshold[-(num_places - half):]
        return sorted(selected, key=lambda x: int(x[1:]))

    # "spread" — evenly spaced through the place index range
    if len(all_places) <= num_places:
        return all_places
    indices = np.linspace(0, len(all_places) - 1, num_places, dtype=int)
    return [all_places[i] for i in indices]


def gaussian(x, mu, sigma):
    """Evaluate Gaussian PDF."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def plot_place_distributions(place_images, place_thresholds, selected_places,
                             output_dir, dataset_label):
    """Generate the main visualization: one subplot per selected place."""
    n = len(selected_places)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), squeeze=False)

    for idx, place in enumerate(selected_places):
        ax = axes[idx, 0]
        components = place_images[place]
        thresholds = place_thresholds[place]

        # Determine x range from all components
        all_mu = [c[0] for c in components]
        all_sigma = [c[1] for c in components]
        x_min = min(mu - 3 * sig for mu, sig in zip(all_mu, all_sigma))
        x_max = max(mu + 3 * sig for mu, sig in zip(all_mu, all_sigma))
        x_min = max(x_min, -0.05)
        x = np.linspace(x_min, x_max, 500)

        # Compute weights: w_i = n_images_for_component / total
        # In our case each image contributes one component, equal weight
        n_components = len(components)
        w = 1.0 / n_components

        # Plot individual components
        mixture = np.zeros_like(x)
        for j, (mu, sigma, _) in enumerate(components):
            y = gaussian(x, mu, sigma)
            ax.plot(x, w * y, color="steelblue", alpha=0.3, linewidth=0.8)
            mixture += w * y

        # Plot mixture
        ax.plot(x, mixture, color="steelblue", linewidth=2.0, label="Mixture $M(p)$")
        ax.fill_between(x, mixture, alpha=0.1, color="steelblue")

        # Plot thresholds
        theta = thresholds["simple_avg"]
        ax.axvline(x=theta, color="red", linestyle="--", linewidth=1.5,
                   label=f"$\\theta_p$ = {theta:.4f}")

        # Annotate
        ax.set_title(f"Place {place[1:]}  "
                     f"({n_components} images, "
                     f"$\\theta_p$={theta:.4f}, "
                     f"$\\sigma_{{\\theta}}$={thresholds['std_dev']:.4f})",
                     fontsize=11)
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.2)

    axes[-1, 0].set_xlabel("Cosine Similarity (negative match scores)")
    fig.suptitle(f"Negative Gaussian Mixture Distributions — {dataset_label}",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = os.path.join(output_dir, "negative_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_threshold_histogram(place_thresholds, output_dir, dataset_label):
    """Histogram of per-place thresholds to show variation across places."""
    places = sorted(place_thresholds.keys(), key=lambda x: int(x[1:]))
    simple_thresholds = [place_thresholds[p]["simple_avg"] for p in places]
    weighted_thresholds = [place_thresholds[p]["weighted_avg"] for p in places]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: histogram of thresholds
    bins = min(40, max(10, len(places) // 5))
    ax1.hist(simple_thresholds, bins=bins, alpha=0.7, color="steelblue",
             edgecolor="white", label="Simple Avg")
    ax1.hist(weighted_thresholds, bins=bins, alpha=0.5, color="darkorange",
             edgecolor="white", label="Weighted Avg")
    ax1.set_xlabel("Threshold $\\theta_p$")
    ax1.set_ylabel("Number of Places")
    ax1.set_title("Distribution of Per-Place Thresholds")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # Right: threshold vs place index (shows spatial variation)
    place_indices = [int(p[1:]) for p in places]
    ax2.scatter(place_indices, simple_thresholds, s=8, alpha=0.6,
                color="steelblue", label="Simple Avg")
    ax2.scatter(place_indices, weighted_thresholds, s=8, alpha=0.6,
                color="darkorange", label="Weighted Avg")

    # Global mean line
    global_mean = np.mean(simple_thresholds)
    ax2.axhline(y=global_mean, color="red", linestyle="--", linewidth=1,
                label=f"Global mean = {global_mean:.4f}")

    ax2.set_xlabel("Place Index")
    ax2.set_ylabel("Threshold $\\theta_p$")
    ax2.set_title("Per-Place Threshold Variation")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    fig.suptitle(f"Threshold Analysis — {dataset_label}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "threshold_variation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def print_summary(place_images, place_thresholds, selected_places):
    """Print summary statistics."""
    all_thresholds = [place_thresholds[p]["simple_avg"]
                      for p in place_thresholds]
    print(f"\n{'=' * 50}")
    print(f"  Total places:       {len(place_thresholds)}")
    print(f"  Threshold range:    [{min(all_thresholds):.4f}, {max(all_thresholds):.4f}]")
    print(f"  Threshold mean:     {np.mean(all_thresholds):.4f}")
    print(f"  Threshold std:      {np.std(all_thresholds):.4f}")
    print(f"  Selected places:    {', '.join(selected_places)}")
    print(f"{'=' * 50}")


def main():
    args = parse_args()
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Derive a label from the path (e.g. "Nordland_Mini_g3s3 / eigenplaces")
    parts = os.path.normpath(args.results_dir).split(os.sep)
    dataset_label = " / ".join(parts[-2:]) if len(parts) >= 2 else parts[-1]

    print(f"Loading data from: {args.results_dir}")
    place_images = load_image_averages(args.results_dir)
    place_thresholds = load_place_averages(args.results_dir)

    selected = select_places(place_images, place_thresholds,
                             args.num_places, args.selection)

    print_summary(place_images, place_thresholds, selected)

    output_files = []
    output_files.append(
        plot_place_distributions(place_images, place_thresholds, selected,
                                 output_dir, dataset_label)
    )
    output_files.append(
        plot_threshold_histogram(place_thresholds, output_dir, dataset_label)
    )

    print(f"\nDone. {len(output_files)} figures saved to {output_dir}")


if __name__ == "__main__":
    main()
