"""
Visualize positive vs negative similarity score distributions per place.

Shows the cosine similarity scores for true matches (same place) vs
non-matches (different places), with the computed threshold as a
vertical line. This illustrates mechanistically why the threshold works.

Usage (from project root):
    python visualizations/visualize_score_separation.py \
        --dataset gardens_point_mini --descriptor eigenplaces \
        --num_places 4 --selection extreme
"""

import argparse
import csv
import os
import pickle
import sys

# Allow imports from project root when running from visualizations/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from config import get_dataset_config, auto_detect_dataset_structure
from data_utils import DatasetLoader
from utils import normalize_l2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize positive vs negative similarity score distributions"
    )
    parser.add_argument("--dataset", type=str, default="gardens_point_mini",
                        help="Dataset config key (default: gardens_point_mini)")
    parser.add_argument("--descriptor", type=str, default="eigenplaces",
                        help="Descriptor name (default: eigenplaces)")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Path to results/<Dataset>/<descriptor>/ for thresholds "
                             "(default: auto-detected from dataset/descriptor)")
    parser.add_argument("--num_places", type=int, default=4,
                        help="Number of example places to show (default: 4)")
    parser.add_argument("--selection", type=str, default="extreme",
                        choices=["spread", "first", "extreme"],
                        help="How to select example places (default: extreme)")
    parser.add_argument("--max_total_images", type=int, default=3000,
                        help="Max images to load for large datasets (default: 3000)")
    parser.add_argument("--output_dir", type=str, default="results/visualizations",
                        help="Output directory for PNGs")
    return parser.parse_args()


def load_thresholds(results_dir):
    """Load per-place thresholds from place_averages.csv."""
    path = os.path.join(results_dir, "place_averages.csv")
    thresholds = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            thresholds[row["place"]] = float(row["simple_avg_threshold"])
    return thresholds


def load_cached_descriptors(cache_dir, required_indices):
    """Load descriptors from img_{idx}_descriptor.pkl cache files."""
    descriptors = {}
    missing = []
    for idx in required_indices:
        path = os.path.join(cache_dir, f"img_{idx}_descriptor.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            desc = data["descriptor"]
            # Flatten to 1D if needed
            if desc.ndim > 1:
                desc = desc.reshape(-1)
            descriptors[idx] = desc.astype(np.float32)
        else:
            missing.append(idx)
    if missing:
        print(f"  Warning: {len(missing)} cached descriptors not found, skipping.")
    return descriptors


def build_feature_matrix(place_map, descriptors, max_total_images):
    """Build (num_places, dim) matrix by averaging descriptors per place.

    For large datasets, sample places to stay under max_total_images.
    Returns: feat_matrix (N, D), sampled_place_indices
    """
    total = sum(len(p) for p in place_map)
    if total > max_total_images:
        # Sample places proportionally
        n_sample = max(1, int(max_total_images / max(1, total / len(place_map))))
        n_sample = min(n_sample, len(place_map))
        rng = np.random.default_rng(42)
        place_indices = sorted(rng.choice(len(place_map), size=n_sample, replace=False))
        print(f"  Large dataset: sampling {n_sample} of {len(place_map)} places")
    else:
        place_indices = list(range(len(place_map)))

    # Collect per-image descriptors for sampled places
    place_image_feats = []  # list of lists of feature vectors
    valid_place_indices = []

    for pi in place_indices:
        imgs = place_map[pi]
        feats = [descriptors[idx] for idx in imgs if idx in descriptors]
        if feats:
            place_image_feats.append(feats)
            valid_place_indices.append(pi)

    return place_image_feats, valid_place_indices


def compute_similarity_scores(place_image_feats):
    """Compute all pairwise cosine similarities.

    Returns:
        positive_scores: list of arrays (one per place) — same-place similarities
        negative_scores: list of arrays (one per place) — cross-place similarities
    """
    # Stack all features into a matrix for fast dot products
    # Also track which place each feature belongs to
    all_feats = []
    feat_place_labels = []
    for p_idx, feats in enumerate(place_image_feats):
        for f in feats:
            all_feats.append(f)
            feat_place_labels.append(p_idx)

    feat_matrix = np.stack(all_feats, axis=0).astype(np.float32)
    feat_matrix = normalize_l2(feat_matrix)
    feat_place_labels = np.array(feat_place_labels)

    # Full cosine similarity matrix (N x N)
    sim_matrix = feat_matrix @ feat_matrix.T

    n_places = len(place_image_feats)
    positive_scores = []
    negative_scores = []

    for p_idx in range(n_places):
        mask_pos = feat_place_labels == p_idx
        mask_neg = ~mask_pos
        rows = np.where(mask_pos)[0]

        # Positive: similarities between same-place images (exclude self)
        pos = []
        for r in rows:
            for c in rows:
                if r != c:
                    pos.append(sim_matrix[r, c])

        # Negative: similarities from this place's images to all other places
        neg = []
        for r in rows:
            neg.extend(sim_matrix[r, mask_neg].tolist())

        positive_scores.append(np.array(pos, dtype=np.float32))
        negative_scores.append(np.array(neg, dtype=np.float32))

    return positive_scores, negative_scores


def select_places(valid_place_indices, thresholds, num_places, selection):
    """Select which places to visualize."""
    # Only keep places that have thresholds
    keyed = [(pi, f"p{pi}") for pi in valid_place_indices
             if f"p{pi}" in thresholds]

    if selection == "first":
        keyed = keyed[:num_places]
    elif selection == "extreme":
        sorted_by_thresh = sorted(keyed, key=lambda x: thresholds[x[1]])
        half = num_places // 2
        keyed = sorted_by_thresh[:half] + sorted_by_thresh[-(num_places - half):]
        keyed = sorted(keyed, key=lambda x: x[0])
    else:  # spread
        if len(keyed) > num_places:
            indices = np.linspace(0, len(keyed) - 1, num_places, dtype=int)
            keyed = [keyed[i] for i in indices]

    return keyed[:num_places]


def kde_curve(scores, x):
    """Compute KDE curve, return zeros if too few points."""
    if len(scores) < 3 or np.std(scores) < 1e-8:
        return np.zeros_like(x)
    try:
        kde = gaussian_kde(scores, bw_method="scott")
        return kde(x)
    except Exception:
        return np.zeros_like(x)


def plot_score_separation(place_image_feats, valid_place_indices,
                          positive_scores, negative_scores,
                          thresholds, selected, output_dir, dataset_label):
    """One subplot per selected place: positive vs negative score distributions."""
    n = len(selected)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.5 * n), squeeze=False)

    for row, (pi, place_key) in enumerate(selected):
        ax = axes[row, 0]

        # Map valid_place_indices -> index in our arrays
        arr_idx = valid_place_indices.index(pi)
        pos = positive_scores[arr_idx]
        neg = negative_scores[arr_idx]
        theta = thresholds[place_key]

        # X range covering both distributions
        all_scores = np.concatenate([pos, neg]) if len(pos) > 0 else neg
        x_min = max(float(all_scores.min()) - 0.05, -0.1)
        x_max = min(float(all_scores.max()) + 0.05, 1.0)
        x = np.linspace(x_min, x_max, 500)

        # KDE curves
        kde_neg = kde_curve(neg, x)
        kde_pos = kde_curve(pos, x)

        ax.fill_between(x, kde_neg, alpha=0.3, color="#e74c3c", label="Negative (different place)")
        ax.plot(x, kde_neg, color="#e74c3c", linewidth=1.5)

        if len(pos) > 0:
            ax.fill_between(x, kde_pos, alpha=0.3, color="#27ae60", label="Positive (same place)")
            ax.plot(x, kde_pos, color="#27ae60", linewidth=1.5)

        ax.axvline(x=theta, color="#2980b9", linestyle="--", linewidth=2.0,
                   label=f"$\\theta_p$ = {theta:.4f}")

        # Compute overlap area as a difficulty indicator
        if len(pos) > 0 and kde_neg.sum() > 0 and kde_pos.sum() > 0:
            overlap = np.trapz(np.minimum(kde_neg, kde_pos), x)
            ax.set_title(
                f"Place {pi}  —  "
                f"$\\theta_p$={theta:.4f}  |  "
                f"pos: n={len(pos)}, $\\mu$={pos.mean():.3f}  |  "
                f"neg: n={len(neg)}, $\\mu$={neg.mean():.3f}  |  "
                f"overlap={overlap:.3f}",
                fontsize=10
            )
        else:
            ax.set_title(f"Place {pi}  —  $\\theta_p$={theta:.4f}  |  "
                         f"neg: n={len(neg)}, $\\mu$={neg.mean():.3f}", fontsize=10)

        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.2)

    axes[-1, 0].set_xlabel("Cosine Similarity")
    fig.suptitle(f"Score Separation: Positive vs Negative Matches — {dataset_label}",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = os.path.join(output_dir, "score_separation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_separation_summary(place_image_feats, valid_place_indices,
                            positive_scores, negative_scores,
                            thresholds, output_dir, dataset_label):
    """Summary scatter: mean(pos) vs mean(neg) per place, colored by threshold."""
    pos_means, neg_means, theta_vals, place_labels = [], [], [], []

    for arr_idx, pi in enumerate(valid_place_indices):
        key = f"p{pi}"
        if key not in thresholds:
            continue
        pos = positive_scores[arr_idx]
        neg = negative_scores[arr_idx]
        if len(pos) == 0 or len(neg) == 0:
            continue
        pos_means.append(pos.mean())
        neg_means.append(neg.mean())
        theta_vals.append(thresholds[key])
        place_labels.append(pi)

    if not pos_means:
        print("  No valid places for summary scatter, skipping.")
        return None

    pos_means = np.array(pos_means)
    neg_means = np.array(neg_means)
    theta_vals = np.array(theta_vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: scatter of mean_neg vs mean_pos per place
    sc = ax1.scatter(neg_means, pos_means, c=theta_vals, cmap="viridis",
                     s=20, alpha=0.7, edgecolors="none")
    # Diagonal line (pos == neg means no separation)
    lims = [min(neg_means.min(), pos_means.min()) - 0.02,
            max(neg_means.max(), pos_means.max()) + 0.02]
    ax1.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="No separation")
    ax1.set_xlabel("Mean Negative Score")
    ax1.set_ylabel("Mean Positive Score")
    ax1.set_title("Per-Place Score Separation\n(above diagonal = good separation)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    fig.colorbar(sc, ax=ax1, label="Threshold $\\theta_p$")

    # Right: separation gap (mean_pos - mean_neg) distribution
    gaps = pos_means - neg_means
    ax2.hist(gaps, bins=min(40, max(10, len(gaps) // 5)),
             color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axvline(x=0, color="red", linestyle="--", linewidth=1, label="No gap")
    ax2.axvline(x=gaps.mean(), color="green", linestyle="--", linewidth=1.5,
                label=f"Mean gap = {gaps.mean():.3f}")
    ax2.set_xlabel("Separation Gap (mean pos − mean neg)")
    ax2.set_ylabel("Number of Places")
    ax2.set_title("Distribution of Separation Gaps")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    fig.suptitle(f"Score Separation Summary — {dataset_label}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "score_separation_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset config and build place map
    print(f"Loading dataset config: {args.dataset}")
    dataset_config = get_dataset_config(args.dataset)
    if dataset_config.format == "landmark":
        dataset_config = auto_detect_dataset_structure(dataset_config)

    # Build place map without triggering feature extraction
    # DatasetLoader.__init__ builds place_map from filenames only
    print("Building place map (no feature extraction) ...")
    loader = DatasetLoader(dataset_config, use_cache=True,
                           descriptor_name=args.descriptor)
    place_map = loader.place_map
    print(f"  {len(place_map)} places loaded")

    # Auto-detect results dir
    results_dir = args.results_dir or os.path.join(
        "results", dataset_config.name, args.descriptor
    )
    print(f"Loading thresholds from: {results_dir}")
    thresholds = load_thresholds(results_dir)

    # Load cached descriptors
    cache_dir = os.path.join("cache", dataset_config.name, args.descriptor)
    print(f"Loading cached descriptors from: {cache_dir}")
    all_required = sorted(set(idx for place in place_map for idx in place))
    descriptors = load_cached_descriptors(cache_dir, all_required)
    print(f"  Loaded {len(descriptors)} descriptors")

    # Build feature arrays per place (with sampling for large datasets)
    place_image_feats, valid_place_indices = build_feature_matrix(
        place_map, descriptors, args.max_total_images
    )
    print(f"  Using {len(valid_place_indices)} places for similarity computation")

    # Compute positive and negative similarity scores
    print("Computing similarity scores ...")
    positive_scores, negative_scores = compute_similarity_scores(place_image_feats)

    # Select example places to highlight
    selected = select_places(valid_place_indices, thresholds,
                             args.num_places, args.selection)
    print(f"  Selected places: {[k for _, k in selected]}")

    # Dataset label for titles
    parts = [dataset_config.name, args.descriptor]
    dataset_label = " / ".join(parts)

    # Generate figures
    output_files = []
    output_files.append(
        plot_score_separation(place_image_feats, valid_place_indices,
                              positive_scores, negative_scores,
                              thresholds, selected, args.output_dir, dataset_label)
    )
    output_files.append(
        plot_separation_summary(place_image_feats, valid_place_indices,
                                positive_scores, negative_scores,
                                thresholds, args.output_dir, dataset_label)
    )

    print(f"\nDone. {len([f for f in output_files if f])} figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
