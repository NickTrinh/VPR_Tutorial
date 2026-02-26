"""
Overlay positive vs negative score distributions across multiple places.

Left panel: single place in full detail (histogram + KDE + threshold).
Right panel: N places overlaid as KDE curves showing spread of distributions.

Usage (from project root):
    python visualizations/visualize_score_distributions_overlay.py \
        --dataset gardens_point_mini --descriptor eigenplaces \
        --num_places 10 --output_dir results/visualizations
"""

import argparse
import csv
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from config import get_dataset_config, auto_detect_dataset_structure
from data_utils import DatasetLoader
from utils import normalize_l2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-place overlay of positive vs negative score distributions"
    )
    parser.add_argument("--dataset", type=str, default="gardens_point_mini")
    parser.add_argument("--descriptor", type=str, default="eigenplaces")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--num_places", type=int, default=12,
                        help="Number of places to overlay (default: 12)")
    parser.add_argument("--focus_place", type=int, default=None,
                        help="Place index to show in detail panel (default: median-threshold place)")
    parser.add_argument("--max_total_images", type=int, default=3000)
    parser.add_argument("--output_dir", type=str, default="results/visualizations")
    return parser.parse_args()


def load_thresholds(results_dir):
    path = os.path.join(results_dir, "place_averages.csv")
    thresholds = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            thresholds[row["place"]] = float(row["simple_avg_threshold"])
    return thresholds


def load_cached_descriptors(cache_dir, required_indices):
    descriptors = {}
    for idx in required_indices:
        path = os.path.join(cache_dir, f"img_{idx}_descriptor.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            desc = data["descriptor"]
            if desc.ndim > 1:
                desc = desc.reshape(-1)
            descriptors[idx] = desc.astype(np.float32)
    return descriptors


def compute_all_scores(place_map, descriptors, max_total_images):
    """Compute positive and negative similarity scores for sampled places."""
    total = sum(len(p) for p in place_map)
    if total > max_total_images:
        n_sample = max(1, int(max_total_images / max(1, total / len(place_map))))
        rng = np.random.default_rng(42)
        place_indices = sorted(rng.choice(len(place_map), size=n_sample, replace=False))
    else:
        place_indices = list(range(len(place_map)))

    # Build per-place feature lists
    place_feats = []
    valid_indices = []
    for pi in place_indices:
        feats = [descriptors[idx] for idx in place_map[pi] if idx in descriptors]
        if feats:
            place_feats.append(np.stack(feats, axis=0).astype(np.float32))
            valid_indices.append(pi)

    # Stack everything and normalize
    all_feats = np.concatenate(place_feats, axis=0)
    all_feats = normalize_l2(all_feats)
    sim_matrix = all_feats @ all_feats.T

    # Build place labels for each row
    labels = []
    for p_idx, feats in enumerate(place_feats):
        labels.extend([p_idx] * len(feats))
    labels = np.array(labels)

    positive_scores, negative_scores = [], []
    for p_idx in range(len(place_feats)):
        mask = labels == p_idx
        rows = np.where(mask)[0]
        pos = [sim_matrix[r, c] for r in rows for c in rows if r != c]
        neg = sim_matrix[rows][:, ~mask].flatten().tolist()
        positive_scores.append(np.array(pos, dtype=np.float32))
        negative_scores.append(np.array(neg, dtype=np.float32))

    return place_feats, valid_indices, positive_scores, negative_scores


def kde_curve(scores, x):
    if len(scores) < 3 or np.std(scores) < 1e-8:
        return np.zeros_like(x)
    try:
        return gaussian_kde(scores, bw_method="scott")(x)
    except Exception:
        return np.zeros_like(x)


def plot_combined(valid_indices, positive_scores, negative_scores,
                  thresholds, num_places, focus_idx, output_dir, dataset_label):
    """
    Left: single place in detail (histogram + KDE).
    Right: N places overlaid as KDE curves.
    """
    # Select N evenly-spaced places that have thresholds
    keyed = [(pi, f"p{pi}") for pi in valid_indices if f"p{pi}" in thresholds]
    if len(keyed) > num_places:
        step_indices = np.linspace(0, len(keyed) - 1, num_places, dtype=int)
        keyed = [keyed[i] for i in step_indices]

    # Pick focus place: the one closest to median threshold
    if focus_idx is not None:
        focus_entry = next(((pi, k) for pi, k in keyed if pi == focus_idx), keyed[len(keyed)//2])
    else:
        median_thresh = np.median([thresholds[k] for _, k in keyed])
        focus_entry = min(keyed, key=lambda x: abs(thresholds[x[1]] - median_thresh))

    focus_pi, focus_key = focus_entry
    arr_idx_focus = valid_indices.index(focus_pi)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))

    # ── LEFT: single place detail ───────────────────────────────────────────
    pos = positive_scores[arr_idx_focus]
    neg = negative_scores[arr_idx_focus]
    theta = thresholds[focus_key]

    all_s = np.concatenate([pos, neg]) if len(pos) else neg
    x = np.linspace(max(all_s.min() - 0.05, -0.1), min(all_s.max() + 0.05, 1.0), 500)

    # Histogram (normalized)
    ax_left.hist(neg, bins=40, density=True, alpha=0.25, color="#e74c3c",
                 label="Negative (different place)")
    if len(pos) > 0:
        ax_left.hist(pos, bins=20, density=True, alpha=0.25, color="#27ae60",
                     label="Positive (same place)")

    # KDE
    ax_left.plot(x, kde_curve(neg, x), color="#e74c3c", linewidth=2.0)
    if len(pos) > 0:
        ax_left.plot(x, kde_curve(pos, x), color="#27ae60", linewidth=2.0)

    ax_left.axvline(x=theta, color="#2980b9", linestyle="--", linewidth=2.0,
                    label=f"$\\theta_p$ = {theta:.4f}")

    if len(pos) > 0:
        gap = pos.mean() - neg.mean()
        ax_left.set_title(
            f"Place {focus_pi} — Detail View\n"
            f"pos $\\mu$={pos.mean():.3f}  |  neg $\\mu$={neg.mean():.3f}  |  "
            f"gap={gap:.3f}  |  $\\theta_p$={theta:.4f}",
            fontsize=11
        )
    else:
        ax_left.set_title(f"Place {focus_pi} — Detail View\n"
                          f"neg $\\mu$={neg.mean():.3f}  |  $\\theta_p$={theta:.4f}",
                          fontsize=11)

    ax_left.set_xlabel("Cosine Similarity")
    ax_left.set_ylabel("Density")
    ax_left.legend(fontsize=9)
    ax_left.grid(True, alpha=0.2)

    # ── RIGHT: multi-place overlay ───────────────────────────────────────────
    # Color gradient: low threshold = cool, high threshold = warm
    thresh_vals = np.array([thresholds[k] for _, k in keyed])
    t_min, t_max = thresh_vals.min(), thresh_vals.max()

    # Shared x range across all places
    all_neg = np.concatenate([negative_scores[valid_indices.index(pi)]
                               for pi, _ in keyed])
    all_pos_list = [positive_scores[valid_indices.index(pi)] for pi, _ in keyed]
    all_pos = np.concatenate([p for p in all_pos_list if len(p) > 0]) \
        if any(len(p) > 0 for p in all_pos_list) else np.array([])

    x_min = max(float(all_neg.min()) - 0.03, -0.1)
    x_max = 1.0
    if len(all_pos) > 0:
        x_max = min(float(all_pos.max()) + 0.05, 1.0)
    x = np.linspace(x_min, x_max, 500)

    cmap_neg = plt.cm.Reds
    cmap_pos = plt.cm.Greens

    for pi, key in keyed:
        arr_idx = valid_indices.index(pi)
        neg_p = negative_scores[arr_idx]
        pos_p = positive_scores[arr_idx]
        t = thresholds[key]
        norm_t = (t - t_min) / (t_max - t_min + 1e-9)

        alpha = 0.55

        kde_n = kde_curve(neg_p, x)
        if kde_n.sum() > 0:
            ax_right.plot(x, kde_n, color=cmap_neg(0.4 + 0.5 * norm_t),
                          linewidth=1.2, alpha=alpha)

        if len(pos_p) > 0:
            kde_p = kde_curve(pos_p, x)
            if kde_p.sum() > 0:
                ax_right.plot(x, kde_p, color=cmap_pos(0.4 + 0.5 * norm_t),
                              linewidth=1.2, alpha=alpha)

        ax_right.axvline(x=t, color="#2980b9", linewidth=0.6, alpha=0.3)

    # Highlight focus place on right panel too
    arr_idx_f = valid_indices.index(focus_pi)
    kde_n_f = kde_curve(negative_scores[arr_idx_f], x)
    kde_p_f = kde_curve(positive_scores[arr_idx_f], x) \
        if len(positive_scores[arr_idx_f]) > 0 else None

    ax_right.plot(x, kde_n_f, color="#c0392b", linewidth=2.5,
                  label=f"Place {focus_pi} neg (detail)")
    if kde_p_f is not None:
        ax_right.plot(x, kde_p_f, color="#1e8449", linewidth=2.5,
                      label=f"Place {focus_pi} pos (detail)")
    ax_right.axvline(x=thresholds[focus_key], color="#2980b9",
                     linewidth=2.0, linestyle="--",
                     label=f"Place {focus_pi} $\\theta_p$={thresholds[focus_key]:.4f}")

    # Legend entries for the color meaning
    legend_elements = [
        Line2D([0], [0], color=cmap_neg(0.45), linewidth=1.5, label="Negative scores (low $\\theta_p$)"),
        Line2D([0], [0], color=cmap_neg(0.85), linewidth=1.5, label="Negative scores (high $\\theta_p$)"),
        Line2D([0], [0], color=cmap_pos(0.45), linewidth=1.5, label="Positive scores (low $\\theta_p$)"),
        Line2D([0], [0], color=cmap_pos(0.85), linewidth=1.5, label="Positive scores (high $\\theta_p$)"),
        Line2D([0], [0], color="#2980b9", linewidth=0.8, alpha=0.5, label="Thresholds $\\theta_p$"),
    ]
    ax_right.legend(handles=legend_elements, fontsize=8, loc="upper right")

    ax_right.set_xlabel("Cosine Similarity")
    ax_right.set_ylabel("Density")
    ax_right.set_title(
        f"{len(keyed)} Places Overlaid — Score Distributions\n"
        f"Red shades = negative, Green shades = positive  |  "
        f"Color intensity ∝ threshold level",
        fontsize=11
    )
    ax_right.grid(True, alpha=0.2)

    fig.suptitle(f"Positive vs Negative Score Distributions — {dataset_label}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "score_distributions_overlay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_config = get_dataset_config(args.dataset)
    if dataset_config.format == "landmark":
        dataset_config = auto_detect_dataset_structure(dataset_config)

    print("Building place map ...")
    loader = DatasetLoader(dataset_config, use_cache=True,
                           descriptor_name=args.descriptor)
    place_map = loader.place_map

    results_dir = args.results_dir or os.path.join(
        "results", dataset_config.name, args.descriptor)
    thresholds = load_thresholds(results_dir)

    cache_dir = os.path.join("cache", dataset_config.name, args.descriptor)
    all_required = sorted(set(idx for place in place_map for idx in place))
    descriptors = load_cached_descriptors(cache_dir, all_required)
    print(f"  Loaded {len(descriptors)} descriptors")

    place_feats, valid_indices, positive_scores, negative_scores = \
        compute_all_scores(place_map, descriptors, args.max_total_images)

    dataset_label = f"{dataset_config.name} / {args.descriptor}"

    plot_combined(valid_indices, positive_scores, negative_scores,
                  thresholds, args.num_places, args.focus_place,
                  args.output_dir, dataset_label)

    print(f"\nDone. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
