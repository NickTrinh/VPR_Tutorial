"""
Discovery-then-Recognition Pipeline.

Combines online place discovery with place recognition:
  1. Discover places on reference condition (e.g., day_left) using online algorithm
  2. Compute per-place thresholds from discovered places' negative statistics
  3. Query with a different condition (e.g., day_right)
  4. Evaluate using frame-to-frame ground truth correspondence

Usage:
    python experiments/discovery_then_recognition.py
"""

import os
import pickle
import sys
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import normalize_l2
from experiments.online_place_discovery import (
    OnlineFeatureExtractor, OnlinePlaceDiscovery
)


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Discover places on reference condition
# ──────────────────────────────────────────────────────────────────────────────

def discover_reference_places(ref_img_paths, ref_cache_dir, filter_n_cap=10):
    """Run online place discovery on the reference condition."""
    extractor = OnlineFeatureExtractor(ref_cache_dir)
    discoverer = OnlinePlaceDiscovery(
        min_place_size=3, hysteresis=2, filter_n_cap=filter_n_cap
    )

    print(f"Discovering places on {len(ref_img_paths)} reference images...")
    for i, path in enumerate(ref_img_paths):
        desc = extractor.get_descriptor(path, i)
        discoverer.process_frame(desc, i, verbose=False)

    places = discoverer.places
    descriptors = np.array(discoverer.descriptors)
    print(f"  Found {len(places)} places: {[len(p) for p in places]}")
    return places, descriptors


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Compute per-place thresholds from negative statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_per_place_thresholds(ref_descriptors, places, method="simple_avg"):
    """
    Compute per-place thresholds using negative statistics.

    For each place p, for each image i in p:
      - Compute mean similarity to all images NOT in p (mean_bad for image i)
    Then:
      - simple_avg: θ_p = mean of all per-image mean_bad scores
      - filter_n:   θ_p = mean_bad + min(filter_n, cap) * std_bad

    Returns dict: place_idx -> {threshold, mean_bad, std_bad, mean_good, filter_n}
    """
    n_places = len(places)
    thresholds = {}

    for p_idx in range(n_places):
        target_frames = places[p_idx]
        other_frames = []
        for i, p in enumerate(places):
            if i != p_idx:
                other_frames.extend(p)

        if not other_frames or len(target_frames) < 2:
            thresholds[p_idx] = {
                "threshold": 0.0, "mean_bad": 0.0, "std_bad": 0.1,
                "mean_good": 0.5, "filter_n": 1.0
            }
            continue

        target_descs = ref_descriptors[target_frames]
        other_descs = ref_descriptors[other_frames]

        # Negative: similarity of each target image to all other-place images
        neg_sims = target_descs @ other_descs.T  # (n_target, n_other)
        per_image_mean_bad = neg_sims.mean(axis=1)  # (n_target,)
        mean_bad = float(per_image_mean_bad.mean())
        std_bad = float(per_image_mean_bad.std()) if len(per_image_mean_bad) > 1 else 0.1

        # Positive: within-place similarity (exclude self)
        pos_sims = target_descs @ target_descs.T  # (n_target, n_target)
        np.fill_diagonal(pos_sims, 0)
        n_t = len(target_frames)
        per_image_mean_good = pos_sims.sum(axis=1) / max(n_t - 1, 1)
        mean_good = float(per_image_mean_good.mean())

        # filter_n
        if std_bad < 1e-8:
            filter_n = 1.0
        else:
            filter_n = float(np.floor((mean_good - mean_bad) / std_bad))
        filter_n = max(0, min(filter_n, 10))

        if method == "simple_avg":
            threshold = mean_bad
        elif method == "filter_n":
            threshold = mean_bad + filter_n * std_bad
        else:
            threshold = mean_bad

        thresholds[p_idx] = {
            "threshold": threshold,
            "mean_bad": mean_bad,
            "std_bad": std_bad,
            "mean_good": mean_good,
            "filter_n": filter_n
        }

    return thresholds


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Query with a different condition
# ──────────────────────────────────────────────────────────────────────────────

def extract_query_features(query_img_paths, query_cache_dir):
    """Extract features for all query images."""
    extractor = OnlineFeatureExtractor(query_cache_dir)
    descriptors = []
    print(f"Extracting features for {len(query_img_paths)} query images...")
    for i, path in enumerate(query_img_paths):
        desc = extractor.get_descriptor(path, i)
        descriptors.append(desc)
    return np.array(descriptors)


def recognize_places(query_descriptors, ref_descriptors, places, thresholds,
                     use_threshold=True):
    """
    For each query image, find the best matching place.

    For each query q:
      1. Compute similarity to all reference images
      2. For each place p, compute score = mean similarity to images in p
      3. If use_threshold: filter out places where score < θ_p
      4. Return the place with highest score (or -1 if all filtered)

    Returns:
      predictions: list of predicted place indices (-1 = no match)
      all_scores: (n_query, n_places) matrix of per-place scores
    """
    n_query = len(query_descriptors)
    n_places = len(places)

    # Compute full query-to-reference similarity
    S = query_descriptors @ ref_descriptors.T  # (n_query, n_ref)

    all_scores = np.zeros((n_query, n_places))
    for p_idx, frames in enumerate(places):
        all_scores[:, p_idx] = S[:, frames].mean(axis=1)

    predictions = []
    for q_idx in range(n_query):
        scores = all_scores[q_idx].copy()

        if use_threshold:
            for p_idx in range(n_places):
                if scores[p_idx] < thresholds[p_idx]["threshold"]:
                    scores[p_idx] = -np.inf

        if np.all(scores == -np.inf):
            predictions.append(-1)
        else:
            predictions.append(int(np.argmax(scores)))

    return predictions, all_scores


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Evaluate
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(predictions, all_scores, places, n_query, thresholds=None,
             use_threshold=True):
    """
    Evaluate recognition performance.

    Ground truth: query frame i corresponds to reference frame i.
    A prediction is correct if the predicted place contains the
    corresponding reference frame.

    Reports:
      - Recall@1: correct place is the top prediction
      - Recall@K: correct place is in top K predictions
      - Rejection rate: fraction of queries with no match
    """
    # Build frame-to-place mapping
    frame_to_place = {}
    for p_idx, frames in enumerate(places):
        for f in frames:
            frame_to_place[f] = p_idx

    n_places = len(places)

    # Recall@K
    results = {}
    for K in [1, 3, 5, 10]:
        correct = 0
        for q_idx in range(n_query):
            gt_place = frame_to_place.get(q_idx, -1)
            if gt_place == -1:
                continue  # query frame not in any reference place (shouldn't happen)

            scores = all_scores[q_idx].copy()

            if use_threshold and thresholds is not None:
                for p_idx in range(n_places):
                    if scores[p_idx] < thresholds[p_idx]["threshold"]:
                        scores[p_idx] = -np.inf

            # Top K places
            if np.all(scores == -np.inf):
                continue
            top_k = np.argsort(scores)[::-1][:K]
            # Remove -inf entries
            top_k = [p for p in top_k if scores[p] > -np.inf]

            if gt_place in top_k:
                correct += 1

        results[f"Recall@{K}"] = correct / n_query * 100

    # Rejection rate
    n_rejected = sum(1 for p in predictions if p == -1)
    results["rejection_rate"] = n_rejected / n_query * 100

    # Precision / Recall / F1
    # TP: accepted and predicted place is correct
    # FP: accepted but predicted place is wrong
    # FN: rejected but should have matched (query has a GT place)
    # TN: rejected and truly no match (N/A here — every query has a GT place)
    TP = 0
    FP = 0
    FN = 0
    for q_idx in range(n_query):
        gt_place = frame_to_place.get(q_idx, -1)
        pred = predictions[q_idx]

        if pred == -1:
            # Rejected
            if gt_place != -1:
                FN += 1
        else:
            # Accepted
            if pred == gt_place:
                TP += 1
            else:
                FP += 1

    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results["TP"] = TP
    results["FP"] = FP
    results["FN"] = FN
    results["Precision"] = precision
    results["Recall"] = recall
    results["F1"] = f1

    # Per-place accuracy
    place_correct = np.zeros(n_places)
    place_total = np.zeros(n_places)
    for q_idx in range(n_query):
        gt_place = frame_to_place.get(q_idx, -1)
        if gt_place == -1:
            continue
        place_total[gt_place] += 1
        if predictions[q_idx] == gt_place:
            place_correct[gt_place] += 1

    results["per_place_accuracy"] = {
        p: (place_correct[p] / place_total[p] * 100 if place_total[p] > 0 else 0)
        for p in range(n_places)
    }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(all_scores, places, predictions, thresholds, ref_descriptors,
                 n_query, output_dir, method_name):
    """Generate visualization of recognition results."""
    n_places = len(places)
    frame_to_place = {}
    for p_idx, frames in enumerate(places):
        for f in frames:
            frame_to_place[f] = p_idx

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_places, 1)))

    # ── Top-left: per-place scores for each query ────────────────────
    ax = axes[0, 0]
    im = ax.imshow(all_scores.T, aspect="auto", cmap="hot", vmin=0, vmax=1)
    # Overlay thresholds as horizontal lines
    for p_idx in range(n_places):
        ax.axhline(y=p_idx, color="white", linewidth=0.3, alpha=0.5)
    ax.set_xlabel("Query frame")
    ax.set_ylabel("Place index")
    ax.set_title(f"Per-place scores ({method_name})")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # ── Top-right: predictions vs ground truth ───────────────────────
    ax2 = axes[0, 1]
    gt_places = [frame_to_place.get(q, -1) for q in range(n_query)]
    ax2.scatter(range(n_query), gt_places, s=8, alpha=0.6, color="#2980b9",
                label="Ground truth", zorder=2)
    ax2.scatter(range(n_query), predictions, s=8, alpha=0.6, color="#e74c3c",
                marker="x", label="Predicted", zorder=3)

    # Shade correct/incorrect
    for q_idx in range(n_query):
        if predictions[q_idx] == gt_places[q_idx]:
            ax2.axvspan(q_idx - 0.5, q_idx + 0.5, alpha=0.05, color="green")
        elif predictions[q_idx] != -1:
            ax2.axvspan(q_idx - 0.5, q_idx + 0.5, alpha=0.1, color="red")

    ax2.set_xlabel("Query frame")
    ax2.set_ylabel("Place index")
    ax2.set_title("Predicted vs ground truth place")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    # ── Bottom-left: best score per query with threshold ─────────────
    ax3 = axes[1, 0]
    best_scores = all_scores.max(axis=1)
    best_places = all_scores.argmax(axis=1)
    correct_mask = np.array([best_places[q] == gt_places[q] for q in range(n_query)])

    ax3.scatter(np.where(correct_mask)[0], best_scores[correct_mask],
                s=10, alpha=0.6, color="#27ae60", label="Correct", zorder=2)
    ax3.scatter(np.where(~correct_mask)[0], best_scores[~correct_mask],
                s=10, alpha=0.6, color="#e74c3c", label="Wrong", zorder=2)

    # Plot threshold for the GT place of each query
    gt_thresholds = [thresholds[gt_places[q]]["threshold"]
                     for q in range(n_query)]
    ax3.plot(range(n_query), gt_thresholds, color="gray", linewidth=0.8,
             alpha=0.5, label="GT place θ")

    ax3.set_xlabel("Query frame")
    ax3.set_ylabel("Best place score")
    ax3.set_title("Best match score per query")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.2)

    # ── Bottom-right: per-place accuracy bar chart ───────────────────
    ax4 = axes[1, 1]
    place_correct = np.zeros(n_places)
    place_total = np.zeros(n_places)
    for q_idx in range(n_query):
        gt = gt_places[q_idx]
        if gt >= 0:
            place_total[gt] += 1
            if predictions[q_idx] == gt:
                place_correct[gt] += 1

    accuracies = np.where(place_total > 0, place_correct / place_total * 100, 0)
    bars = ax4.bar(range(n_places), accuracies, color=colors[:n_places], alpha=0.8)
    ax4.set_xlabel("Place index")
    ax4.set_ylabel("Accuracy (%)")
    ax4.set_title("Per-place recognition accuracy")
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.2, axis="y")

    # Add place size labels
    for p_idx in range(n_places):
        ax4.text(p_idx, accuracies[p_idx] + 2, f"n={int(place_total[p_idx])}",
                 ha="center", fontsize=6, rotation=45)

    plt.suptitle(f"Discovery-then-Recognition: {method_name}\n"
                 f"{n_places} auto-discovered places on reference, "
                 f"{n_query} query images",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, f"recognition_{method_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Discovery-then-Recognition pipeline")
    p.add_argument("--data_dir", default="images/GardensPoint",
                   help="Dataset directory containing condition subfolders")
    p.add_argument("--dataset_name", default=None,
                   help="Dataset name for cache dir (auto-detected from data_dir if not set)")
    p.add_argument("--ref_condition", default="day_left",
                   help="Reference condition subfolder")
    p.add_argument("--query_condition", default="day_right",
                   help="Query condition subfolder")
    p.add_argument("--descriptor", default="eigenplaces")
    p.add_argument("--max_images", type=int, default=200,
                   help="Max images to use (0 = all)")
    p.add_argument("--filter_n_cap", type=int, default=10)
    p.add_argument("--img_ext", default="*.jpg",
                   help="Image file extension glob pattern")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (auto-generated if not set)")
    return p.parse_args()


def main():
    args = parse_args()

    # Auto-detect dataset name from data_dir
    dataset_name = args.dataset_name or os.path.basename(args.data_dir.rstrip("/"))

    output_dir = args.output_dir or f"results/visualizations/discovery_recognition_{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Paths
    ref_condition = args.ref_condition
    query_condition = args.query_condition
    data_dir = args.data_dir

    ref_img_paths = sorted(glob(os.path.join(data_dir, ref_condition, args.img_ext)))
    query_img_paths = sorted(glob(os.path.join(data_dir, query_condition, args.img_ext)))
    if args.max_images > 0:
        ref_img_paths = ref_img_paths[:args.max_images]
        query_img_paths = query_img_paths[:args.max_images]
    N = len(ref_img_paths)

    ref_cache_dir = f"cache/{dataset_name}/{ref_condition}/{args.descriptor}"
    query_cache_dir = f"cache/{dataset_name}/{query_condition}/{args.descriptor}"

    # Step 1: Discover places on reference
    print("=" * 60)
    print("STEP 1: Online Place Discovery (reference)")
    print("=" * 60)
    places, ref_descriptors = discover_reference_places(
        ref_img_paths, ref_cache_dir, filter_n_cap=args.filter_n_cap
    )

    # Step 2: Extract query features
    print("\n" + "=" * 60)
    print("STEP 2: Extract Query Features")
    print("=" * 60)
    query_descriptors = extract_query_features(query_img_paths, query_cache_dir)

    # Step 3 & 4: Recognize and evaluate with different threshold methods
    for method in ["simple_avg", "filter_n"]:
        print(f"\n{'=' * 60}")
        print(f"STEP 3-4: Recognition & Evaluation (method={method})")
        print("=" * 60)

        # Compute thresholds
        thresholds = compute_per_place_thresholds(
            ref_descriptors, places, method=method
        )
        print(f"\nPer-place thresholds ({method}):")
        for p_idx, info in thresholds.items():
            print(f"  Place {p_idx} ({len(places[p_idx]):>3} frames): "
                  f"θ={info['threshold']:.3f}  "
                  f"mean_bad={info['mean_bad']:.3f}  "
                  f"std_bad={info['std_bad']:.3f}  "
                  f"mean_good={info['mean_good']:.3f}  "
                  f"filter_n={info['filter_n']:.0f}")

        # With threshold
        print(f"\n--- With threshold ({method}) ---")
        preds_thresh, scores = recognize_places(
            query_descriptors, ref_descriptors, places, thresholds,
            use_threshold=True
        )
        results_thresh = evaluate(
            preds_thresh, scores, places, N, thresholds, use_threshold=True
        )
        for k, v in results_thresh.items():
            if k == "per_place_accuracy":
                continue
            elif k in ("TP", "FP", "FN"):
                print(f"  {k}: {int(v)}")
            elif k in ("Precision", "Recall", "F1"):
                print(f"  {k}: {v:.1f}%")
            else:
                print(f"  {k}: {v:.1f}%")

        plot_results(scores, places, preds_thresh, thresholds,
                     ref_descriptors, N, output_dir,
                     f"With threshold ({method})")

        # Without threshold (baseline)
        print(f"\n--- Without threshold (baseline) ---")
        preds_base, scores_base = recognize_places(
            query_descriptors, ref_descriptors, places, thresholds,
            use_threshold=False
        )
        results_base = evaluate(
            preds_base, scores_base, places, N, thresholds=None,
            use_threshold=False
        )
        for k, v in results_base.items():
            if k == "per_place_accuracy":
                continue
            elif k in ("TP", "FP", "FN"):
                print(f"  {k}: {int(v)}")
            elif k in ("Precision", "Recall", "F1"):
                print(f"  {k}: {v:.1f}%")
            else:
                print(f"  {k}: {v:.1f}%")

        plot_results(scores_base, places, preds_base, thresholds,
                     ref_descriptors, N, output_dir,
                     f"No threshold (baseline)")

    # Summary comparison
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Reference: {ref_condition} ({N} images)")
    print(f"Query: {query_condition} ({N} images)")
    print(f"Discovered places: {len(places)}")
    print(f"Place sizes: {[len(p) for p in places]}")
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
