"""
Compare threshold methods head-to-head:
  1. Our method: negative statistics (mean_bad, mean_bad + filter_n * std_bad)
  2. Vysotska: GMM + KS test + Kalman filter on sliding window patches

Both use the same dataset, descriptors, and evaluation pipeline.

Usage:
    python experiments/compare_thresholds.py [--args]
"""

import argparse
import os
import sys
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# Add Vysotska's sequence matcher to path
sys.path.insert(0, "/tmp/image_sequence_matcher")
from utils import normalize_l2
from experiments.online_place_discovery import OnlineFeatureExtractor, OnlinePlaceDiscovery
from experiments.vysotska_threshold import VysotskaDaptiveThreshold


def discover_places(ref_paths, cache_dir, filter_n_cap=10):
    extractor = OnlineFeatureExtractor(cache_dir)
    discoverer = OnlinePlaceDiscovery(
        min_place_size=3, hysteresis=2, filter_n_cap=filter_n_cap
    )
    for i, p in enumerate(ref_paths):
        desc = extractor.get_descriptor(p, i)
        discoverer.process_frame(desc, i, verbose=False)
    return discoverer.places, np.array(discoverer.descriptors)


def extract_features(img_paths, cache_dir):
    extractor = OnlineFeatureExtractor(cache_dir)
    descs = [extractor.get_descriptor(p, i) for i, p in enumerate(img_paths)]
    return np.array(descs)


def compute_our_thresholds(ref_descs, places, method="simple_avg"):
    """Per-place thresholds from negative statistics."""
    thresholds = {}
    for p_idx in range(len(places)):
        target = places[p_idx]
        other = [f for i, p in enumerate(places) if i != p_idx for f in p]

        if not other or len(target) < 2:
            thresholds[p_idx] = 0.0
            continue

        neg_sims = ref_descs[target] @ ref_descs[other].T
        per_img_mean_bad = neg_sims.mean(axis=1)
        mean_bad = float(per_img_mean_bad.mean())
        std_bad = float(per_img_mean_bad.std()) if len(per_img_mean_bad) > 1 else 0.1

        pos_sims = ref_descs[target] @ ref_descs[target].T
        np.fill_diagonal(pos_sims, 0)
        per_img_mean_good = pos_sims.sum(axis=1) / max(len(target) - 1, 1)
        mean_good = float(per_img_mean_good.mean())

        if std_bad < 1e-8:
            filter_n = 1.0
        else:
            filter_n = float(np.floor((mean_good - mean_bad) / std_bad))
        filter_n = max(0, min(filter_n, 10))

        if method == "simple_avg":
            thresholds[p_idx] = mean_bad
        elif method == "filter_n":
            thresholds[p_idx] = mean_bad + filter_n * std_bad
        else:
            thresholds[p_idx] = mean_bad

    return thresholds


def evaluate_our_method(query_descs, ref_descs, places, place_thresholds,
                        use_threshold=True):
    """Evaluate using our per-place thresholds."""
    S = query_descs @ ref_descs.T
    n_query = len(query_descs)
    n_places = len(places)

    # Frame-to-place mapping (GT: query i corresponds to ref i)
    frame_to_place = {}
    for p_idx, frames in enumerate(places):
        for f in frames:
            frame_to_place[f] = p_idx

    # Per-place scores
    place_scores = np.zeros((n_query, n_places))
    for p_idx, frames in enumerate(places):
        place_scores[:, p_idx] = S[:, frames].mean(axis=1)

    TP, FP, FN = 0, 0, 0
    recall_at = {1: 0, 3: 0, 5: 0, 10: 0}

    for q_idx in range(n_query):
        gt_place = frame_to_place.get(q_idx, -1)
        scores = place_scores[q_idx].copy()

        if use_threshold:
            for p_idx in range(n_places):
                if scores[p_idx] < place_thresholds[p_idx]:
                    scores[p_idx] = -np.inf

        if np.all(scores == -np.inf):
            pred = -1
        else:
            pred = int(np.argmax(scores))

        # Precision/Recall
        if pred == -1:
            if gt_place != -1:
                FN += 1
        else:
            if pred == gt_place:
                TP += 1
            else:
                FP += 1

        # Recall@K
        for K in recall_at:
            if np.all(scores == -np.inf):
                continue
            top_k = np.argsort(scores)[::-1][:K]
            top_k = [p for p in top_k if scores[p] > -np.inf]
            if gt_place in top_k:
                recall_at[K] += 1

    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": TP, "FP": FP, "FN": FN,
        **{f"Recall@{K}": v / n_query * 100 for K, v in recall_at.items()},
        "rejection_rate": FN / n_query * 100,
    }


def _vysotska_eval_common(query_descs, ref_descs, places, per_query_thresholds):
    """Shared evaluation logic for Vysotska variants."""
    S = query_descs @ ref_descs.T
    n_query = len(query_descs)
    n_places = len(places)

    frame_to_place = {}
    for p_idx, frames in enumerate(places):
        for f in frames:
            frame_to_place[f] = p_idx

    place_scores = np.zeros((n_query, n_places))
    for p_idx, frames in enumerate(places):
        place_scores[:, p_idx] = S[:, frames].mean(axis=1)

    TP, FP, FN = 0, 0, 0
    recall_at = {1: 0, 3: 0, 5: 0, 10: 0}

    for q_idx in range(n_query):
        gt_place = frame_to_place.get(q_idx, -1)
        scores = place_scores[q_idx].copy()
        theta = per_query_thresholds[q_idx]

        best_score = scores.max()
        if best_score < theta:
            pred = -1
        else:
            for p_idx in range(n_places):
                if scores[p_idx] < theta:
                    scores[p_idx] = -np.inf
            pred = int(np.argmax(scores))

        if pred == -1:
            if gt_place != -1:
                FN += 1
        else:
            if pred == gt_place:
                TP += 1
            else:
                FP += 1

        for K in recall_at:
            s = place_scores[q_idx].copy()
            for p_idx in range(n_places):
                if s[p_idx] < theta:
                    s[p_idx] = -np.inf
            if np.all(s == -np.inf):
                continue
            top_k = np.argsort(s)[::-1][:K]
            top_k = [p for p in top_k if s[p] > -np.inf]
            if gt_place in top_k:
                recall_at[K] += 1

    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": TP, "FP": FP, "FN": FN,
        **{f"Recall@{K}": v / n_query * 100 for K, v in recall_at.items()},
        "rejection_rate": FN / n_query * 100,
    }


def evaluate_vysotska(query_descs, ref_descs, places, patch_size=20):
    """
    Option B: Vysotska threshold with simple argmax matching (no sequence matcher).
    """
    S = query_descs @ ref_descs.T

    vysotska = VysotskaDaptiveThreshold(patch_size=patch_size)
    per_query_thresholds, history = vysotska.compute_thresholds(S)

    results = _vysotska_eval_common(query_descs, ref_descs, places,
                                     per_query_thresholds)
    return results, per_query_thresholds, history


def evaluate_vysotska_with_seqmatch(query_descs, ref_descs, places,
                                     patch_size=20, fanout=5):
    """
    Option A: Vysotska threshold with their graph-based sequence matcher.

    Uses Vysotska's sequence matcher to find optimal query-reference pairings,
    then computes adaptive thresholds on patches around those pairings.
    """
    from src.graph import Graph
    import src.path_tools as pt

    S = query_descs @ ref_descs.T
    n_query, n_ref = S.shape

    # Convert similarity to cost (their code expects lower = better)
    # Their code uses positive costs, so: cost = max_sim - sim
    cost_matrix = S.max() - S + 1e-6  # small offset to keep positive

    print(f"    Running sequence matcher (fanout={fanout})...")
    graph = Graph()
    graph.initFromMatrix(cost_matrix, fanout)
    graph.computePath()
    img_path = graph.getImageCorrespodences()

    # img_path is array of [query_idx, ref_idx] pairs
    # Build best_matches array: for each query, which ref was matched
    best_matches = np.argmax(S, axis=1)  # default fallback
    for pair in img_path:
        q_idx, r_idx = int(pair[0]), int(pair[1])
        if 0 <= q_idx < n_query and 0 <= r_idx < n_ref:
            best_matches[q_idx] = r_idx

    print(f"    Sequence matcher found {len(img_path)} correspondences")

    # Compute Vysotska thresholds using sequence-matched pairs
    vysotska = VysotskaDaptiveThreshold(patch_size=patch_size)
    per_query_thresholds, history = vysotska.compute_thresholds(
        S, best_matches=best_matches
    )

    results = _vysotska_eval_common(query_descs, ref_descs, places,
                                     per_query_thresholds)
    return results, per_query_thresholds, history


def plot_comparison(all_results, per_query_thresholds, output_dir):
    """Plot comparison of all methods."""
    methods = list(all_results.keys())
    metrics = ["Recall@1", "Precision", "Recall", "F1"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    for i, metric in enumerate(metrics):
        values = [all_results[m][metric] for m in methods]
        colors = ["#2980b9", "#27ae60", "#e74c3c", "#f39c12", "#9b59b6"]
        bars = axes[i].bar(range(len(methods)), values,
                           color=colors[:len(methods)], alpha=0.8)
        axes[i].set_xticks(range(len(methods)))
        axes[i].set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        axes[i].set_ylabel(f"{metric} (%)")
        axes[i].set_title(metric)
        axes[i].set_ylim(0, 105)
        axes[i].grid(True, alpha=0.2, axis="y")

        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, val + 1,
                         f"{val:.1f}", ha="center", fontsize=8)

    plt.suptitle("Threshold Method Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "threshold_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Compare threshold methods")
    p.add_argument("--data_dir", default="images/GardensPoint")
    p.add_argument("--ref_condition", default="day_left")
    p.add_argument("--query_condition", default="day_right")
    p.add_argument("--descriptor", default="eigenplaces")
    p.add_argument("--max_images", type=int, default=200)
    p.add_argument("--filter_n_cap", type=int, default=10)
    p.add_argument("--img_ext", default="*.jpg")
    p.add_argument("--output_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    dataset_name = os.path.basename(args.data_dir.rstrip("/"))
    output_dir = args.output_dir or f"results/visualizations/compare_thresholds_{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Load images
    ref_paths = sorted(glob(os.path.join(args.data_dir, args.ref_condition, args.img_ext)))
    query_paths = sorted(glob(os.path.join(args.data_dir, args.query_condition, args.img_ext)))
    if args.max_images > 0:
        ref_paths = ref_paths[:args.max_images]
        query_paths = query_paths[:args.max_images]
    N = len(ref_paths)
    print(f"Reference: {args.ref_condition} ({N} images)")
    print(f"Query: {args.query_condition} ({N} images)")

    ref_cache = f"cache/{dataset_name}/{args.ref_condition}/{args.descriptor}"
    query_cache = f"cache/{dataset_name}/{args.query_condition}/{args.descriptor}"

    # Step 1: Discover places
    print("\n--- Discovering places on reference ---")
    places, ref_descs = discover_places(ref_paths, ref_cache, args.filter_n_cap)
    print(f"  {len(places)} places: {[len(p) for p in places]}")

    # Step 2: Extract query features
    print("\n--- Extracting query features ---")
    query_descs = extract_features(query_paths, query_cache)

    all_results = {}

    # Method 1: No threshold (baseline)
    print("\n--- Baseline (no threshold) ---")
    dummy_thresholds = {p: -np.inf for p in range(len(places))}
    results = evaluate_our_method(query_descs, ref_descs, places,
                                  dummy_thresholds, use_threshold=True)
    all_results["Baseline\n(no threshold)"] = results
    print(f"  R@1={results['Recall@1']:.1f}%  P={results['Precision']:.1f}%  "
          f"R={results['Recall']:.1f}%  F1={results['F1']:.1f}%")

    # Method 2: Our simple_avg
    print("\n--- Ours: simple_avg (θ = mean_bad) ---")
    our_thresh_simple = compute_our_thresholds(ref_descs, places, "simple_avg")
    results = evaluate_our_method(query_descs, ref_descs, places,
                                  our_thresh_simple, use_threshold=True)
    all_results["Ours\n(mean_bad)"] = results
    print(f"  R@1={results['Recall@1']:.1f}%  P={results['Precision']:.1f}%  "
          f"R={results['Recall']:.1f}%  F1={results['F1']:.1f}%")

    # Method 3: Our filter_n
    print("\n--- Ours: filter_n (θ = mean_bad + filter_n * std_bad) ---")
    our_thresh_fn = compute_our_thresholds(ref_descs, places, "filter_n")
    results = evaluate_our_method(query_descs, ref_descs, places,
                                  our_thresh_fn, use_threshold=True)
    all_results["Ours\n(filter_n)"] = results
    print(f"  R@1={results['Recall@1']:.1f}%  P={results['Precision']:.1f}%  "
          f"R={results['Recall']:.1f}%  F1={results['F1']:.1f}%")

    # Method 4: Vysotska adaptive (no sequence matcher — Option B)
    print("\n--- Vysotska: GMM + KS + Kalman, argmax matching (patch=20) ---")
    vysotska_results, per_query_thresh, vysotska_history = evaluate_vysotska(
        query_descs, ref_descs, places, patch_size=20
    )
    all_results["Vysotska\n(argmax)"] = vysotska_results
    print(f"  R@1={vysotska_results['Recall@1']:.1f}%  "
          f"P={vysotska_results['Precision']:.1f}%  "
          f"R={vysotska_results['Recall']:.1f}%  "
          f"F1={vysotska_results['F1']:.1f}%")

    # Method 5: Vysotska adaptive + sequence matcher (Option A)
    print("\n--- Vysotska: GMM + KS + Kalman + sequence matcher (patch=20) ---")
    try:
        vysotska_seq_results, _, _ = evaluate_vysotska_with_seqmatch(
            query_descs, ref_descs, places, patch_size=20, fanout=5
        )
        all_results["Vysotska\n(seq match)"] = vysotska_seq_results
        print(f"  R@1={vysotska_seq_results['Recall@1']:.1f}%  "
              f"P={vysotska_seq_results['Precision']:.1f}%  "
              f"R={vysotska_seq_results['Recall']:.1f}%  "
              f"F1={vysotska_seq_results['F1']:.1f}%")
    except Exception as e:
        print(f"  Sequence matcher failed: {e}")
        print("  (Make sure image_sequence_matcher is cloned to /tmp/)")

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Method':<30} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'Prec':>6} "
          f"{'Rec':>6} {'F1':>6} {'Rej%':>6}")
    print(f"{'-'*80}")
    for name, r in all_results.items():
        clean = name.replace('\n', ' ')
        print(f"{clean:<30} {r['Recall@1']:>5.1f}% {r['Recall@3']:>5.1f}% "
              f"{r['Recall@5']:>5.1f}% {r['Precision']:>5.1f}% "
              f"{r['Recall']:>5.1f}% {r['F1']:>5.1f}% "
              f"{r['rejection_rate']:>5.1f}%")
    print(f"{'='*80}")

    # Plot
    plot_comparison(all_results, per_query_thresh, output_dir)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
