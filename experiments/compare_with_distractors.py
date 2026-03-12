"""
Compare threshold methods with distractor queries.

Setup:
  - Reference: GardensPoint day_left (discovered places)
  - Genuine queries: GardensPoint day_right (should match)
  - Distractors: SFU dry images (should be REJECTED — no matching place)

This tests whether thresholding actually helps distinguish known from unknown.

Usage:
    python experiments/compare_with_distractors.py
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

    return thresholds


def evaluate_with_distractors(query_descs, ref_descs, places, n_genuine,
                               place_thresholds=None, per_query_thresholds=None,
                               use_threshold=True):
    """
    Evaluate with mixed genuine + distractor queries.

    Args:
        query_descs: (n_genuine + n_distractor, dim) — genuine first, then distractors
        ref_descs: reference descriptors
        places: discovered places
        n_genuine: number of genuine queries (first n_genuine are genuine)
        place_thresholds: dict p_idx -> threshold (our method)
        per_query_thresholds: array of per-query thresholds (Vysotska method)
        use_threshold: whether to apply thresholding
    """
    S = query_descs @ ref_descs.T
    n_query = len(query_descs)
    n_places = len(places)

    frame_to_place = {}
    for p_idx, frames in enumerate(places):
        for f in frames:
            frame_to_place[f] = p_idx

    # Per-place scores
    place_scores = np.zeros((n_query, n_places))
    for p_idx, frames in enumerate(places):
        place_scores[:, p_idx] = S[:, frames].mean(axis=1)

    # TP: genuine, correctly matched
    # FP: distractor accepted (or genuine matched to wrong place)
    # FN: genuine rejected
    # TN: distractor correctly rejected
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for q_idx in range(n_query):
        is_genuine = q_idx < n_genuine
        scores = place_scores[q_idx].copy()

        if use_threshold:
            if place_thresholds is not None:
                for p_idx in range(n_places):
                    if scores[p_idx] < place_thresholds[p_idx]:
                        scores[p_idx] = -np.inf
            elif per_query_thresholds is not None:
                theta = per_query_thresholds[q_idx]
                for p_idx in range(n_places):
                    if scores[p_idx] < theta:
                        scores[p_idx] = -np.inf

        if np.all(scores == -np.inf):
            pred = -1  # rejected
        else:
            pred = int(np.argmax(scores))

        if is_genuine:
            gt_place = frame_to_place.get(q_idx, -1)
            if pred == -1:
                FN += 1  # genuine but rejected
            elif pred == gt_place:
                TP += 1  # genuine and correct
            else:
                FP += 1  # genuine but wrong place
        else:
            # Distractor
            if pred == -1:
                TN += 1  # distractor correctly rejected
            else:
                FP += 1  # distractor incorrectly accepted

    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Distractor-specific metrics
    n_distractor = n_query - n_genuine
    distractor_reject_rate = TN / n_distractor * 100 if n_distractor > 0 else 0
    genuine_accept_rate = (TP + (n_genuine - TP - FN)) / n_genuine * 100  # not quite right
    genuine_accept_rate = (n_genuine - FN) / n_genuine * 100

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "distractor_reject_rate": distractor_reject_rate,
        "genuine_accept_rate": genuine_accept_rate,
    }


def plot_comparison(all_results, output_dir):
    methods = list(all_results.keys())
    metrics = ["Precision", "Recall", "F1", "distractor_reject_rate"]
    labels = ["Precision", "Recall", "F1", "Distractor Rejection"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    colors = ["#2980b9", "#27ae60", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c"]

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        values = [all_results[m][metric] for m in methods]
        bars = axes[i].bar(range(len(methods)), values,
                           color=colors[:len(methods)], alpha=0.8)
        axes[i].set_xticks(range(len(methods)))
        axes[i].set_xticklabels(methods, rotation=30, ha="right", fontsize=7)
        axes[i].set_ylabel(f"{label} (%)")
        axes[i].set_title(label)
        axes[i].set_ylim(0, 105)
        axes[i].grid(True, alpha=0.2, axis="y")

        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, val + 1,
                         f"{val:.1f}", ha="center", fontsize=7)

    plt.suptitle("Threshold Comparison WITH Distractors\n"
                 "(genuine queries + unknown-location distractors)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "distractor_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_genuine", type=int, default=100)
    p.add_argument("--n_distractors", type=int, default=100)
    p.add_argument("--filter_n_cap", type=int, default=10)
    p.add_argument("--output_dir", default="results/visualizations/distractor_comparison")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Reference: GardensPoint day_left (all 200)
    ref_paths = sorted(glob("images/GardensPoint/day_left/*.jpg"))[:200]
    ref_cache = "cache/GardensPoint/day_left/eigenplaces"

    # Genuine queries: GardensPoint day_right (first n_genuine)
    genuine_paths = sorted(glob("images/GardensPoint/day_right/*.jpg"))[:args.n_genuine]
    genuine_cache = "cache/GardensPoint/day_right/eigenplaces"

    # Distractors: SFU dry (first n_distractors)
    distractor_paths = sorted(glob("images/SFU/dry/*.jpg"))[:args.n_distractors]
    distractor_cache = "cache/SFU/dry/eigenplaces"

    print(f"Reference: GardensPoint day_left ({len(ref_paths)} images)")
    print(f"Genuine queries: GardensPoint day_right ({len(genuine_paths)} images)")
    print(f"Distractors: SFU dry ({len(distractor_paths)} images)")

    # Discover places
    print("\n--- Discovering places ---")
    places, ref_descs = discover_places(ref_paths, ref_cache, args.filter_n_cap)
    print(f"  {len(places)} places")

    # Extract query features
    print("\n--- Extracting features ---")
    genuine_descs = extract_features(genuine_paths, genuine_cache)
    distractor_descs = extract_features(distractor_paths, distractor_cache)

    # Combine: genuine first, then distractors
    query_descs = np.vstack([genuine_descs, distractor_descs])
    n_genuine = len(genuine_descs)
    n_total = len(query_descs)
    print(f"  Total queries: {n_total} ({n_genuine} genuine + {n_total - n_genuine} distractors)")

    all_results = {}

    # 1. No threshold (baseline)
    print("\n--- Baseline (no threshold) ---")
    results = evaluate_with_distractors(
        query_descs, ref_descs, places, n_genuine, use_threshold=False
    )
    all_results["Baseline\n(no threshold)"] = results
    print(f"  P={results['Precision']:.1f}%  R={results['Recall']:.1f}%  "
          f"F1={results['F1']:.1f}%  "
          f"Distractor rejection={results['distractor_reject_rate']:.1f}%")

    # 2. Ours: simple_avg
    print("\n--- Ours: mean_bad ---")
    thresh_simple = compute_our_thresholds(ref_descs, places, "simple_avg")
    results = evaluate_with_distractors(
        query_descs, ref_descs, places, n_genuine,
        place_thresholds=thresh_simple
    )
    all_results["Ours\n(mean_bad)"] = results
    print(f"  P={results['Precision']:.1f}%  R={results['Recall']:.1f}%  "
          f"F1={results['F1']:.1f}%  "
          f"Distractor rejection={results['distractor_reject_rate']:.1f}%")

    # 3. Ours: filter_n
    print("\n--- Ours: filter_n ---")
    thresh_fn = compute_our_thresholds(ref_descs, places, "filter_n")
    results = evaluate_with_distractors(
        query_descs, ref_descs, places, n_genuine,
        place_thresholds=thresh_fn
    )
    all_results["Ours\n(filter_n)"] = results
    print(f"  P={results['Precision']:.1f}%  R={results['Recall']:.1f}%  "
          f"F1={results['F1']:.1f}%  "
          f"Distractor rejection={results['distractor_reject_rate']:.1f}%")

    # 4. Vysotska (argmax)
    print("\n--- Vysotska (GMM + KS + KF) ---")
    S_full = query_descs @ ref_descs.T
    vysotska = VysotskaDaptiveThreshold(patch_size=20)
    per_q_thresh, _ = vysotska.compute_thresholds(S_full)
    results = evaluate_with_distractors(
        query_descs, ref_descs, places, n_genuine,
        per_query_thresholds=per_q_thresh
    )
    all_results["Vysotska\n(GMM+KS+KF)"] = results
    print(f"  P={results['Precision']:.1f}%  R={results['Recall']:.1f}%  "
          f"F1={results['F1']:.1f}%  "
          f"Distractor rejection={results['distractor_reject_rate']:.1f}%")

    # Summary
    print(f"\n{'='*90}")
    print(f"{'Method':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} "
          f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Dist Rej%':>10}")
    print(f"{'-'*90}")
    for name, r in all_results.items():
        clean = name.replace('\n', ' ')
        print(f"{clean:<25} {r['Precision']:>5.1f}% {r['Recall']:>5.1f}% "
              f"{r['F1']:>5.1f}% {r['TP']:>5} {r['FP']:>5} {r['FN']:>5} "
              f"{r['TN']:>5} {r['distractor_reject_rate']:>9.1f}%")
    print(f"{'='*90}")

    plot_comparison(all_results, args.output_dir)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
