"""
Discovery-then-Recognition on GardensPoint Mini.

Fair comparison to the RCC paper:
  - Same dataset (20 places, group=3, step=10)
  - Discover places on day_left (60 images)
  - Query with day_right (60 images)
  - Evaluate Recall@K and compare to paper's Table I

Usage:
    python experiments/discovery_recognition_mini.py
"""

import os
import re
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


def get_place_from_filename(fname):
    """Extract place index from filename like Place0003_Cond01_G02.jpg"""
    m = re.match(r'Place(\d+)_Cond(\d+)_G(\d+)', os.path.basename(fname))
    if m:
        return int(m.group(1))
    return -1


def main():
    output_dir = "results/visualizations/discovery_recognition_mini"
    os.makedirs(output_dir, exist_ok=True)

    data_dir = "images/GardensPoint_Mini"
    ref_condition = "day_left"
    query_condition = "day_right"

    ref_paths = sorted(glob(os.path.join(data_dir, ref_condition, "*.jpg")))
    query_paths = sorted(glob(os.path.join(data_dir, query_condition, "*.jpg")))
    N_ref = len(ref_paths)
    N_query = len(query_paths)
    print(f"Reference ({ref_condition}): {N_ref} images")
    print(f"Query ({query_condition}): {N_query} images")

    # Ground truth: place label for each image
    ref_gt = [get_place_from_filename(p) for p in ref_paths]
    query_gt = [get_place_from_filename(p) for p in query_paths]
    n_gt_places = len(set(ref_gt))
    print(f"Ground truth places: {n_gt_places}")
    print(f"Ref place sequence: {ref_gt}")

    # ── Step 1: Extract features ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: Extract features")
    print("=" * 60)

    ref_cache = f"cache/GardensPoint_Mini/{ref_condition}/eigenplaces"
    query_cache = f"cache/GardensPoint_Mini/{query_condition}/eigenplaces"

    ref_extractor = OnlineFeatureExtractor(ref_cache)
    query_extractor = OnlineFeatureExtractor(query_cache)

    ref_descs = np.array([ref_extractor.get_descriptor(p, i)
                          for i, p in enumerate(ref_paths)])
    query_descs = np.array([query_extractor.get_descriptor(p, i)
                            for i, p in enumerate(query_paths)])
    print(f"  Ref descriptors: {ref_descs.shape}")
    print(f"  Query descriptors: {query_descs.shape}")

    # ── Step 2: Discover places on reference ─────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Online Place Discovery on reference")
    print("=" * 60)

    discoverer = OnlinePlaceDiscovery(
        min_place_size=2, hysteresis=2, filter_n_cap=10
    )
    for i in range(N_ref):
        discoverer.process_frame(ref_descs[i], i, verbose=True)

    disc_places = discoverer.places
    print(f"\nDiscovered {len(disc_places)} places: {[len(p) for p in disc_places]}")

    # Show mapping to ground truth
    print("\nDiscovered → Ground truth mapping:")
    for pi, frames in enumerate(disc_places):
        gt_labels = [ref_gt[f] for f in frames]
        print(f"  Disc Place {pi} (frames {frames}): GT places {gt_labels}")

    # ── Step 3: Compute per-place thresholds ─────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Compute per-place thresholds")
    print("=" * 60)

    def compute_thresholds(descs, places, method="simple_avg"):
        thresholds = {}
        for p_idx in range(len(places)):
            target = places[p_idx]
            other = [f for i, p in enumerate(places) if i != p_idx for f in p]

            if not other or len(target) < 2:
                thresholds[p_idx] = {"threshold": 0.0, "mean_bad": 0.0,
                                     "std_bad": 0.1, "mean_good": 0.5,
                                     "filter_n": 1.0}
                continue

            neg_sims = descs[target] @ descs[other].T
            per_img_mean_bad = neg_sims.mean(axis=1)
            mean_bad = float(per_img_mean_bad.mean())
            std_bad = float(per_img_mean_bad.std()) if len(per_img_mean_bad) > 1 else 0.1

            pos_sims = descs[target] @ descs[target].T
            np.fill_diagonal(pos_sims, 0)
            per_img_mean_good = pos_sims.sum(axis=1) / max(len(target) - 1, 1)
            mean_good = float(per_img_mean_good.mean())

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
                "threshold": threshold, "mean_bad": mean_bad,
                "std_bad": std_bad, "mean_good": mean_good,
                "filter_n": filter_n
            }
        return thresholds

    # ── Step 4: Recognize and evaluate ───────────────────────────────
    # Build discovered-place to GT-place mapping
    # A query is correct if it matches a discovered place that contains
    # the corresponding GT reference frame.
    #
    # For each query image q with GT place label g:
    #   Find which discovered place(s) contain reference images from GT place g
    #   If the top-K predicted discovered places include any of those, it's correct

    # Map: for each query index, which discovered place(s) are correct?
    query_correct_disc_places = {}
    for q_idx in range(N_query):
        gt_place = query_gt[q_idx]
        # Find reference frames with same GT place
        matching_ref_frames = [f for f in range(N_ref) if ref_gt[f] == gt_place]
        # Find which discovered places contain those frames
        correct_discs = set()
        for dp_idx, dp_frames in enumerate(disc_places):
            if any(f in dp_frames for f in matching_ref_frames):
                correct_discs.add(dp_idx)
        query_correct_disc_places[q_idx] = correct_discs

    def evaluate(query_descs, ref_descs, disc_places, thresholds,
                 use_threshold=True):
        S = query_descs @ ref_descs.T
        n_disc = len(disc_places)

        # Per-place scores
        all_scores = np.zeros((N_query, n_disc))
        for dp_idx, frames in enumerate(disc_places):
            all_scores[:, dp_idx] = S[:, frames].mean(axis=1)

        results = {}
        for K in [1, 3, 5, 10]:
            correct = 0
            for q_idx in range(N_query):
                scores = all_scores[q_idx].copy()
                if use_threshold:
                    for dp_idx in range(n_disc):
                        if scores[dp_idx] < thresholds[dp_idx]["threshold"]:
                            scores[dp_idx] = -np.inf

                if np.all(scores == -np.inf):
                    continue

                top_k = np.argsort(scores)[::-1][:K]
                top_k = [p for p in top_k if scores[p] > -np.inf]

                if any(p in query_correct_disc_places[q_idx] for p in top_k):
                    correct += 1

            results[f"Recall@{K}"] = correct / N_query * 100

        # Rejection rate
        preds = []
        for q_idx in range(N_query):
            scores = all_scores[q_idx].copy()
            if use_threshold:
                for dp_idx in range(n_disc):
                    if scores[dp_idx] < thresholds[dp_idx]["threshold"]:
                        scores[dp_idx] = -np.inf
            if np.all(scores == -np.inf):
                preds.append(-1)
            else:
                preds.append(int(np.argmax(scores)))
        n_rejected = sum(1 for p in preds if p == -1)
        results["rejection_rate"] = n_rejected / N_query * 100

        return results, all_scores, preds

    # Run all methods
    all_results = {}

    for method in ["simple_avg", "filter_n"]:
        thresholds = compute_thresholds(ref_descs, disc_places, method=method)

        print(f"\n--- Method: {method} ---")
        for p_idx, info in thresholds.items():
            print(f"  Place {p_idx} ({len(disc_places[p_idx])} frames): "
                  f"θ={info['threshold']:.3f}  "
                  f"mean_bad={info['mean_bad']:.3f}  "
                  f"filter_n={info['filter_n']:.0f}")

        # With threshold
        results_t, scores_t, preds_t = evaluate(
            query_descs, ref_descs, disc_places, thresholds, use_threshold=True)
        all_results[f"{method} (threshold)"] = results_t
        print(f"\n  With threshold:")
        for k, v in results_t.items():
            print(f"    {k}: {v:.1f}%")

        # Without threshold
        results_b, scores_b, preds_b = evaluate(
            query_descs, ref_descs, disc_places, thresholds, use_threshold=False)
        all_results["baseline"] = results_b
        print(f"\n  Without threshold (baseline):")
        for k, v in results_b.items():
            print(f"    {k}: {v:.1f}%")

    # ── Also run with GT places for reference ────────────────────────
    print("\n" + "=" * 60)
    print("BONUS: Using GT places (paper's setup)")
    print("=" * 60)

    gt_places = {}
    for f_idx in range(N_ref):
        p = ref_gt[f_idx]
        if p not in gt_places:
            gt_places[p] = []
        gt_places[p].append(f_idx)
    gt_places_list = [gt_places[p] for p in sorted(gt_places.keys())]

    # Override correct places for GT evaluation
    query_correct_disc_places_backup = query_correct_disc_places.copy()
    for q_idx in range(N_query):
        gt_place = query_gt[q_idx]
        query_correct_disc_places[q_idx] = {gt_place}

    for method in ["simple_avg", "filter_n"]:
        thresholds_gt = compute_thresholds(ref_descs, gt_places_list, method=method)
        print(f"\n--- GT places, method: {method} ---")

        results_gt, _, _ = evaluate(
            query_descs, ref_descs, gt_places_list, thresholds_gt, use_threshold=True)
        all_results[f"GT {method} (threshold)"] = results_gt
        print(f"  With threshold:")
        for k, v in results_gt.items():
            print(f"    {k}: {v:.1f}%")

        results_gt_b, _, _ = evaluate(
            query_descs, ref_descs, gt_places_list, thresholds_gt, use_threshold=False)
        all_results["GT baseline"] = results_gt_b
        print(f"  Without threshold (baseline):")
        for k, v in results_gt_b.items():
            print(f"    {k}: {v:.1f}%")

    # Restore
    query_correct_disc_places = query_correct_disc_places_backup

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(f"{'Method':<40} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'Rej%':>6}")
    print(f"{'-' * 80}")

    # Paper results
    print(f"{'RCC Paper — Baseline':<40} {'61.67':>6} {'96.67':>6} {'100':>6} {'100':>6} {'—':>6}")
    print(f"{'RCC Paper — Simple Avg':<40} {'93.33':>6} {'98.33':>6} {'100':>6} {'100':>6} {'—':>6}")
    print(f"{'RCC Paper — Weighted Avg':<40} {'93.33':>6} {'98.33':>6} {'100':>6} {'100':>6} {'—':>6}")
    print(f"{'-' * 80}")

    for name, results in all_results.items():
        r1 = f"{results['Recall@1']:.1f}"
        r3 = f"{results['Recall@3']:.1f}"
        r5 = f"{results['Recall@5']:.1f}"
        r10 = f"{results['Recall@10']:.1f}"
        rej = f"{results['rejection_rate']:.1f}"
        print(f"{name:<40} {r1:>6} {r3:>6} {r5:>6} {r10:>6} {rej:>6}")

    print(f"{'=' * 80}")
    print(f"\nDiscovered places: {len(disc_places)} (GT has {n_gt_places})")
    print(f"Discovered sizes: {[len(p) for p in disc_places]}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
