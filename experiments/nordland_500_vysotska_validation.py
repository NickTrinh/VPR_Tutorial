"""
Nordland-500 Vysotska Validation.

Replicates Vysotska et al. (ICRA 2025) Table I setup:
  - First 500 Nordland images
  - Winter = reference, Summer = query
  - 1:1 correspondence (image i in query matches image i in reference)

Their reported F1 = 0.98 on this dataset.

We run both:
  1. Our Vysotska reimplementation — should match their F1=0.98
  2. Our filter_n method — head-to-head comparison

Usage:
    python -m experiments.nordland_500_vysotska_validation
    python -m experiments.nordland_500_vysotska_validation --descriptor dinov2_salad
"""

import argparse
import os
import sys
import json
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from experiments.vysotska_threshold import VysotskaDaptiveThreshold
from experiments.vysotska_sequence_matcher import sequence_match, evaluate_sequence_match
from utils import normalize_l2


# ── Config ────────────────────────────────────────────────────────────────────
N = 500  # Vysotska used first 500 images

DESCRIPTOR_CONFIGS = {
    "eigenplaces": {
        "ref_cache": "cache/Nordland_filtered/winter/eigenplaces",
        "query_cache": "cache/Nordland_filtered/summer/eigenplaces",
        "dim": 2048,
        "name": "EigenPlaces (ResNet50, 2048-dim)",
    },
    "dinov2_salad": {
        "ref_cache": "cache/Nordland_salad/winter/dinov2_salad",
        "query_cache": "cache/Nordland_salad/summer/dinov2_salad",
        "dim": 8448,
        "name": "DINOv2 SALAD (ViT-B/14, 8448-dim)",
    },
}

OUTPUT_DIR = "results/nordland_500_validation"


def load_descriptors(cache_dir, n):
    """Load first n descriptors from cache."""
    descs = []
    for i in range(n):
        path = os.path.join(cache_dir, f"img_{i}_descriptor.pkl")
        with open(path, "rb") as f:
            d = pickle.load(f)
            if isinstance(d, dict):
                d = d["descriptor"]
            descs.append(d.reshape(-1))
    arr = np.array(descs, dtype=np.float32)
    return normalize_l2(arr)


def discover_places(ref_descs, min_place_size=3, hysteresis=2, filter_n_cap=10):
    """Online place discovery."""
    from experiments.online_place_discovery import OnlinePlaceDiscovery
    discoverer = OnlinePlaceDiscovery(
        min_place_size=min_place_size,
        hysteresis=hysteresis,
        filter_n_cap=filter_n_cap
    )
    for i in range(len(ref_descs)):
        discoverer.process_frame(ref_descs[i], i, verbose=False)
    return discoverer.places


def compute_thresholds(ref_descs, places, method="filter_n", cap=10):
    """Per-place thresholds from negative statistics."""
    thresholds = {}
    for p_idx in range(len(places)):
        target = places[p_idx]
        other = [f for i, p in enumerate(places) if i != p_idx for f in p]
        if not other or len(target) < 2:
            thresholds[p_idx] = 0.0
            continue
        neg_sims = ref_descs[target] @ ref_descs[other].T
        per_img = neg_sims.mean(axis=1)
        mean_bad = float(per_img.mean())
        std_bad = float(per_img.std()) if len(per_img) > 1 else 0.1
        pos_sims = ref_descs[target] @ ref_descs[target].T
        np.fill_diagonal(pos_sims, 0)
        mean_good = float(pos_sims.sum(axis=1).mean() / max(len(target) - 1, 1))
        fn = max(0, min(np.floor((mean_good - mean_bad) / max(std_bad, 1e-8)), cap))
        if method == "filter_n":
            thresholds[p_idx] = mean_bad + fn * std_bad
        else:
            thresholds[p_idx] = mean_bad
    return thresholds


def evaluate_sequence_matching(query_descs, ref_descs, thresholds_per_query=None,
                                threshold_scalar=None, tolerance=0, S=None):
    """
    Vysotska-style sequence matching evaluation.

    In Vysotska's setup, the ground truth is 1:1 correspondence:
    query image i matches reference image i.

    For each query, find the best-matching reference image.
    A match is correct if |predicted - ground_truth| <= tolerance.
    Apply threshold to decide accept/reject.

    Returns: precision, recall, F1, and per-query details.
    """
    if S is None:
        S = query_descs @ ref_descs.T
    n = len(query_descs)

    best_matches = np.argmax(S, axis=1)
    best_scores = np.max(S, axis=1)

    TP = 0  # correctly matched
    FP = 0  # matched but wrong
    FN = 0  # should have matched but rejected (or matched wrong)

    correct = 0
    accepted = 0
    details = []

    for q in range(n):
        pred_ref = best_matches[q]
        score = best_scores[q]
        gt_ref = q  # 1:1 correspondence

        # Determine threshold for this query
        if thresholds_per_query is not None:
            thresh = thresholds_per_query[q]
        elif threshold_scalar is not None:
            thresh = threshold_scalar
        else:
            thresh = -np.inf  # no threshold

        accepted_q = score >= thresh
        correct_q = abs(int(pred_ref) - gt_ref) <= tolerance

        if accepted_q:
            accepted += 1
            if correct_q:
                TP += 1
                correct += 1
            else:
                FP += 1
        else:
            # Rejected — this is a false negative (query had a match but we rejected)
            FN += 1

        details.append({
            "query": q,
            "pred_ref": int(pred_ref),
            "gt_ref": gt_ref,
            "score": float(score),
            "threshold": float(thresh),
            "accepted": bool(accepted_q),
            "correct": bool(correct_q),
        })

    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": TP, "FP": FP, "FN": FN,
        "accepted": accepted,
        "correct": correct,
        "total": n,
    }, details


def evaluate_with_places(query_descs, ref_descs, places, thresholds):
    """
    Our place-based evaluation adapted for Nordland 1:1 ground truth.

    For each query q, ground truth place is the place containing frame q
    (since query i matches reference i).
    """
    S = query_descs @ ref_descs.T
    n_query = len(query_descs)
    n_places = len(places)

    # Build frame-to-place mapping
    frame_to_place = {}
    for p_idx, frames in enumerate(places):
        for f in frames:
            frame_to_place[f] = p_idx

    # Compute per-place scores
    place_scores = np.zeros((n_query, n_places))
    for p_idx, frames in enumerate(places):
        place_scores[:, p_idx] = S[:, frames].mean(axis=1)

    TP, FP, FN = 0, 0, 0
    for q in range(n_query):
        gt_place = frame_to_place.get(q, -1)
        if gt_place == -1:
            # Query index not in any discovered place — skip
            FN += 1
            continue

        scores = place_scores[q].copy()
        for p in range(n_places):
            if scores[p] < thresholds.get(p, -np.inf):
                scores[p] = -np.inf

        pred = -1 if np.all(scores == -np.inf) else int(np.argmax(scores))

        if pred == gt_place:
            TP += 1
        elif pred == -1:
            FN += 1
        else:
            FP += 1

    P = TP / max(TP + FP, 1)
    R = TP / max(TP + FN, 1)
    F1 = 2 * P * R / max(P + R, 1e-8)
    return {"P": P, "R": R, "F1": F1, "TP": TP, "FP": FP, "FN": FN}


def parse_args():
    p = argparse.ArgumentParser(description="Nordland-500 Vysotska validation")
    p.add_argument("--descriptor", default="eigenplaces",
                   choices=list(DESCRIPTOR_CONFIGS.keys()),
                   help="Descriptor to use (default: eigenplaces)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = DESCRIPTOR_CONFIGS[args.descriptor]
    REF_CACHE = cfg["ref_cache"]
    QUERY_CACHE = cfg["query_cache"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"{'='*60}")
    print(f"NORDLAND-500 VYSOTSKA VALIDATION")
    print(f"{'='*60}")
    print(f"Descriptor: {cfg['name']}")
    print(f"Setup: First {N} images, winter=ref, summer=query")
    print(f"Ground truth: 1:1 correspondence (query i = ref i)")
    print(f"Vysotska reported F1 = 0.98 on this dataset")
    print()

    # ── Load descriptors ──────────────────────────────────────────────
    print("Loading descriptors...")
    ref_descs = load_descriptors(REF_CACHE, N)
    query_descs = load_descriptors(QUERY_CACHE, N)
    print(f"  Reference: {ref_descs.shape}")
    print(f"  Query:     {query_descs.shape}")

    # ── Compute similarity matrix ─────────────────────────────────────
    S = query_descs @ ref_descs.T
    print(f"\nSimilarity matrix: {S.shape}")
    print(f"  Mean: {S.mean():.4f}")
    print(f"  Diagonal mean (correct matches): {np.diag(S).mean():.4f}")
    print(f"  Off-diagonal mean: {(S.sum() - np.trace(S)) / (N*N - N):.4f}")

    # ── Baseline: no threshold ────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("1. BASELINE (no threshold)")
    result_baseline, _ = evaluate_sequence_matching(query_descs, ref_descs, S=S)
    print(f"   Precision: {result_baseline['precision']:.4f}")
    print(f"   Recall:    {result_baseline['recall']:.4f}")
    print(f"   F1:        {result_baseline['f1']:.4f}")
    print(f"   Correct:   {result_baseline['correct']}/{result_baseline['total']}")

    # ── Vysotska adaptive thresholding ────────────────────────────────
    print(f"\n{'─'*60}")
    print("2. VYSOTSKA ADAPTIVE THRESHOLDING (patch=20, KS test + GMM + KF)")
    vysotska = VysotskaDaptiveThreshold(patch_size=20)
    vysotska_thresholds, vysotska_history = vysotska.compute_thresholds(S)

    result_vysotska, details_v = evaluate_sequence_matching(
        query_descs, ref_descs, thresholds_per_query=vysotska_thresholds, S=S
    )
    print(f"   Precision: {result_vysotska['precision']:.4f}")
    print(f"   Recall:    {result_vysotska['recall']:.4f}")
    print(f"   F1:        {result_vysotska['f1']:.4f}")
    print(f"   Accepted:  {result_vysotska['accepted']}/{result_vysotska['total']}")
    print(f"   Correct:   {result_vysotska['correct']}/{result_vysotska['total']}")
    print(f"   Target F1: 0.98 (Vysotska Table I)")

    n_bimodal = sum(1 for h in vysotska_history if h.get("is_bimodal", False))
    n_skipped = sum(1 for h in vysotska_history if h.get("skipped", False))
    print(f"   Patches bimodal: {n_bimodal}/{N}")
    print(f"   Patches skipped: {n_skipped}/{N}")

    # ── Vysotska sequence matcher (graph-based shortest path) ────────
    print(f"\n{'─'*60}")
    print("2b. VYSOTSKA SEQUENCE MATCHER (graph shortest path)")
    print("    This is the missing piece — their full pipeline uses a DAG")
    print("    shortest path through the similarity matrix, not just argmax.")

    # Try different non_matching_cost thresholds
    # In cost space: non_matching_cost = 1 - sim_threshold
    # Their default seems to be around 0.5 (cost), i.e., sim > 0.5 = real match
    for nmc_label, nmc in [("0.3 (sim>0.7)", 0.3),
                            ("0.4 (sim>0.6)", 0.4),
                            ("0.5 (sim>0.5)", 0.5),
                            ("0.6 (sim>0.4)", 0.6)]:
        matches, all_path, path_real, path_hidden = sequence_match(
            S, non_matching_cost=nmc, fanout=3
        )
        for tol in [0, 1, 2, 5]:
            result_sm = evaluate_sequence_match(S, matches, tolerance=tol)
            if tol == 0:
                print(f"\n   nmc={nmc_label}, fanout=3:")
                print(f"     Path length: {len(all_path)}, "
                      f"real={len(path_real)}, hidden={len(path_hidden)}")
            print(f"     tol=±{tol}: P={result_sm['precision']:.4f} "
                  f"R={result_sm['recall']:.4f} F1={result_sm['f1']:.4f} "
                  f"matched={result_sm['n_matched']}/{result_sm['n_total']}")

    # Best config: try fanout values too
    print(f"\n   --- Fanout sweep (nmc=0.5) ---")
    for fanout in [1, 2, 3, 5, 8]:
        matches, all_path, path_real, path_hidden = sequence_match(
            S, non_matching_cost=0.5, fanout=fanout
        )
        r = evaluate_sequence_match(S, matches, tolerance=0)
        print(f"   fanout={fanout}: P={r['precision']:.4f} R={r['recall']:.4f} "
              f"F1={r['f1']:.4f} matched={r['n_matched']}/{r['n_total']}")

    # Run with Vysotska adaptive thresholds as non_matching_cost
    # Use median of Vysotska thresholds converted to cost
    median_thresh = float(np.median(vysotska_thresholds))
    median_cost = 1.0 - median_thresh
    print(f"\n   --- With Vysotska adaptive threshold (median sim={median_thresh:.3f}, "
          f"cost={median_cost:.3f}) ---")
    matches_va, all_path_va, path_real_va, path_hidden_va = sequence_match(
        S, non_matching_cost=median_cost, fanout=3
    )
    for tol in [0, 1, 2, 5]:
        r = evaluate_sequence_match(S, matches_va, tolerance=tol)
        print(f"   tol=±{tol}: P={r['precision']:.4f} R={r['recall']:.4f} "
              f"F1={r['f1']:.4f} matched={r['n_matched']}/{r['n_total']}")

    # ── Our method: discover places + filter_n ────────────────────────
    print(f"\n{'─'*60}")
    print("3. OUR METHOD: ONLINE DISCOVERY + FILTER_N")
    places = discover_places(ref_descs)
    print(f"   Discovered {len(places)} places")
    sizes = [len(p) for p in places]
    print(f"   Place sizes: min={min(sizes)}, max={max(sizes)}, "
          f"mean={np.mean(sizes):.1f}")

    # Check coverage: how many of the 500 frames are in discovered places?
    covered = sum(len(p) for p in places)
    print(f"   Coverage: {covered}/{N} frames in places")

    for method in ["filter_n", "mean_bad"]:
        thresholds = compute_thresholds(ref_descs, places, method=method)
        result = evaluate_with_places(query_descs, ref_descs, places, thresholds)
        print(f"\n   {method}:")
        print(f"     Precision: {result['P']:.4f}")
        print(f"     Recall:    {result['R']:.4f}")
        print(f"     F1:        {result['F1']:.4f}")
        print(f"     TP={result['TP']} FP={result['FP']} FN={result['FN']}")

    # No-threshold baseline with places
    no_thresh = {p: -np.inf for p in range(len(places))}
    result_no = evaluate_with_places(query_descs, ref_descs, places, no_thresh)
    print(f"\n   baseline (no threshold):")
    print(f"     Precision: {result_no['P']:.4f}")
    print(f"     Recall:    {result_no['R']:.4f}")
    print(f"     F1:        {result_no['F1']:.4f}")

    # ── Tolerance analysis ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("4. TOLERANCE ANALYSIS (how many correct within ±k frames?)")
    for tol in [0, 1, 2, 5, 10, 20]:
        r, _ = evaluate_sequence_matching(
            query_descs, ref_descs, tolerance=tol, S=S
        )
        print(f"   tol=±{tol:2d}: P={r['precision']:.4f} R={r['recall']:.4f} "
              f"F1={r['f1']:.4f} correct={r['correct']}/{N}")

    # ── Vysotska with tolerance ───────────────────────────────────────
    print(f"\n{'─'*60}")
    print("5. VYSOTSKA + TOLERANCE (does tolerance explain F1 gap?)")
    for tol in [0, 1, 2, 5, 10, 20]:
        r, _ = evaluate_sequence_matching(
            query_descs, ref_descs, thresholds_per_query=vysotska_thresholds,
            tolerance=tol, S=S
        )
        print(f"   tol=±{tol:2d}: P={r['precision']:.4f} R={r['recall']:.4f} "
              f"F1={r['f1']:.4f} accepted={r['accepted']}/{N}")

    # ── Error analysis ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("6. ERROR ANALYSIS (where do wrong matches go?)")
    errors = []
    for q in range(N):
        pred = int(np.argmax(S[q]))
        if pred != q:
            errors.append((q, pred, abs(pred - q), float(S[q, pred]), float(S[q, q])))
    errors.sort(key=lambda x: -x[2])  # sort by distance
    print(f"   Total wrong (tol=0): {len(errors)}/{N}")
    print(f"   Within ±5 frames: {sum(1 for e in errors if e[2] <= 5)}")
    print(f"   Within ±10 frames: {sum(1 for e in errors if e[2] <= 10)}")
    print(f"   Within ±20 frames: {sum(1 for e in errors if e[2] <= 20)}")
    print(f"   Beyond ±20 frames: {sum(1 for e in errors if e[2] > 20)}")
    print(f"\n   Worst 10 errors (query, pred, dist, pred_score, gt_score):")
    for e in errors[:10]:
        print(f"     q={e[0]:3d} pred={e[1]:3d} dist={e[2]:3d} "
              f"pred_sim={e[3]:.4f} gt_sim={e[4]:.4f}")

    # ── Oracle threshold sweep ────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("7. ORACLE THRESHOLD SWEEP (best possible fixed threshold)")
    best_f1 = 0
    best_t = 0
    for t in np.arange(0.0, 1.0, 0.01):
        r, _ = evaluate_sequence_matching(
            query_descs, ref_descs, threshold_scalar=t, S=S
        )
        if r['f1'] > best_f1:
            best_f1 = r['f1']
            best_t = t
    print(f"   Best fixed threshold: {best_t:.2f}")
    r_best, _ = evaluate_sequence_matching(
        query_descs, ref_descs, threshold_scalar=best_t, S=S
    )
    print(f"   P={r_best['precision']:.4f} R={r_best['recall']:.4f} "
          f"F1={r_best['f1']:.4f}")

    # ── Save results ──────────────────────────────────────────────────
    # Recompute key results with best tolerance
    r_v_tol5, _ = evaluate_sequence_matching(
        query_descs, ref_descs, thresholds_per_query=vysotska_thresholds,
        tolerance=5, S=S
    )
    r_b_tol5, _ = evaluate_sequence_matching(
        query_descs, ref_descs, tolerance=5, S=S
    )
    # Best sequence matcher result for summary
    matches_best, _, _, _ = sequence_match(S, non_matching_cost=0.5, fanout=3)
    r_seqmatch = evaluate_sequence_match(S, matches_best, tolerance=0)

    results = {
        "dataset": "Nordland-500",
        "n_images": N,
        "ref_condition": "winter",
        "query_condition": "summer",
        "descriptor": cfg['name'],
        "baseline_tol0": result_baseline,
        "baseline_tol5": r_b_tol5,
        "vysotska_tol0": result_vysotska,
        "vysotska_tol5": r_v_tol5,
        "sequence_matcher_tol0": r_seqmatch,
        "vysotska_target_f1": 0.98,
        "oracle_threshold": best_t,
        "oracle_f1": best_f1,
        "n_places_discovered": len(places),
        "place_sizes": sizes,
    }

    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Vysotska (their paper):     F1 = 0.98")
    print(f"  Vysotska (our reimpl):      F1 = {result_vysotska['f1']:.4f}")
    print(f"  + Sequence matcher:         F1 = {r_seqmatch['f1']:.4f}")
    print(f"  Baseline (no threshold):    F1 = {result_baseline['f1']:.4f}")
    print(f"  Oracle (best fixed thresh): F1 = {best_f1:.4f} (θ={best_t:.2f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
