"""
Vysotska validation on Bonn and Freiburg datasets.

These datasets use Vysotska's original format:
  - reference/images/ and query/images/ folders
  - Ground truth: gt_<name>.txt with format: queryId numMatches refId1 refId2 ...
  - Config: config.yaml with fanOut, nonMatchCost, etc.

We run:
  1. Our sequence matcher reimplementation with their exact parameters
  2. Our adaptive per-place threshold method
  3. Compare results to their reported numbers
"""

import os
import sys
import numpy as np
import pickle
from glob import glob
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import DatasetLoader
from experiments.vysotska_sequence_matcher import sequence_match
from experiments.vysotska_threshold import VysotskaDaptiveThreshold


# ── Dataset loading ──────────────────────────────────────────────────

def load_ground_truth(gt_path):
    """Load Vysotska ground truth format: queryId numMatches refId1 refId2 ..."""
    gt = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            q_id = int(parts[0])
            n_matches = int(parts[1])
            ref_ids = [int(x) for x in parts[2:2+n_matches]]
            gt[q_id] = ref_ids
    return gt


def load_images_sorted(image_dir, ext="*.jpg"):
    """Load image paths sorted by filename."""
    paths = sorted(glob(os.path.join(image_dir, ext)))
    return paths


def extract_or_load_descriptors(image_paths, cache_dir, descriptor_name, use_cache=True):
    """Extract descriptors for a list of images, with caching."""
    os.makedirs(cache_dir, exist_ok=True)
    descs = []
    to_extract = []

    for i, path in enumerate(image_paths):
        cache_path = os.path.join(cache_dir, f"img_{i}_descriptor.pkl")
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                d = pickle.load(f)
                if isinstance(d, dict):
                    d = d["descriptor"]
                descs.append((i, d))
        else:
            to_extract.append((i, path))

    if to_extract:
        print(f"  Extracting {len(to_extract)}/{len(image_paths)} descriptors...")
        loader = DatasetLoader.__new__(DatasetLoader)
        fe = loader._init_feature_extractor(descriptor_name)
        from PIL import Image
        batch_size = 32
        for start in range(0, len(to_extract), batch_size):
            batch = to_extract[start:start+batch_size]
            images = [np.array(Image.open(p)) for _, p in batch]
            feats = fe.compute_features(images)
            for j, (idx, _) in enumerate(batch):
                d = feats[j]
                cache_path = os.path.join(cache_dir, f"img_{idx}_descriptor.pkl")
                with open(cache_path, "wb") as f:
                    pickle.dump(d, f)
                descs.append((idx, d))
    else:
        print(f"  All {len(image_paths)} descriptors loaded from cache.")

    descs.sort(key=lambda x: x[0])
    return np.array([d for _, d in descs])


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_sequence_matcher_with_gt(matches, gt, tolerance=0):
    """Evaluate sequence matcher against many-to-many ground truth."""
    TP, FP, FN = 0, 0, 0

    all_query_ids = sorted(gt.keys())
    for q in all_query_ids:
        gt_refs = set(gt[q])
        if q in matches:
            pred = matches[q]
            # Check if prediction is within tolerance of any GT ref
            hit = any(abs(pred - r) <= tolerance for r in gt_refs)
            if hit:
                TP += 1
            else:
                FP += 1
        else:
            if len(gt_refs) > 0:
                FN += 1
            # If gt_refs is empty, query has no match — being hidden is correct (TN)

    # Queries with 0 GT matches that are hidden = TN, matched = FP
    for q in all_query_ids:
        if len(gt[q]) == 0 and q in matches:
            FP += 1  # already counted above? No — only if q not in gt check

    P = TP / max(TP + FP, 1) * 100
    R = TP / max(TP + FN, 1) * 100
    F1 = 2 * P * R / max(P + R, 1e-8)
    return {"P": P, "R": R, "F1": F1, "TP": TP, "FP": FP, "FN": FN,
            "n_matched": len(matches)}


# ── Main experiment ──────────────────────────────────────────────────

def run_validation(dataset_name, dataset_dir, descriptor_name="eigenplaces",
                   use_cache=True):
    """Run full validation on a Vysotska-format dataset."""
    print(f"\n{'='*70}")
    print(f"VYSOTSKA VALIDATION: {dataset_name}")
    print(f"{'='*70}")

    # Parse config
    import yaml
    config_path = os.path.join(dataset_dir, f"config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    fanout = int(config.get("fanOut", 5))
    nmc = float(config.get("nonMatchCost", 5.5))
    img_ext = config.get("imgExt", ".jpg")
    query_size = int(config.get("querySize", 0))

    print(f"Config: fanOut={fanout}, nonMatchCost={nmc}, imgExt={img_ext}, querySize={query_size}")

    # Load images
    ref_dir = os.path.join(dataset_dir, "reference", "images")
    query_dir = os.path.join(dataset_dir, "query", "images")
    ref_paths = load_images_sorted(ref_dir, f"*{img_ext}")
    query_paths = load_images_sorted(query_dir, f"*{img_ext}")
    print(f"Reference: {len(ref_paths)} images")
    print(f"Query: {len(query_paths)} images")

    # Load ground truth
    gt_files = glob(os.path.join(dataset_dir, "gt_*.txt"))
    gt = load_ground_truth(gt_files[0])
    n_with_match = sum(1 for refs in gt.values() if len(refs) > 0)
    n_no_match = sum(1 for refs in gt.values() if len(refs) == 0)
    print(f"Ground truth: {len(gt)} queries, {n_with_match} with matches, {n_no_match} without")

    # Extract descriptors
    cache_base = os.path.join("cache", dataset_name)
    print(f"\nExtracting/loading {descriptor_name} descriptors...")
    ref_descs = extract_or_load_descriptors(
        ref_paths, os.path.join(cache_base, "reference", descriptor_name),
        descriptor_name, use_cache
    )
    query_descs = extract_or_load_descriptors(
        query_paths, os.path.join(cache_base, "query", descriptor_name),
        descriptor_name, use_cache
    )
    print(f"  Reference: {ref_descs.shape}, Query: {query_descs.shape}")

    # Compute similarity matrix
    S = query_descs @ ref_descs.T
    print(f"Similarity matrix: {S.shape}, range [{S.min():.3f}, {S.max():.3f}]")

    results = {}

    # ── 1. Vysotska sequence matcher with their original parameters ──
    print(f"\n--- Vysotska Sequence Matcher (fanout={fanout}, nmc={nmc}) ---")
    # Their nmc is in cost space (1-sim), convert if needed
    # Their config uses OverFeat features where cost = Euclidean distance
    # With cosine similarity, cost = 1 - sim, so we need to convert nmc
    # Their nmc=6.0/5.5 is for OverFeat distances, not cosine.
    # We need to find a good nmc for cosine similarity.

    # Use their adaptive threshold to determine nmc for our descriptor
    vysotska = VysotskaDaptiveThreshold(patch_size=20)
    thresholds, _ = vysotska.compute_thresholds(S)
    median_thresh = float(np.median(thresholds))
    adapted_nmc = 1.0 - median_thresh
    print(f"  Adapted nmc from Vysotska threshold: {adapted_nmc:.4f} (median sim thresh={median_thresh:.4f})")

    for tol in [0, 1, 2]:
        matches, all_path, path_real, path_hidden = sequence_match(
            S, non_matching_cost=adapted_nmc, fanout=fanout
        )
        r = evaluate_sequence_matcher_with_gt(matches, gt, tolerance=tol)
        print(f"  tol=±{tol}: P={r['P']:.1f}% R={r['R']:.1f}% F1={r['F1']:.1f}% "
              f"(matched {r['n_matched']}/{len(gt)})")
        results[f"seq_match_tol{tol}"] = r

    # Also try different nmc values
    print(f"\n  NMC sweep:")
    for nmc_val in [0.1, 0.2, 0.3, 0.4, 0.5]:
        matches, _, _, _ = sequence_match(S, non_matching_cost=nmc_val, fanout=fanout)
        r = evaluate_sequence_matcher_with_gt(matches, gt, tolerance=1)
        print(f"    nmc={nmc_val:.1f}: P={r['P']:.1f}% R={r['R']:.1f}% F1={r['F1']:.1f}% "
              f"(matched {r['n_matched']}/{len(gt)})")

    # ── 2. Vysotska adaptive threshold (per-query) ──
    print(f"\n--- Vysotska Adaptive Threshold ---")
    TP, FP, FN = 0, 0, 0
    for q_id in sorted(gt.keys()):
        if q_id >= S.shape[0]:
            continue
        gt_refs = gt[q_id]
        best_ref = np.argmax(S[q_id])
        sim = S[q_id, best_ref]

        if q_id < len(thresholds):
            thresh = thresholds[q_id]
        else:
            thresh = median_thresh

        if sim >= thresh:
            hit = any(abs(best_ref - r) <= 1 for r in gt_refs) if gt_refs else False
            if hit:
                TP += 1
            else:
                FP += 1
        else:
            if gt_refs:
                FN += 1

    P = TP / max(TP + FP, 1) * 100
    R = TP / max(TP + FN, 1) * 100
    F1 = 2 * P * R / max(P + R, 1e-8)
    print(f"  tol=±1: P={P:.1f}% R={R:.1f}% F1={F1:.1f}%")
    results["vysotska_thresh"] = {"P": P, "R": R, "F1": F1, "TP": TP, "FP": FP, "FN": FN}

    # ── 3. Our method: per-place threshold with online place discovery ──
    print(f"\n--- Our Method (per-place threshold, k=2) ---")
    from experiments.online_place_discovery import OnlinePlaceDiscovery

    discoverer = OnlinePlaceDiscovery(
        bootstrap_std_factor=1.5,
        min_place_size=3,
        hysteresis=2,
    )
    for i in range(len(ref_descs)):
        discoverer.process_frame(ref_descs[i], i, verbose=False)
    places = discoverer.places
    sizes = [len(p) for p in places]
    print(f"  Discovered {len(places)} places (sizes: {min(sizes)}-{max(sizes)}, mean={np.mean(sizes):.1f})")

    # Compute per-place thresholds
    ref_S = ref_descs @ ref_descs.T
    k = 2
    place_thresholds = []
    for pidx, place in enumerate(places):
        neg_sims = []
        for i in place:
            for other_pidx, other_place in enumerate(places):
                if other_pidx == pidx:
                    continue
                for j in other_place:
                    neg_sims.append(ref_S[i, j])
        neg_sims = np.array(neg_sims)
        mu_neg = neg_sims.mean()
        std_neg = neg_sims.std()
        thresh = mu_neg + k * std_neg
        place_thresholds.append(thresh)

    # Evaluate
    TP, FP, FN = 0, 0, 0
    for q_id in sorted(gt.keys()):
        if q_id >= S.shape[0]:
            continue
        gt_refs = gt[q_id]

        # Compute per-place scores
        place_scores = []
        for pidx, place in enumerate(places):
            score = np.mean([S[q_id, j] for j in place])
            place_scores.append(score)

        # Filter: keep places above threshold
        surviving = [(pidx, score) for pidx, score in enumerate(place_scores)
                     if score >= place_thresholds[pidx]]

        if not surviving:
            if gt_refs:
                FN += 1
            continue

        # Rank: pick best surviving place
        best_pidx, best_score = max(surviving, key=lambda x: x[1])
        best_place = places[best_pidx]

        # Find closest ref in that place to the query
        best_ref = max(best_place, key=lambda j: S[q_id, j])

        hit = any(abs(best_ref - r) <= 1 for r in gt_refs) if gt_refs else False
        if hit:
            TP += 1
        else:
            FP += 1

    P = TP / max(TP + FP, 1) * 100
    R = TP / max(TP + FN, 1) * 100
    F1 = 2 * P * R / max(P + R, 1e-8)
    print(f"  tol=±1: P={P:.1f}% R={R:.1f}% F1={F1:.1f}%")
    results["ours_k2"] = {"P": P, "R": R, "F1": F1, "TP": TP, "FP": FP, "FN": FN}

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY: {dataset_name}")
    print(f"{'='*70}")
    print(f"  {'Method':<30} {'P':>6} {'R':>6} {'F1':>6}")
    print(f"  {'─'*50}")
    sm = results.get("seq_match_tol1", {})
    vt = results.get("vysotska_thresh", {})
    ours = results.get("ours_k2", {})
    print(f"  {'Vysotska seq.match (tol=1)':<30} {sm.get('P',0):5.1f}% {sm.get('R',0):5.1f}% {sm.get('F1',0):5.1f}%")
    print(f"  {'Vysotska thresh (tol=1)':<30} {vt.get('P',0):5.1f}% {vt.get('R',0):5.1f}% {vt.get('F1',0):5.1f}%")
    print(f"  {'Ours k=2 (tol=1)':<30} {ours.get('P',0):5.1f}% {ours.get('R',0):5.1f}% {ours.get('F1',0):5.1f}%")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", default="eigenplaces", choices=["eigenplaces", "dinov2-salad"])
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    results = {}
    for name, dirname in [("Freiburg", "freiburg_example"), ("Bonn", "bonn_example")]:
        dpath = os.path.join(base, dirname)
        if os.path.exists(dpath):
            results[name] = run_validation(name, dpath, args.descriptor, not args.no_cache)
        else:
            print(f"Dataset {name} not found at {dpath}")

    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    for name, r in results.items():
        print(f"\n{name}:")
        sm = r.get("seq_match_tol1", {})
        vt = r.get("vysotska_thresh", {})
        ours = r.get("ours_k2", {})
        print(f"  Vysotska seq.match: F1={sm.get('F1',0):.1f}%")
        print(f"  Vysotska thresh:    F1={vt.get('F1',0):.1f}%")
        print(f"  Ours (k=2):         F1={ours.get('F1',0):.1f}%")
