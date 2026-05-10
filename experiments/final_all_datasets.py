"""
Final consolidated experiment: all 6 datasets, DINOv2 SALAD, closed-set + natural open-set.
Produces the numbers reported in the IEEE RAL submission.

Datasets (all tol=±2, ref_fraction=0.7 for open-set):
  - Nordland-500: 500 ref (winter) / 500 query (summer), 1-to-1
  - Bonn:         488 ref / 544 query, Vysotska GT format
  - Freiburg:     361 ref / 676 query, Vysotska GT format
  - GardensPoint: 200 ref (day_left) / 200 query (day_right), 1-to-1
  - SFU:          385 ref (dry) / 385 query (jan), GT.npz
  - ESSEX3IN1:    210 ref / 210 query, 1-to-1

Methods:
  - Baseline (no threshold)
  - Ours: online place discovery (m=2, h=2, α=1.5) + continuous adaptive k = clip(sep/2, 1, 2)
  - Vysotska: GMM+KS+Kalman threshold + graph sequence matcher (fanout=5)

Output: results/final_all_datasets_dinov2salad.json
"""

import os
import sys
import re
import json
import pickle
import numpy as np
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.online_place_discovery import OnlinePlaceDiscovery
from experiments.vysotska_sequence_matcher import sequence_match
from experiments.vysotska_threshold import VysotskaDaptiveThreshold


# ── Helpers ──────────────────────────────────────────────────────────

def load_descs_pkl(cache_dir):
    """Load descriptors from pickle cache, sorted numerically."""
    def nkey(p):
        m = re.search(r'img_(\d+)_', os.path.basename(p))
        return int(m.group(1)) if m else 0
    files = sorted(glob(os.path.join(cache_dir, "*.pkl")), key=nkey)
    descs = []
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
            if isinstance(d, dict):
                d = d["descriptor"]
            descs.append(d)
    return np.array(descs)


def load_vysotska_gt(gt_path):
    """Load Vysotska-format GT: queryId numMatches refId1 refId2 ..."""
    gt = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            q = int(parts[0])
            n = int(parts[1])
            gt[q] = [int(x) for x in parts[2:2+n]]
    return gt


def discover_places(ref_descs, min_place_size=2):
    """Run online place discovery on reference descriptors."""
    discoverer = OnlinePlaceDiscovery(
        bootstrap_std_factor=1.5,
        min_place_size=min_place_size,
        hysteresis=2,
    )
    for i in range(len(ref_descs)):
        discoverer.process_frame(ref_descs[i], i, verbose=False)
    return discoverer.places


def compute_adaptive_thresholds(ref_descs, places):
    """Per-place thresholds with continuous adaptive k = clip(sep/2, 1, 2)."""
    ref_S = ref_descs @ ref_descs.T
    thresholds = []
    ks = []
    for pidx, place in enumerate(places):
        pos_sims = []
        neg_sims = []
        for i in place:
            for j in place:
                if i != j:
                    pos_sims.append(ref_S[i, j])
            for op, other in enumerate(places):
                if op == pidx:
                    continue
                for j in other:
                    neg_sims.append(ref_S[i, j])
        pos_sims = np.array(pos_sims) if pos_sims else np.array([0.5])
        neg_sims = np.array(neg_sims)
        mu_pos = pos_sims.mean()
        mu_neg = neg_sims.mean()
        sigma_neg = neg_sims.std()
        sep = (mu_pos - mu_neg) / max(sigma_neg, 1e-8)
        k = max(1.0, min(sep / 2.0, 2.0))
        thresholds.append(mu_neg + k * sigma_neg)
        ks.append(k)
    return thresholds, ks


def query_our_method(S, places, thresholds):
    """Run filter-then-rank for all queries. Returns {query_idx: best_ref_idx}."""
    accepted = {}
    for q in range(S.shape[0]):
        scores = [np.mean([S[q, j] for j in place]) for place in places]
        surviving = [(pi, sc) for pi, sc in enumerate(scores) if sc >= thresholds[pi]]
        if not surviving:
            continue
        best_pi = max(surviving, key=lambda x: x[1])[0]
        best_ref = max(places[best_pi], key=lambda j: S[q, j])
        accepted[q] = best_ref
    return accepted


def run_vysotska(S, fanout=5):
    """Run Vysotska sequence matcher. Returns {query_idx: ref_idx}."""
    vysotska_thresh = VysotskaDaptiveThreshold(patch_size=20)
    thresholds, _ = vysotska_thresh.compute_thresholds(S)
    median_thresh = float(np.median(thresholds))
    nmc = 1.0 - median_thresh
    matches, _, _, _ = sequence_match(S, non_matching_cost=nmc, fanout=fanout)
    return {q: r for q, r in matches.items() if r is not None}


def run_vysotska_threshold_only(S):
    """Vysotska's per-query adaptive threshold WITHOUT the sequence matcher.

    For each query: accept argmax(S[q]) if S[q, argmax] >= threshold[q],
    else reject as unknown. This isolates the thresholding component for
    apples-to-apples comparison with our per-place thresholds.
    """
    vt = VysotskaDaptiveThreshold(patch_size=20)
    best = np.argmax(S, axis=1)
    thresholds, _ = vt.compute_thresholds(S, best_matches=best)
    accepted = {}
    for q in range(S.shape[0]):
        ref_idx = int(best[q])
        if S[q, ref_idx] >= thresholds[q]:
            accepted[q] = ref_idx
    return accepted


def evaluate(predictions, genuine_queries, distractor_queries, gt_map, tolerance):
    """
    Evaluate predictions.
    gt_map: {query_idx: true_ref_idx} or {query_idx: [list of valid refs]}
    """
    TP = FP = FN = TN = 0
    dist_rej = 0

    for q in genuine_queries:
        gt = gt_map.get(q)
        if gt is None:
            continue
        if q in predictions:
            pred = predictions[q]
            if isinstance(gt, list):
                hit = any(abs(pred - r) <= tolerance for r in gt)
            else:
                hit = abs(pred - gt) <= tolerance
            if hit:
                TP += 1
            else:
                FP += 1
        else:
            FN += 1

    for q in distractor_queries:
        if q in predictions:
            FP += 1
        else:
            TN += 1
            dist_rej += 1

    n_dist = len(distractor_queries)
    P = TP / max(TP + FP, 1) * 100
    R = TP / max(TP + FN, 1) * 100
    F1 = 2 * P * R / max(P + R, 1e-8)
    rej = dist_rej / max(n_dist, 1) * 100 if n_dist > 0 else None

    return {
        "P": round(P, 2), "R": round(R, 2), "F1": round(F1, 2),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "rejection": round(rej, 2) if rej is not None else None,
        "n_distractor": n_dist
    }


# ── Dataset configs ──────────────────────────────────────────────────

def load_sequential_dataset(name, ref_cache, query_cache, n_ref, n_query, tolerance):
    """Load a sequential dataset with 1-to-1 GT matching."""
    ref_descs = load_descs_pkl(ref_cache)[:n_ref]
    query_descs = load_descs_pkl(query_cache)[:n_query]
    n = min(n_ref, n_query)
    gt_map = {i: i for i in range(n)}  # 1-to-1
    return ref_descs, query_descs, gt_map, tolerance


def load_vysotska_dataset(name, dataset_dir, ref_cache, query_cache, tolerance):
    """Load a Vysotska-format dataset with GT file."""
    ref_descs = load_descs_pkl(ref_cache)
    query_descs = load_descs_pkl(query_cache)
    gt_files = glob(os.path.join(dataset_dir, "gt_*.txt"))
    gt_raw = load_vysotska_gt(gt_files[0])
    # gt_map: query -> list of valid refs
    gt_map = {q: refs for q, refs in gt_raw.items() if refs}
    return ref_descs, query_descs, gt_map, tolerance


def load_sfu_dataset():
    """Load SFU with GT.npz matching matrix."""
    ref_descs = load_descs_pkl("cache/SFU/dry/dinov2-salad")
    query_descs = load_descs_pkl("cache/SFU/jan/dinov2-salad")
    gt_data = np.load("images/SFU/GT.npz")
    G = gt_data[list(gt_data.keys())[0]]
    # Build GT map from matrix
    gt_map = {}
    for q in range(G.shape[0]):
        refs = list(np.where(G[q] > 0)[0])
        if refs:
            gt_map[q] = refs
    return ref_descs, query_descs, gt_map, 1


# ── Main pipeline ────────────────────────────────────────────────────

def run_dataset(name, ref_descs, query_descs, gt_map, tolerance,
                ref_fraction=0.7, min_place_size=2, fanout=5):
    """Run closed-set and natural open-set on one dataset."""
    n_ref = len(ref_descs)
    n_query = len(query_descs)

    print(f"\n{'='*70}")
    print(f"DATASET: {name}")
    print(f"  Ref: {n_ref}, Query: {n_query}, Tolerance: ±{tolerance}")
    print(f"{'='*70}")

    results = {}

    # ── CLOSED-SET ──
    print(f"\n--- CLOSED-SET ---")
    S_closed = query_descs @ ref_descs.T

    # All queries with GT are genuine, no distractors
    genuine_closed = sorted(gt_map.keys())
    print(f"  Genuine queries (with GT): {len(genuine_closed)}")

    # Baseline
    bl_pred = {q: int(np.argmax(S_closed[q])) for q in range(n_query)}
    r = evaluate(bl_pred, genuine_closed, [], gt_map, tolerance)
    print(f"  Baseline:     P={r['P']:.1f}% R={r['R']:.1f}% F1={r['F1']:.1f}%")
    results["closed_Baseline"] = r

    # Our method
    places = discover_places(ref_descs, min_place_size=min_place_size)
    sizes = [len(p) for p in places]
    print(f"  Discovered {len(places)} places (sizes: {min(sizes)}-{max(sizes)}, mean={np.mean(sizes):.1f})")
    thresholds, ks = compute_adaptive_thresholds(ref_descs, places)
    print(f"  Adaptive k: min={min(ks):.2f} mean={np.mean(ks):.2f} max={max(ks):.2f}")

    ours_pred = query_our_method(S_closed, places, thresholds)
    r = evaluate(ours_pred, genuine_closed, [], gt_map, tolerance)
    print(f"  Ours:         P={r['P']:.1f}% R={r['R']:.1f}% F1={r['F1']:.1f}%")
    results["closed_Ours"] = r

    # Vysotska threshold only (no sequence matcher)
    vt_pred = run_vysotska_threshold_only(S_closed)
    r = evaluate(vt_pred, genuine_closed, [], gt_map, tolerance)
    print(f"  Vys (thresh): P={r['P']:.1f}% R={r['R']:.1f}% F1={r['F1']:.1f}%")
    results["closed_VysotskaThresh"] = r

    # Vysotska full pipeline
    vys_pred = run_vysotska(S_closed, fanout=fanout)
    r = evaluate(vys_pred, genuine_closed, [], gt_map, tolerance)
    print(f"  Vys (full):   P={r['P']:.1f}% R={r['R']:.1f}% F1={r['F1']:.1f}%")
    results["closed_Vysotska"] = r

    # ── NATURAL OPEN-SET ──
    print(f"\n--- NATURAL OPEN-SET ({ref_fraction*100:.0f}% ref) ---")
    ref_cutoff = int(n_ref * ref_fraction)
    ref_trunc = ref_descs[:ref_cutoff]
    S_nat = query_descs @ ref_trunc.T

    # Classify queries
    genuine_nat = []
    distractor_nat = []
    gt_map_trunc = {}

    for q in range(n_query):
        gt = gt_map.get(q)
        if gt is None:
            distractor_nat.append(q)
            continue
        if isinstance(gt, list):
            refs_in_range = [r for r in gt if r < ref_cutoff]
            if refs_in_range:
                genuine_nat.append(q)
                gt_map_trunc[q] = refs_in_range
            else:
                distractor_nat.append(q)
        else:
            if gt < ref_cutoff:
                genuine_nat.append(q)
                gt_map_trunc[q] = gt
            else:
                distractor_nat.append(q)

    print(f"  Ref: {ref_cutoff}/{n_ref}, Genuine: {len(genuine_nat)}, Distractor: {len(distractor_nat)}")

    # Baseline
    bl_pred_nat = {q: int(np.argmax(S_nat[q])) for q in range(n_query)}
    r = evaluate(bl_pred_nat, genuine_nat, distractor_nat, gt_map_trunc, tolerance)
    print(f"  Baseline:     P={r['P']:.1f}% R={r['R']:.1f}% F1={r['F1']:.1f}% Rej={r['rejection']:.1f}%")
    results["natural_Baseline"] = r

    # Our method on truncated refs
    places_trunc = discover_places(ref_trunc, min_place_size=min_place_size)
    sizes_t = [len(p) for p in places_trunc]
    print(f"  Discovered {len(places_trunc)} places on truncated refs")
    thresholds_t, ks_t = compute_adaptive_thresholds(ref_trunc, places_trunc)
    print(f"  Adaptive k (trunc): min={min(ks_t):.2f} mean={np.mean(ks_t):.2f} max={max(ks_t):.2f}")

    ours_pred_nat = query_our_method(S_nat, places_trunc, thresholds_t)
    r = evaluate(ours_pred_nat, genuine_nat, distractor_nat, gt_map_trunc, tolerance)
    print(f"  Ours:         P={r['P']:.1f}% R={r['R']:.1f}% F1={r['F1']:.1f}% Rej={r['rejection']:.1f}%")
    results["natural_Ours"] = r

    # Vysotska threshold only (no sequence matcher)
    vt_pred_nat = run_vysotska_threshold_only(S_nat)
    r = evaluate(vt_pred_nat, genuine_nat, distractor_nat, gt_map_trunc, tolerance)
    print(f"  Vys (thresh): P={r['P']:.1f}% R={r['R']:.1f}% F1={r['F1']:.1f}% Rej={r['rejection']:.1f}%")
    results["natural_VysotskaThresh"] = r

    # Vysotska full pipeline
    vys_pred_nat = run_vysotska(S_nat, fanout=fanout)
    r = evaluate(vys_pred_nat, genuine_nat, distractor_nat, gt_map_trunc, tolerance)
    print(f"  Vys (full):   P={r['P']:.1f}% R={r['R']:.1f}% F1={r['F1']:.1f}% Rej={r['rejection']:.1f}%")
    results["natural_Vysotska"] = r

    return results


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    all_results = {}

    # 1. Nordland-500 (tol=2 to match Vysotska evaluation)
    ref, query, gt, tol = load_sequential_dataset(
        "Nordland-500",
        "cache/Nordland_salad/winter/dinov2_salad",
        "cache/Nordland_salad/summer/dinov2_salad",
        500, 500, tolerance=2
    )
    all_results["Nordland-500"] = run_dataset("Nordland-500", ref, query, gt, tol, fanout=5)

    # 2. Bonn
    ref, query, gt, tol = load_vysotska_dataset(
        "Bonn", "images/bonn_example",
        "cache/Bonn/reference/dinov2-salad",
        "cache/Bonn/query/dinov2-salad",
        tolerance=2
    )
    all_results["Bonn"] = run_dataset("Bonn", ref, query, gt, tol, fanout=5)

    # 3. Freiburg
    ref, query, gt, tol = load_vysotska_dataset(
        "Freiburg", "images/freiburg_example",
        "cache/Freiburg/reference/dinov2-salad",
        "cache/Freiburg/query/dinov2-salad",
        tolerance=2
    )
    all_results["Freiburg"] = run_dataset("Freiburg", ref, query, gt, tol, fanout=5)

    # 4. GardensPoint (tol=2 to match Vysotska datasets — day_left/day_right have spatial offset)
    ref, query, gt, tol = load_sequential_dataset(
        "GardensPoint",
        "cache/GardensPoint/day_left/dinov2-salad",
        "cache/GardensPoint/day_right/dinov2-salad",
        200, 200, tolerance=2
    )
    all_results["GardensPoint"] = run_dataset("GardensPoint", ref, query, gt, tol, fanout=5)

    # 5. SFU
    ref, query, gt, tol = load_sfu_dataset()
    all_results["SFU"] = run_dataset("SFU", ref, query, gt, 2, fanout=5)

    # 6. ESSEX3IN1
    ref, query, gt, tol = load_sequential_dataset(
        "ESSEX3IN1",
        "cache/ESSEX3IN1/reference/dinov2-salad",
        "cache/ESSEX3IN1/query/dinov2-salad",
        210, 210, tolerance=2
    )
    all_results["ESSEX3IN1"] = run_dataset("ESSEX3IN1", ref, query, gt, tol, fanout=5)

    # ── Summary tables ──
    ds_order = ["Nordland-500", "Bonn", "Freiburg", "GardensPoint", "SFU", "ESSEX3IN1"]

    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY — DINOv2 SALAD, continuous adaptive k (m=2)")
    print(f"{'='*70}")

    print(f"\nClosed-Set F1%:")
    print(f"  {'Dataset':<15} {'Baseline':>10} {'Ours':>10} {'Vys-thr':>10} {'Vys-full':>10}")
    print(f"  {'─'*59}")
    for name in ds_order:
        r = all_results[name]
        bl = r["closed_Baseline"]["F1"]
        ours = r["closed_Ours"]["F1"]
        vt = r["closed_VysotskaThresh"]["F1"]
        vys = r["closed_Vysotska"]["F1"]
        print(f"  {name:<15} {bl:>9.1f}% {ours:>9.1f}% {vt:>9.1f}% {vys:>9.1f}%")

    print(f"\nNatural Open-Set (70% ref map):")
    print(f"  {'Dataset':<15} {'Ours F1':>8} {'Ours Rej':>9} "
          f"{'VT F1':>8} {'VT Rej':>9} {'Vys F1':>8} {'Vys Rej':>9}")
    print(f"  {'─'*72}")
    for name in ds_order:
        r = all_results[name]
        of1 = r["natural_Ours"]["F1"]
        orej = r["natural_Ours"]["rejection"]
        vtf1 = r["natural_VysotskaThresh"]["F1"]
        vtrej = r["natural_VysotskaThresh"]["rejection"]
        vf1 = r["natural_Vysotska"]["F1"]
        vrej = r["natural_Vysotska"]["rejection"]
        print(f"  {name:<15} {of1:>7.1f}% {orej:>8.1f}% "
              f"{vtf1:>7.1f}% {vtrej:>8.1f}% {vf1:>7.1f}% {vrej:>8.1f}%")

    out_path = "results/final_all_datasets_dinov2salad.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
