"""
Nordland-500 experiments for the IEEE RAL paper.

Runs both closed-set and open-set evaluation on Nordland-500,
comparing our filter_n method against Vysotska's adaptive thresholding.

Demonstrates that our method:
  (b) Competes in closed-set AND excels in open-set.

Uses GardensPoint day_left images as distractors for open-set.

Usage:
    python -m experiments.nordland_500_paper_experiments
    python -m experiments.nordland_500_paper_experiments --descriptor dinov2-salad
"""

import argparse
import os
import sys
import json
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from experiments.experiment_utils import (
    load_cached_descriptors, discover_places, compute_thresholds,
    evaluate, evaluate_vysotska, count_cached
)
from experiments.vysotska_threshold import VysotskaDaptiveThreshold
from experiments.vysotska_sequence_matcher import sequence_match, evaluate_sequence_match
from utils import normalize_l2


N = 500  # Nordland first 500

DESCRIPTOR_CONFIGS = {
    "eigenplaces": {
        "ref_cache": "cache/Nordland_filtered/winter/eigenplaces",
        "query_cache": "cache/Nordland_filtered/summer/eigenplaces",
        "distractor_cache": "cache/GardensPoint/day_left/eigenplaces",
        "name": "EigenPlaces (ResNet50, 2048-dim)",
    },
    "dinov2-salad": {
        "ref_cache": "cache/Nordland_salad/winter/dinov2_salad",
        "query_cache": "cache/Nordland_salad/summer/dinov2_salad",
        "distractor_cache": "cache/GardensPoint/day_left/dinov2-salad",
        "name": "DINOv2 SALAD (ViT-B/14, 8448-dim)",
    },
}

OUTPUT_DIR = "results/nordland_500_paper"


def load_descs(cache_dir, n):
    """Load and L2-normalize descriptors."""
    descs = load_cached_descriptors(cache_dir, n)
    return normalize_l2(descs.astype(np.float32))


def run_sequence_matching_eval(S, label, vysotska_thresholds=None):
    """Run sequence matching evaluation with various configs."""
    print(f"\n   [{label}]")

    # Use Vysotska adaptive threshold converted to cost
    if vysotska_thresholds is not None:
        median_thresh = float(np.median(vysotska_thresholds))
        nmc = 1.0 - median_thresh
        nmc_label = f"adaptive (median_sim={median_thresh:.3f})"
    else:
        nmc = 0.5
        nmc_label = "fixed (sim>0.5)"

    matches, all_path, path_real, path_hidden = sequence_match(
        S, non_matching_cost=nmc, fanout=3
    )
    for tol in [0, 1, 2, 5]:
        r = evaluate_sequence_match(S, matches, tolerance=tol)
        print(f"     tol=±{tol}: P={r['precision']:.4f} R={r['recall']:.4f} "
              f"F1={r['f1']:.4f} matched={r['n_matched']}/{r['n_total']}")
    return evaluate_sequence_match(S, matches, tolerance=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", default="eigenplaces",
                        choices=list(DESCRIPTOR_CONFIGS.keys()))
    args = parser.parse_args()
    cfg = DESCRIPTOR_CONFIGS[args.descriptor]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"{'='*70}")
    print(f"NORDLAND-500 PAPER EXPERIMENTS")
    print(f"{'='*70}")
    print(f"Descriptor: {cfg['name']}")
    print(f"Setup: First {N} images, winter=ref, summer=query")

    # ── Load descriptors ──────────────────────────────────────────────
    print("\nLoading reference and query descriptors...")
    ref_descs = load_descs(cfg["ref_cache"], N)
    query_descs = load_descs(cfg["query_cache"], N)
    print(f"  Reference: {ref_descs.shape}")
    print(f"  Query:     {query_descs.shape}")

    # Load distractors (GardensPoint day_left)
    n_dist_available = count_cached(cfg["distractor_cache"])
    if n_dist_available > 0:
        n_dist = min(n_dist_available, 200)
        print(f"\nLoading {n_dist} distractor descriptors from GardensPoint...")
        dist_descs = load_descs(cfg["distractor_cache"], n_dist)
        has_distractors = True
    else:
        print(f"\nNo distractors cached at {cfg['distractor_cache']}")
        print("  Skipping open-set experiments. Run with GPU to extract features first.")
        has_distractors = False
        n_dist = 0
        dist_descs = None

    # ── Similarity matrix ─────────────────────────────────────────────
    S = query_descs @ ref_descs.T
    print(f"\nSimilarity matrix: {S.shape}")
    print(f"  Diagonal mean (correct matches): {np.diag(S).mean():.4f}")
    print(f"  Off-diagonal mean: {(S.sum() - np.trace(S)) / (N*N - N):.4f}")

    # ── Place discovery ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("PLACE DISCOVERY")
    places = discover_places(ref_descs)
    sizes = [len(p) for p in places]
    print(f"  Discovered {len(places)} places")
    print(f"  Place sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
    covered = sum(len(p) for p in places)
    print(f"  Coverage: {covered}/{N} frames")

    results = {
        "descriptor": cfg["name"],
        "n_ref": N, "n_query": N, "n_places": len(places),
    }

    # ══════════════════════════════════════════════════════════════════
    # PART 1: CLOSED-SET (all queries have a match)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PART 1: CLOSED-SET EVALUATION")
    print(f"{'='*70}")

    # 1a. Baseline (no threshold)
    no_thresh = {p: -np.inf for p in range(len(places))}
    r_base = evaluate(query_descs, ref_descs, places, no_thresh)
    print(f"\n  Baseline (no threshold):")
    print(f"    P={r_base['P']:.1f}% R={r_base['R']:.1f}% F1={r_base['F1']:.1f}%")

    # 1b. Mean_bad
    thresh_mb = compute_thresholds(ref_descs, places, method="mean_bad")
    r_mb = evaluate(query_descs, ref_descs, places, thresh_mb)
    print(f"  Mean_bad:")
    print(f"    P={r_mb['P']:.1f}% R={r_mb['R']:.1f}% F1={r_mb['F1']:.1f}%")

    # 1c. Filter_n
    thresh_fn = compute_thresholds(ref_descs, places, method="filter_n")
    r_fn = evaluate(query_descs, ref_descs, places, thresh_fn)
    print(f"  Filter_n:")
    print(f"    P={r_fn['P']:.1f}% R={r_fn['R']:.1f}% F1={r_fn['F1']:.1f}%")

    # 1d. Vysotska adaptive thresholding
    r_vy = evaluate_vysotska(query_descs, ref_descs, places)
    print(f"  Vysotska:")
    print(f"    P={r_vy['P']:.1f}% R={r_vy['R']:.1f}% F1={r_vy['F1']:.1f}%")

    # 1e. Vysotska sequence matcher (their full pipeline)
    print(f"\n  Vysotska Sequence Matcher (graph shortest path):")
    vysotska = VysotskaDaptiveThreshold(patch_size=20)
    vysotska_thresholds, _ = vysotska.compute_thresholds(S)
    r_sm = run_sequence_matching_eval(S, "Seq.Match + adaptive thresh", vysotska_thresholds)

    # 1f. Image-level baseline (argmax, no places)
    best_matches = np.argmax(S, axis=1)
    n_correct_img = sum(1 for q in range(N) if best_matches[q] == q)
    n_correct_tol1 = sum(1 for q in range(N) if abs(best_matches[q] - q) <= 1)
    print(f"\n  Image-level argmax (no places, no threshold):")
    print(f"    Exact match: {n_correct_img}/{N} ({n_correct_img/N*100:.1f}%)")
    print(f"    tol±1:       {n_correct_tol1}/{N} ({n_correct_tol1/N*100:.1f}%)")

    results["closed_set"] = {
        "baseline": r_base, "mean_bad": r_mb, "filter_n": r_fn,
        "vysotska": r_vy,
        "seq_matcher_tol1": r_sm,
        "image_argmax_exact": n_correct_img / N * 100,
        "image_argmax_tol1": n_correct_tol1 / N * 100,
    }

    # ══════════════════════════════════════════════════════════════════
    # PART 2: OPEN-SET (add distractors)
    # ══════════════════════════════════════════════════════════════════
    if has_distractors:
        print(f"\n{'='*70}")
        print(f"PART 2: OPEN-SET EVALUATION ({N} genuine + {n_dist} GardensPoint distractors)")
        print(f"{'='*70}")

        # Concatenate genuine + distractor queries
        query_open = np.vstack([query_descs, dist_descs])

        for ratio_label, n_d in [("1:1", min(n_dist, N)),
                                  ("1:2", min(n_dist, N * 2))]:
            actual_n_d = min(n_d, n_dist)
            if actual_n_d < n_d:
                if actual_n_d == 0:
                    continue
                ratio_label = f"1:{actual_n_d/N:.1f}"

            q_subset = np.vstack([query_descs, dist_descs[:actual_n_d]])
            print(f"\n  --- Ratio {ratio_label} ({N} genuine + {actual_n_d} distractors) ---")

            r_base_o = evaluate(q_subset, ref_descs, places, no_thresh, n_genuine=N)
            print(f"  Baseline:  P={r_base_o['P']:.1f}% R={r_base_o['R']:.1f}% "
                  f"F1={r_base_o['F1']:.1f}% Rej={r_base_o['dist_rej']:.0f}%")

            r_mb_o = evaluate(q_subset, ref_descs, places, thresh_mb, n_genuine=N)
            print(f"  Mean_bad:  P={r_mb_o['P']:.1f}% R={r_mb_o['R']:.1f}% "
                  f"F1={r_mb_o['F1']:.1f}% Rej={r_mb_o['dist_rej']:.0f}%")

            r_fn_o = evaluate(q_subset, ref_descs, places, thresh_fn, n_genuine=N)
            print(f"  Filter_n:  P={r_fn_o['P']:.1f}% R={r_fn_o['R']:.1f}% "
                  f"F1={r_fn_o['F1']:.1f}% Rej={r_fn_o['dist_rej']:.0f}%")

            r_vy_o = evaluate_vysotska(q_subset, ref_descs, places, n_genuine=N)
            print(f"  Vysotska:  P={r_vy_o['P']:.1f}% R={r_vy_o['R']:.1f}% "
                  f"F1={r_vy_o['F1']:.1f}% Rej={r_vy_o['dist_rej']:.0f}%")

            results[f"open_set_{ratio_label}"] = {
                "n_genuine": N, "n_distractors": actual_n_d,
                "baseline": r_base_o, "mean_bad": r_mb_o,
                "filter_n": r_fn_o, "vysotska": r_vy_o,
            }

        # Distractor ratio sweep (like GardensPoint experiment)
        print(f"\n  --- Distractor Ratio Sweep ---")
        sweep_results = []
        for n_d in [0, 50, 100, 150, 200]:
            if n_d > n_dist:
                break
            if n_d == 0:
                q_sub = query_descs
                n_gen = N
            else:
                q_sub = np.vstack([query_descs, dist_descs[:n_d]])
                n_gen = N
            r_fn_s = evaluate(q_sub, ref_descs, places, thresh_fn, n_genuine=n_gen)
            r_vy_s = evaluate_vysotska(q_sub, ref_descs, places, n_genuine=n_gen)
            ratio = n_d / N
            print(f"  ratio={ratio:.1f}: filter_n F1={r_fn_s['F1']:.1f}% "
                  f"Vysotska F1={r_vy_s['F1']:.1f}%")
            sweep_results.append({
                "ratio": ratio, "n_distractors": n_d,
                "filter_n": r_fn_s, "vysotska": r_vy_s
            })
        results["distractor_sweep"] = sweep_results

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SUMMARY TABLE (for paper)")
    print(f"{'='*70}")
    print(f"\n{'Method':<25} {'Closed F1':>10} {'Open F1':>10} {'Dist.Rej':>10}")
    print(f"{'─'*55}")
    print(f"{'Baseline':<25} {r_base['F1']:>9.1f}% {'---':>10} {'0%':>10}")
    print(f"{'Mean_bad':<25} {r_mb['F1']:>9.1f}% ", end="")
    if has_distractors:
        r_mb_o1 = results.get("open_set_1:1", {}).get("mean_bad", {})
        print(f"{r_mb_o1.get('F1', 0):>9.1f}% {r_mb_o1.get('dist_rej', 0):>9.0f}%")
    else:
        print("---        ---")
    print(f"{'Filter_n (ours)':<25} {r_fn['F1']:>9.1f}% ", end="")
    if has_distractors:
        r_fn_o1 = results.get("open_set_1:1", {}).get("filter_n", {})
        print(f"{r_fn_o1.get('F1', 0):>9.1f}% {r_fn_o1.get('dist_rej', 0):>9.0f}%")
    else:
        print("---        ---")
    print(f"{'Vysotska threshold':<25} {r_vy['F1']:>9.1f}% ", end="")
    if has_distractors:
        r_vy_o1 = results.get("open_set_1:1", {}).get("vysotska", {})
        print(f"{r_vy_o1.get('F1', 0):>9.1f}% {r_vy_o1.get('dist_rej', 0):>9.0f}%")
    else:
        print("---        ---")
    print(f"{'Vysotska seq.match':<25} ", end="")
    print(f"F1={r_sm['f1']:.4f} (tol±1, closed-set image-level)")
    print(f"{'Vysotska (reported)':<25} {'0.98':>10}")
    print(f"{'='*70}")

    # Save results
    out_path = os.path.join(OUTPUT_DIR, f"results_{args.descriptor}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
