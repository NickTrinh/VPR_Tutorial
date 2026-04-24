"""
Sweep min_place_size (m) across all 6 datasets at the same value to find
a uniform m that works well everywhere. Uses the continuous adaptive k
  k = clip(sep/2, 1, 2).

Output: a table per m of closed-set F1, open-set F1, and rejection.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.final_all_datasets import (
    discover_places, query_our_method, evaluate, run_vysotska,
    compute_adaptive_thresholds,
    load_sequential_dataset, load_vysotska_dataset, load_sfu_dataset,
)


def compute_thresholds_continuous(ref_descs, places):
    thresholds, _ = compute_adaptive_thresholds(ref_descs, places)
    return thresholds


def build_query_classification(n_query, gt_map, ref_cutoff):
    genuine_closed = sorted(gt_map.keys())
    genuine_nat, distractor_nat, gt_map_trunc = [], [], {}
    for q in range(n_query):
        gt = gt_map.get(q)
        if gt is None:
            distractor_nat.append(q)
            continue
        if isinstance(gt, list):
            refs_in = [r for r in gt if r < ref_cutoff]
            if refs_in:
                genuine_nat.append(q)
                gt_map_trunc[q] = refs_in
            else:
                distractor_nat.append(q)
        else:
            if gt < ref_cutoff:
                genuine_nat.append(q)
                gt_map_trunc[q] = gt
            else:
                distractor_nat.append(q)
    return genuine_closed, genuine_nat, distractor_nat, gt_map_trunc


def load_all_datasets():
    datasets = {}

    ref, query, gt, tol = load_sequential_dataset(
        "Nordland-500", "cache/Nordland_salad/winter/dinov2_salad",
        "cache/Nordland_salad/summer/dinov2_salad", 500, 500, tolerance=2)
    datasets["Nordland-500"] = (ref, query, gt, tol)

    ref, query, gt, tol = load_vysotska_dataset(
        "Bonn", "images/bonn_example", "cache/Bonn/reference/dinov2-salad",
        "cache/Bonn/query/dinov2-salad", tolerance=2)
    datasets["Bonn"] = (ref, query, gt, tol)

    ref, query, gt, tol = load_vysotska_dataset(
        "Freiburg", "images/freiburg_example", "cache/Freiburg/reference/dinov2-salad",
        "cache/Freiburg/query/dinov2-salad", tolerance=2)
    datasets["Freiburg"] = (ref, query, gt, tol)

    ref, query, gt, tol = load_sequential_dataset(
        "GardensPoint", "cache/GardensPoint/day_left/dinov2-salad",
        "cache/GardensPoint/day_right/dinov2-salad", 200, 200, tolerance=2)
    datasets["GardensPoint"] = (ref, query, gt, tol)

    ref, query, gt, _ = load_sfu_dataset()
    datasets["SFU"] = (ref, query, gt, 2)

    ref, query, gt, tol = load_sequential_dataset(
        "ESSEX3IN1", "cache/ESSEX3IN1/reference/dinov2-salad",
        "cache/ESSEX3IN1/query/dinov2-salad", 210, 210, tolerance=2)
    datasets["ESSEX3IN1"] = (ref, query, gt, tol)

    return datasets


def run_sweep(m_values=(2, 3, 4, 5), ref_fraction=0.7):
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datasets = load_all_datasets()
    ds_order = ["Nordland-500", "Bonn", "Freiburg", "GardensPoint", "SFU", "ESSEX3IN1"]

    # Pre-compute Vysotska baselines once per dataset (independent of m)
    vys_results = {}
    for name in ds_order:
        ref_descs, query_descs, gt_map, tolerance = datasets[name]
        n_ref = len(ref_descs)
        n_query = len(query_descs)
        ref_cutoff = int(n_ref * ref_fraction)
        ref_trunc = ref_descs[:ref_cutoff]

        S_closed = query_descs @ ref_descs.T
        S_nat = query_descs @ ref_trunc.T

        gc, gn, dn, gt_trunc = build_query_classification(n_query, gt_map, ref_cutoff)

        vys_c_pred = run_vysotska(S_closed, fanout=5)
        vys_c = evaluate(vys_c_pred, gc, [], gt_map, tolerance)
        vys_o_pred = run_vysotska(S_nat, fanout=5)
        vys_o = evaluate(vys_o_pred, gn, dn, gt_trunc, tolerance)
        vys_results[name] = (vys_c, vys_o)

    # Sweep m
    all_m_results = {}
    for m in m_values:
        print(f"\n{'='*90}")
        print(f"m = {m}")
        print(f"{'='*90}")
        print(f"{'Dataset':<14} {'#places':>8} {'C F1':>7} {'O F1':>7} {'O Rej':>7}"
              f"   |   {'Vys C F1':>9} {'Vys O F1':>9} {'Vys O Rej':>10}")
        print("-" * 95)

        m_results = {}
        for name in ds_order:
            ref_descs, query_descs, gt_map, tolerance = datasets[name]
            n_ref = len(ref_descs)
            n_query = len(query_descs)
            ref_cutoff = int(n_ref * ref_fraction)
            ref_trunc = ref_descs[:ref_cutoff]

            S_closed = query_descs @ ref_descs.T
            S_nat = query_descs @ ref_trunc.T

            gc, gn, dn, gt_trunc = build_query_classification(n_query, gt_map, ref_cutoff)

            # Closed-set discovery on full refs
            places_full = discover_places(ref_descs, min_place_size=m)
            t_closed = compute_thresholds_continuous(ref_descs, places_full)
            pred_c = query_our_method(S_closed, places_full, t_closed)
            rc = evaluate(pred_c, gc, [], gt_map, tolerance)

            # Open-set discovery on truncated refs
            places_trunc = discover_places(ref_trunc, min_place_size=m)
            t_open = compute_thresholds_continuous(ref_trunc, places_trunc)
            pred_o = query_our_method(S_nat, places_trunc, t_open)
            ro = evaluate(pred_o, gn, dn, gt_trunc, tolerance)

            vys_c, vys_o = vys_results[name]
            print(f"{name:<14} {len(places_trunc):>8}"
                  f" {rc['F1']:>6.1f}% {ro['F1']:>6.1f}% {ro['rejection'] or 0.0:>6.1f}%"
                  f"   |   {vys_c['F1']:>8.1f}% {vys_o['F1']:>8.1f}%"
                  f" {vys_o['rejection'] or 0.0:>9.1f}%")

            m_results[name] = {
                "n_places_full": len(places_full),
                "n_places_trunc": len(places_trunc),
                "closed_F1": rc["F1"],
                "open_F1": ro["F1"],
                "open_rej": ro["rejection"],
            }
        all_m_results[m] = m_results

    # Summary across m: F1 mean + min rejection across datasets
    print(f"\n{'='*90}")
    print("SUMMARY (averaged across 6 datasets)")
    print(f"{'='*90}")
    print(f"{'m':>3}   {'mean closed F1':>16}   {'mean open F1':>14}   {'mean rej':>10}   {'min rej':>9}")
    for m, r in all_m_results.items():
        cf1 = np.mean([r[d]["closed_F1"] for d in ds_order])
        of1 = np.mean([r[d]["open_F1"] for d in ds_order])
        rej = [r[d]["open_rej"] or 0.0 for d in ds_order]
        print(f"{m:>3}   {cf1:>15.2f}%   {of1:>13.2f}%   {np.mean(rej):>9.2f}%   {min(rej):>8.2f}%")

    return all_m_results


if __name__ == "__main__":
    run_sweep(m_values=(2, 3, 4, 5))
