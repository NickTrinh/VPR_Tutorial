"""
Re-run the canonical natural-open-set pipeline on EigenPlaces descriptors
across all six datasets, using identical method code as
`final_all_datasets.py` (same place discovery, same threshold formula,
same Vysotska reimplementations).

This validates the descriptor-agnostic claim in §III-A of the paper.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.final_all_datasets import (
    load_sequential_dataset,
    load_vysotska_dataset,
    load_sfu_dataset,
    run_dataset,
)


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    all_results = {}

    # 1. Nordland-500
    ref, query, gt, tol = load_sequential_dataset(
        "Nordland-500",
        "cache/Nordland-500/winter/eigenplaces",
        "cache/Nordland-500/summer/eigenplaces",
        500, 500, tolerance=2,
    )
    all_results["Nordland-500"] = run_dataset("Nordland-500", ref, query, gt, tol, fanout=5)

    # 2. Bonn
    ref, query, gt, tol = load_vysotska_dataset(
        "Bonn", "images/bonn_example",
        "cache/Bonn/reference/eigenplaces",
        "cache/Bonn/query/eigenplaces",
        tolerance=2,
    )
    all_results["Bonn"] = run_dataset("Bonn", ref, query, gt, tol, fanout=5)

    # 3. Freiburg
    ref, query, gt, tol = load_vysotska_dataset(
        "Freiburg", "images/freiburg_example",
        "cache/Freiburg/reference/eigenplaces",
        "cache/Freiburg/query/eigenplaces",
        tolerance=2,
    )
    all_results["Freiburg"] = run_dataset("Freiburg", ref, query, gt, tol, fanout=5)

    # 4. GardensPoint
    ref, query, gt, tol = load_sequential_dataset(
        "GardensPoint",
        "cache/GardensPoint/day_left/eigenplaces",
        "cache/GardensPoint/day_right/eigenplaces",
        200, 200, tolerance=2,
    )
    all_results["GardensPoint"] = run_dataset("GardensPoint", ref, query, gt, tol, fanout=5)

    # 5. SFU — inline the canonical loader logic with EP cache paths
    from experiments.final_all_datasets import load_descs_pkl
    import numpy as np
    ref = load_descs_pkl("cache/SFU/dry/eigenplaces")
    query = load_descs_pkl("cache/SFU/jan/eigenplaces")
    gt_data = np.load("images/SFU/GT.npz")
    G = gt_data[list(gt_data.keys())[0]]
    gt = {}
    for q in range(G.shape[0]):
        refs = list(np.where(G[q] > 0)[0])
        if refs:
            gt[q] = [int(r) for r in refs]
    all_results["SFU"] = run_dataset("SFU", ref, query, gt, 2, fanout=5)

    # 6. ESSEX3IN1
    ref, query, gt, tol = load_sequential_dataset(
        "ESSEX3IN1",
        "cache/ESSEX3IN1/reference/eigenplaces",
        "cache/ESSEX3IN1/query/eigenplaces",
        210, 210, tolerance=2,
    )
    all_results["ESSEX3IN1"] = run_dataset("ESSEX3IN1", ref, query, gt, tol, fanout=5)

    # ── Summary tables ──
    ds_order = ["Nordland-500", "Bonn", "Freiburg", "GardensPoint", "SFU", "ESSEX3IN1"]

    print(f"\n\n{'='*78}")
    print("EIGENPLACES — closed-set F1 (%)")
    print(f"{'='*78}")
    print(f"{'Dataset':16s}  {'Baseline':>9s}  {'Ours':>7s}  {'Vys-thresh':>11s}  {'Vys-full':>9s}")
    for ds in ds_order:
        r = all_results[ds]
        print(f"{ds:16s}  "
              f"{r['closed_Baseline']['F1']:>9.1f}  "
              f"{r['closed_Ours']['F1']:>7.1f}  "
              f"{r['closed_VysotskaThresh']['F1']:>11.1f}  "
              f"{r['closed_Vysotska']['F1']:>9.1f}")

    print(f"\n{'='*78}")
    print("EIGENPLACES — natural open-set F1 / rejection rate (%)")
    print(f"{'='*78}")
    print(f"{'Dataset':16s}  {'Ours F1':>8s} {'Ours Rej':>9s}  "
          f"{'VT F1':>7s} {'VT Rej':>7s}  {'Vys F1':>7s} {'Vys Rej':>8s}")
    for ds in ds_order:
        r = all_results[ds]
        print(f"{ds:16s}  "
              f"{r['natural_Ours']['F1']:>8.1f} {r['natural_Ours']['rejection']:>9.1f}  "
              f"{r['natural_VysotskaThresh']['F1']:>7.1f} {r['natural_VysotskaThresh']['rejection']:>7.1f}  "
              f"{r['natural_Vysotska']['F1']:>7.1f} {r['natural_Vysotska']['rejection']:>8.1f}")

    out_path = "results/final_all_datasets_eigenplaces.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
