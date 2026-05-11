"""
Robustness sweep for the four data-touching constants in our pipeline:
  - alpha     (bootstrap sensitivity, default 1.5)
  - m         (min place size, default 2)
  - h         (hysteresis, default 2)
  - divisor   (in k_adaptive = clip(sep/divisor, 1, 2), default 2.0)

For each constant, vary it across a range while holding the other three at
defaults, then re-run the natural open-set protocol on all six datasets.
Reports F1 + rejection rate per (constant, value, dataset).
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.online_place_discovery import OnlinePlaceDiscovery
from experiments.final_all_datasets import (
    load_sequential_dataset,
    load_vysotska_dataset,
    load_sfu_dataset,
    evaluate,
)


# ── Parameterized pipeline pieces ──

def discover_places_p(ref_descs, alpha=1.5, min_place_size=2, hysteresis=2):
    discoverer = OnlinePlaceDiscovery(
        bootstrap_std_factor=alpha,
        min_place_size=min_place_size,
        hysteresis=hysteresis,
    )
    for i in range(len(ref_descs)):
        discoverer.process_frame(ref_descs[i], i, verbose=False)
    return discoverer.places


def compute_thresholds_p(ref_descs, places, divisor=2.0):
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
        k = max(1.0, min(sep / divisor, 2.0))
        thresholds.append(mu_neg + k * sigma_neg)
        ks.append(k)
    return thresholds, ks


def query_our_method(S, places, thresholds):
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


def run_open_set_point(ref_descs, query_descs, gt_map, tolerance,
                       alpha, m, h, divisor, ref_fraction=0.7):
    n_ref = len(ref_descs)
    n_query = len(query_descs)
    ref_cutoff = int(n_ref * ref_fraction)
    ref_trunc = ref_descs[:ref_cutoff]
    S_nat = query_descs @ ref_trunc.T

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

    try:
        places = discover_places_p(ref_trunc, alpha=alpha,
                                   min_place_size=m, hysteresis=h)
        thresholds, _ = compute_thresholds_p(ref_trunc, places, divisor=divisor)
        pred = query_our_method(S_nat, places, thresholds)
        r = evaluate(pred, genuine_nat, distractor_nat, gt_map_trunc, tolerance)
        return {"F1": r["F1"], "rejection": r["rejection"],
                "P": r["P"], "R": r["R"], "n_places": len(places)}
    except Exception as e:
        return {"F1": float("nan"), "rejection": float("nan"),
                "P": float("nan"), "R": float("nan"),
                "n_places": 0, "error": str(e)}


# ── Load all 6 datasets once ──

def load_all():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datasets = {}

    ref, query, gt, tol = load_sequential_dataset(
        "Nordland-500",
        "cache/Nordland-500/winter/dinov2-salad",
        "cache/Nordland-500/summer/dinov2-salad",
        500, 500, tolerance=2,
    )
    datasets["Nordland-500"] = (ref, query, gt, tol)

    ref, query, gt, tol = load_vysotska_dataset(
        "Bonn", "images/bonn_example",
        "cache/Bonn/reference/dinov2-salad",
        "cache/Bonn/query/dinov2-salad",
        tolerance=2,
    )
    datasets["Bonn"] = (ref, query, gt, tol)

    ref, query, gt, tol = load_vysotska_dataset(
        "Freiburg", "images/freiburg_example",
        "cache/Freiburg/reference/dinov2-salad",
        "cache/Freiburg/query/dinov2-salad",
        tolerance=2,
    )
    datasets["Freiburg"] = (ref, query, gt, tol)

    ref, query, gt, tol = load_sequential_dataset(
        "GardensPoint",
        "cache/GardensPoint/day_left/dinov2-salad",
        "cache/GardensPoint/day_right/dinov2-salad",
        200, 200, tolerance=2,
    )
    datasets["GardensPoint"] = (ref, query, gt, tol)

    ref, query, gt, tol = load_sfu_dataset()
    datasets["SFU"] = (ref, query, gt, 2)

    ref, query, gt, tol = load_sequential_dataset(
        "ESSEX3IN1",
        "cache/ESSEX3IN1/reference/dinov2-salad",
        "cache/ESSEX3IN1/query/dinov2-salad",
        210, 210, tolerance=2,
    )
    datasets["ESSEX3IN1"] = (ref, query, gt, tol)

    return datasets


# ── Sweep grid ──

DEFAULTS = {"alpha": 1.5, "m": 2, "h": 2, "divisor": 2.0}

SWEEPS = {
    "alpha":   [1.0, 1.25, 1.5, 1.75, 2.0],
    "m":       [2, 3, 4],
    "h":       [1, 2, 3, 4],
    "divisor": [1.5, 2.0, 2.5, 3.0],
}

DATASET_ORDER = ["Nordland-500", "Bonn", "Freiburg", "GardensPoint", "SFU", "ESSEX3IN1"]


def main():
    datasets = load_all()
    print(f"Loaded {len(datasets)} datasets.")

    all_results = {}  # all_results[const][value][dataset] = {F1, rejection, ...}

    for const, values in SWEEPS.items():
        all_results[const] = {}
        for v in values:
            params = dict(DEFAULTS)
            params[const] = v
            print(f"\n=== Sweep {const}={v} "
                  f"(alpha={params['alpha']}, m={params['m']}, "
                  f"h={params['h']}, divisor={params['divisor']}) ===",
                  flush=True)
            per_ds = {}
            for ds in DATASET_ORDER:
                ref, query, gt, tol = datasets[ds]
                r = run_open_set_point(
                    ref, query, gt, tol,
                    alpha=params["alpha"], m=params["m"],
                    h=params["h"], divisor=params["divisor"],
                )
                per_ds[ds] = r
                print(f"  {ds:14s}: F1={r['F1']:.1f}%  Rej={r['rejection']:.1f}%  "
                      f"({r['n_places']} places)", flush=True)
            all_results[const][str(v)] = per_ds

    # Save raw
    out = "results/robustness_sweep_dinov2salad.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out}")

    # Summary tables
    print("\n\n" + "=" * 80)
    print("ROBUSTNESS SUMMARY (natural open-set F1 / rejection rate, %)")
    print("=" * 80)
    for const, values in SWEEPS.items():
        print(f"\n--- {const} (others at defaults) ---")
        header = f"{const:>8s}  " + "  ".join(f"{d[:12]:>12s}" for d in DATASET_ORDER) + "      mean"
        print(header)
        for v in values:
            row_f1 = []
            row_rej = []
            for ds in DATASET_ORDER:
                r = all_results[const][str(v)][ds]
                row_f1.append(r["F1"])
                row_rej.append(r["rejection"])
            mean_f1 = float(np.mean(row_f1))
            cells_f1 = "  ".join(f"{x:>12.1f}" for x in row_f1)
            cells_rej = "  ".join(f"{x:>12.1f}" for x in row_rej)
            print(f"  F1 {str(v):>4s}: {cells_f1}  {mean_f1:8.1f}")
            mean_rej = float(np.mean(row_rej))
            print(f"  Rej{str(v):>4s}: {cells_rej}  {mean_rej:8.1f}")
        # range
        print(f"  --- range across {const} values ---")
        for ds in DATASET_ORDER + ["mean"]:
            if ds == "mean":
                f1s = [float(np.mean([all_results[const][str(v)][d]["F1"]
                                      for d in DATASET_ORDER])) for v in values]
                rejs = [float(np.mean([all_results[const][str(v)][d]["rejection"]
                                       for d in DATASET_ORDER])) for v in values]
            else:
                f1s = [all_results[const][str(v)][ds]["F1"] for v in values]
                rejs = [all_results[const][str(v)][ds]["rejection"] for v in values]
            print(f"    {ds:14s}  F1 spread = {max(f1s) - min(f1s):5.2f}  "
                  f"Rej spread = {max(rejs) - min(rejs):5.2f}")


if __name__ == "__main__":
    main()
