"""
Compare current k formula (sep/2, equal-variance midpoint) against the
Bayes-derived k that uses both per-place positive and negative spreads:

    k_current = clip(sep / 2,              1, 2)
    k_Bayes   = clip(sep / (1 + sigma_pos/sigma_neg), 1, 2)

Reduces to k_current when sigma_pos = sigma_neg.

Outputs per-dataset:
  - sigma_pos / sigma_neg ratio distribution across places
  - k_current vs k_Bayes statistics across places
  - Natural open-set F1 + rejection for both formulas
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


def discover_places(ref_descs, alpha=1.5, m=2, h=2):
    d = OnlinePlaceDiscovery(bootstrap_std_factor=alpha,
                             min_place_size=m, hysteresis=h)
    for i in range(len(ref_descs)):
        d.process_frame(ref_descs[i], i, verbose=False)
    return d.places


def compute_thresholds(ref_descs, places, formula="midpoint"):
    """
    Return (thresholds, info) where info has per-place fields:
        mu_pos, mu_neg, sigma_pos, sigma_neg, sep, ratio, k
    """
    ref_S = ref_descs @ ref_descs.T
    thresholds = []
    info = []
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
        neg_sims = np.array(neg_sims) if len(neg_sims) else np.array([0.0])
        mu_pos = float(pos_sims.mean())
        mu_neg = float(neg_sims.mean())
        sigma_pos = float(pos_sims.std()) if len(pos_sims) > 1 else 0.0
        sigma_neg = float(neg_sims.std()) if len(neg_sims) > 1 else 1e-8
        sigma_neg = max(sigma_neg, 1e-8)
        sep = (mu_pos - mu_neg) / sigma_neg
        ratio = sigma_pos / sigma_neg if sigma_neg > 1e-8 else 1.0

        if formula == "midpoint":
            k_raw = sep / 2.0
        elif formula == "bayes":
            k_raw = sep / (1.0 + ratio)
        else:
            raise ValueError(formula)

        k = max(1.0, min(k_raw, 2.0))
        thresholds.append(mu_neg + k * sigma_neg)
        info.append({
            "mu_pos": mu_pos, "mu_neg": mu_neg,
            "sigma_pos": sigma_pos, "sigma_neg": sigma_neg,
            "sep": sep, "ratio": ratio, "k_raw": k_raw, "k": k,
            "place_size": len(place),
        })
    return thresholds, info


def query_filter_then_rank(S, places, thresholds):
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


def open_set_evaluation(ref_descs, query_descs, gt_map, tolerance,
                        formula, ref_fraction=0.7):
    n_ref = len(ref_descs)
    n_query = len(query_descs)
    ref_cutoff = int(n_ref * ref_fraction)
    ref_trunc = ref_descs[:ref_cutoff]
    S_nat = query_descs @ ref_trunc.T

    genuine = []
    distractor = []
    gt_trunc = {}
    for q in range(n_query):
        gt = gt_map.get(q)
        if gt is None:
            distractor.append(q)
            continue
        if isinstance(gt, list):
            refs_in_range = [r for r in gt if r < ref_cutoff]
            if refs_in_range:
                genuine.append(q)
                gt_trunc[q] = refs_in_range
            else:
                distractor.append(q)
        else:
            if gt < ref_cutoff:
                genuine.append(q)
                gt_trunc[q] = gt
            else:
                distractor.append(q)

    places = discover_places(ref_trunc)
    thresholds, info = compute_thresholds(ref_trunc, places, formula=formula)
    pred = query_filter_then_rank(S_nat, places, thresholds)
    r = evaluate(pred, genuine, distractor, gt_trunc, tolerance)
    return r, info


def load_all():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = {}

    ref, q, gt, tol = load_sequential_dataset(
        "Nordland-500",
        "cache/Nordland-500/winter/dinov2-salad",
        "cache/Nordland-500/summer/dinov2-salad",
        500, 500, tolerance=2,
    )
    out["Nordland-500"] = (ref, q, gt, tol)

    ref, q, gt, tol = load_vysotska_dataset(
        "Bonn", "images/bonn_example",
        "cache/Bonn/reference/dinov2-salad",
        "cache/Bonn/query/dinov2-salad",
        tolerance=2,
    )
    out["Bonn"] = (ref, q, gt, tol)

    ref, q, gt, tol = load_vysotska_dataset(
        "Freiburg", "images/freiburg_example",
        "cache/Freiburg/reference/dinov2-salad",
        "cache/Freiburg/query/dinov2-salad",
        tolerance=2,
    )
    out["Freiburg"] = (ref, q, gt, tol)

    ref, q, gt, tol = load_sequential_dataset(
        "GardensPoint",
        "cache/GardensPoint/day_left/dinov2-salad",
        "cache/GardensPoint/day_right/dinov2-salad",
        200, 200, tolerance=2,
    )
    out["GardensPoint"] = (ref, q, gt, tol)

    ref, q, gt, tol = load_sfu_dataset()
    out["SFU"] = (ref, q, gt, 2)

    ref, q, gt, tol = load_sequential_dataset(
        "ESSEX3IN1",
        "cache/ESSEX3IN1/reference/dinov2-salad",
        "cache/ESSEX3IN1/query/dinov2-salad",
        210, 210, tolerance=2,
    )
    out["ESSEX3IN1"] = (ref, q, gt, tol)

    return out


ORDER = ["Nordland-500", "Bonn", "Freiburg", "GardensPoint", "SFU", "ESSEX3IN1"]


def main():
    data = load_all()
    summary = {}

    print(f"\n{'='*80}")
    print("Per-dataset sigma_pos / sigma_neg ratio distribution (on 70% ref split)")
    print(f"{'='*80}")
    print(f"{'dataset':16s}  {'n_places':>9s}  "
          f"{'ratio min':>10s} {'med':>10s} {'max':>10s}  "
          f"{'frac in [0.5,2]':>18s}")

    for ds in ORDER:
        ref, q, gt, tol = data[ds]
        # use the same 70% truncation as the open-set protocol
        ref_trunc = ref[:int(len(ref)*0.7)]
        places = discover_places(ref_trunc)
        _, info = compute_thresholds(ref_trunc, places, formula="midpoint")
        ratios = np.array([d["ratio"] for d in info])
        in_band = float(np.mean((ratios >= 0.5) & (ratios <= 2.0)))
        print(f"{ds:16s}  {len(info):>9d}  "
              f"{ratios.min():>10.3f} {np.median(ratios):>10.3f} {ratios.max():>10.3f}  "
              f"{in_band*100:>17.1f}%")
        summary[ds] = {"ratios": ratios.tolist()}

    print(f"\n{'='*80}")
    print("Natural open-set: k_current (sep/2) vs k_Bayes (sep / (1+σ⁺/σ⁻))")
    print(f"{'='*80}")
    print(f"{'dataset':16s}  "
          f"{'F1 cur':>8s} {'F1 Bay':>8s} {'ΔF1':>8s}  "
          f"{'Rej cur':>9s} {'Rej Bay':>9s} {'ΔRej':>8s}  "
          f"{'k_cur med':>10s} {'k_Bay med':>10s}")

    for ds in ORDER:
        ref, q, gt, tol = data[ds]
        r_cur, info_cur = open_set_evaluation(ref, q, gt, tol, "midpoint")
        r_bay, info_bay = open_set_evaluation(ref, q, gt, tol, "bayes")
        k_cur = np.median([d["k"] for d in info_cur])
        k_bay = np.median([d["k"] for d in info_bay])
        df1 = r_bay["F1"] - r_cur["F1"]
        drej = r_bay["rejection"] - r_cur["rejection"]
        print(f"{ds:16s}  "
              f"{r_cur['F1']:>8.1f} {r_bay['F1']:>8.1f} {df1:>+8.1f}  "
              f"{r_cur['rejection']:>9.1f} {r_bay['rejection']:>9.1f} {drej:>+8.1f}  "
              f"{k_cur:>10.3f} {k_bay:>10.3f}")
        summary[ds].update({
            "F1_current": r_cur["F1"], "F1_bayes": r_bay["F1"],
            "rej_current": r_cur["rejection"], "rej_bayes": r_bay["rejection"],
            "k_current_median": float(k_cur), "k_bayes_median": float(k_bay),
        })

    out_path = "results/bayes_k_comparison.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
