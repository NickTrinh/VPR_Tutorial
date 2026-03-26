"""
Shared utilities for all experiment scripts.

Provides:
  - load_cached_descriptors: load pickle descriptors into numpy array
  - discover_places: run online place discovery
  - compute_thresholds: per-place thresholds from negative statistics
  - evaluate: closed-set or open-set P/R/F1 evaluation
  - evaluate_vysotska: same but with Vysotska's per-query thresholds
"""

import os
import pickle
from glob import glob

import numpy as np

from experiments.online_place_discovery import OnlinePlaceDiscovery
from experiments.vysotska_threshold import VysotskaDaptiveThreshold


def load_cached_descriptors(cache_dir, n):
    """Load n descriptors from cache. Returns (N, D) float32 array."""
    descs = []
    for i in range(n):
        path = os.path.join(cache_dir, f"img_{i}_descriptor.pkl")
        with open(path, "rb") as f:
            d = pickle.load(f)
            if isinstance(d, dict):
                d = d["descriptor"]
            descs.append(d)
    return np.array(descs)


def count_cached(cache_dir):
    """Count available cached descriptors."""
    return len(glob(os.path.join(cache_dir, "img_*_descriptor.pkl")))


def discover_places(ref_descs, min_place_size=3, hysteresis=2, filter_n_cap=10):
    """Run online place discovery. Returns list of lists of frame indices."""
    discoverer = OnlinePlaceDiscovery(
        min_place_size=min_place_size, hysteresis=hysteresis,
        filter_n_cap=filter_n_cap
    )
    for i in range(len(ref_descs)):
        discoverer.process_frame(ref_descs[i], i, verbose=False)
    return discoverer.places


def compute_thresholds(ref_descs, places, method="filter_n", cap=10):
    """
    Per-place thresholds from negative statistics.
    method: "filter_n" or "mean_bad"
    Returns dict: place_idx -> threshold value
    """
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


def compute_place_stats(ref_descs, places, cap=10):
    """Compute detailed per-place statistics. Returns list of dicts."""
    stats = []
    for p_idx in range(len(places)):
        target = places[p_idx]
        other = [f for i, p in enumerate(places) if i != p_idx for f in p]
        if not other or len(target) < 2:
            stats.append({"mean_good": 0, "mean_bad": 0, "std_bad": 0,
                          "filter_n": 0, "threshold": 0, "size": len(target)})
            continue
        neg_sims = ref_descs[target] @ ref_descs[other].T
        per_img = neg_sims.mean(axis=1)
        mean_bad = float(per_img.mean())
        std_bad = float(per_img.std()) if len(per_img) > 1 else 0.1
        pos_sims = ref_descs[target] @ ref_descs[target].T
        np.fill_diagonal(pos_sims, 0)
        mean_good = float(pos_sims.sum(axis=1).mean() / max(len(target) - 1, 1))
        raw_fn = (mean_good - mean_bad) / max(std_bad, 1e-8)
        fn = max(0, min(np.floor(raw_fn), cap))
        stats.append({
            "mean_good": mean_good, "mean_bad": mean_bad, "std_bad": std_bad,
            "filter_n": int(fn), "raw_fn": raw_fn,
            "threshold": mean_bad + fn * std_bad, "size": len(target),
            "frames": f"{min(target)}-{max(target)}"
        })
    return stats


def evaluate(query_descs, ref_descs, places, thresholds, n_genuine=None):
    """
    Evaluate recognition performance.

    If n_genuine is None: closed-set (all queries are genuine).
    If n_genuine < len(query_descs): open-set (first n_genuine are genuine, rest are distractors).

    Returns dict with P, R, F1, dist_rej (if open-set), TP, FP, FN, TN.
    """
    S = query_descs @ ref_descs.T
    n_query = len(query_descs)
    n_places = len(places)
    if n_genuine is None:
        n_genuine = n_query

    frame_to_place = {}
    for p_idx, frames in enumerate(places):
        for f in frames:
            frame_to_place[f] = p_idx

    place_scores = np.zeros((n_query, n_places))
    for p_idx, frames in enumerate(places):
        place_scores[:, p_idx] = S[:, frames].mean(axis=1)

    TP, FP, FN, TN = 0, 0, 0, 0
    for q in range(n_query):
        is_genuine = q < n_genuine
        gt = frame_to_place.get(q, -1) if is_genuine else -1
        scores = place_scores[q].copy()
        for p in range(n_places):
            if scores[p] < thresholds.get(p, -np.inf):
                scores[p] = -np.inf
        pred = -1 if np.all(scores == -np.inf) else int(np.argmax(scores))
        if pred == -1:
            TN += 1 if gt == -1 else 0
            FN += 1 if gt != -1 else 0
        else:
            TP += 1 if (is_genuine and pred == gt) else 0
            FP += 1 if not (is_genuine and pred == gt) else 0

    P = TP / max(TP + FP, 1) * 100
    R = TP / max(TP + FN, 1) * 100
    F1 = 2 * P * R / max(P + R, 1e-8)
    n_dist = n_query - n_genuine
    dist_rej = TN / max(n_dist, 1) * 100 if n_dist > 0 else 0
    return {"P": P, "R": R, "F1": F1, "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "dist_rej": dist_rej}


def evaluate_vysotska(query_descs, ref_descs, places, n_genuine=None, patch_size=20):
    """Same as evaluate() but using Vysotska's per-query GMM thresholds."""
    S = query_descs @ ref_descs.T
    n_query = len(query_descs)
    n_places = len(places)
    if n_genuine is None:
        n_genuine = n_query

    vysotska = VysotskaDaptiveThreshold(patch_size=patch_size)
    pq_thresh, _ = vysotska.compute_thresholds(S)

    frame_to_place = {}
    for p_idx, frames in enumerate(places):
        for f in frames:
            frame_to_place[f] = p_idx

    place_scores = np.zeros((n_query, n_places))
    for p_idx, frames in enumerate(places):
        place_scores[:, p_idx] = S[:, frames].mean(axis=1)

    TP, FP, FN, TN = 0, 0, 0, 0
    for q in range(n_query):
        is_genuine = q < n_genuine
        gt = frame_to_place.get(q, -1) if is_genuine else -1
        scores = place_scores[q].copy()
        for p in range(n_places):
            if scores[p] < pq_thresh[q]:
                scores[p] = -np.inf
        pred = -1 if np.all(scores == -np.inf) else int(np.argmax(scores))
        if pred == -1:
            TN += 1 if gt == -1 else 0
            FN += 1 if gt != -1 else 0
        else:
            TP += 1 if (is_genuine and pred == gt) else 0
            FP += 1 if not (is_genuine and pred == gt) else 0

    P = TP / max(TP + FP, 1) * 100
    R = TP / max(TP + FN, 1) * 100
    F1 = 2 * P * R / max(P + R, 1e-8)
    n_dist = n_query - n_genuine
    dist_rej = TN / max(n_dist, 1) * 100 if n_dist > 0 else 0
    return {"P": P, "R": R, "F1": F1, "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "dist_rej": dist_rej}
