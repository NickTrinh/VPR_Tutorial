"""
Sequential VPR experiment runner.

Parallels experiment_runner.py + test_runner.py but for sequential datasets
with online place discovery instead of predefined places.

Pipeline:
  1. Extract/load descriptors for ref and query conditions
  2. Discover places online from reference sequence
  3. Compute per-place thresholds from reference-only statistics
  4. Evaluate: closed-set (query only) and open-set (query + distractors)

Usage:
    python sequential_runner.py --dataset gardens_point --descriptor eigenplaces
    python sequential_runner.py --dataset gardens_point --descriptor dinov2-salad --ref-condition day_left --query-condition day_right
"""

import argparse
import os
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from glob import glob

from config import DatasetConfig, get_dataset_config, DATASETS
from data_utils import DatasetLoader
from utils import normalize_l2
from experiments.online_place_discovery import OnlinePlaceDiscovery
from experiments.vysotska_threshold import VysotskaDaptiveThreshold
from experiments.vysotska_sequence_matcher import sequence_match, evaluate_sequence_match


# ── Feature loading ──────────────────────────────────────────────────

def load_condition_descriptors(
    dataset_path: str,
    condition: str,
    descriptor_name: str,
    image_ext: str = "*.png",
    n: Optional[int] = None,
    feature_extractor=None,
    use_cache: bool = True,
    feature_extractor_factory=None,
) -> np.ndarray:
    """
    Load (or extract+cache) descriptors for one condition of a sequential dataset.

    Cache format: cache/<DatasetName>/<condition>/<descriptor>/img_{i}_descriptor.pkl
    This is the same format used by the experiment scripts.
    """
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    cache_dir = os.path.join("cache", dataset_name, condition, descriptor_name)

    # Find images
    img_dir = os.path.join(dataset_path, condition)
    patterns = [image_ext] if not image_ext.startswith("*.") else [image_ext]
    img_paths = sorted(glob(os.path.join(img_dir, image_ext)))
    if not img_paths:
        # Try common extensions
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            img_paths = sorted(glob(os.path.join(img_dir, ext)))
            if img_paths:
                break

    if n is not None:
        img_paths = img_paths[:n]

    total = len(img_paths)
    if total == 0:
        raise ValueError(f"No images found in {img_dir}")

    os.makedirs(cache_dir, exist_ok=True)

    # Check cache
    descs = []
    to_extract = []
    for i in range(total):
        cache_path = os.path.join(cache_dir, f"img_{i}_descriptor.pkl")
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                d = pickle.load(f)
                if isinstance(d, dict):
                    d = d["descriptor"]
                descs.append((i, d))
        else:
            to_extract.append((i, img_paths[i]))

    # Extract missing — lazily instantiate extractor if needed
    if to_extract and feature_extractor is None and feature_extractor_factory is not None:
        feature_extractor = feature_extractor_factory()
    if to_extract and feature_extractor is not None:
        print(f"  Extracting {len(to_extract)}/{total} descriptors for {condition}...")
        from PIL import Image
        batch_size = 32
        for start in range(0, len(to_extract), batch_size):
            batch = to_extract[start:start + batch_size]
            images = [np.array(Image.open(p)) for _, p in batch]
            feats = feature_extractor.compute_features(images)
            try:
                import torch
                if hasattr(feats, 'detach'):
                    feats = feats.detach().cpu().numpy()
            except ImportError:
                pass
            for j, (idx, _) in enumerate(batch):
                desc = feats[j].reshape(-1)
                descs.append((idx, desc))
                if use_cache:
                    cp = os.path.join(cache_dir, f"img_{idx}_descriptor.pkl")
                    with open(cp, "wb") as f:
                        pickle.dump({"descriptor": desc}, f)
    elif to_extract:
        raise RuntimeError(
            f"Need to extract {len(to_extract)} descriptors but no feature_extractor provided. "
            f"Run with GPU or pre-cache features."
        )

    # Sort by index and stack
    descs.sort(key=lambda x: x[0])
    arr = np.array([d.reshape(-1) for _, d in descs], dtype=np.float32)
    return normalize_l2(arr)


# ── Place discovery ──────────────────────────────────────────────────

def discover_places(ref_descs: np.ndarray, min_place_size: int = 3,
                    hysteresis: int = 2) -> List[List[int]]:
    """Run online place discovery on reference descriptors."""
    discoverer = OnlinePlaceDiscovery(
        min_place_size=min_place_size, hysteresis=hysteresis,
        filter_n_cap=10  # cap only used for boundary detection, not final thresholds
    )
    for i in range(len(ref_descs)):
        discoverer.process_frame(ref_descs[i], i, verbose=False)
    return discoverer.places


# ── Threshold computation ────────────────────────────────────────────

def compute_thresholds(ref_descs: np.ndarray, places: List[List[int]],
                       k: float = 2.0) -> Dict[int, float]:
    """
    Per-place z-score threshold: theta_k = mean_bad + k * std_bad.
    No dependency on positive (intra-place) statistics.
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
        thresholds[p_idx] = mean_bad + k * std_bad
    return thresholds


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate(query_descs: np.ndarray, ref_descs: np.ndarray,
             places: List[List[int]], thresholds: Dict[int, float],
             n_genuine: Optional[int] = None) -> Dict:
    """
    Place-level evaluation with filter-then-rank.

    If n_genuine < len(query_descs): first n_genuine are genuine, rest are distractors.
    Ground truth: query i maps to the place containing ref frame i (1:1 correspondence).
    """
    S = query_descs @ ref_descs.T
    n_query = len(query_descs)
    n_places = len(places)
    if n_genuine is None:
        n_genuine = n_query

    # Frame-to-place mapping
    frame_to_place = {}
    for p_idx, frames in enumerate(places):
        for f in frames:
            frame_to_place[f] = p_idx

    # Place-level scores
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


def evaluate_vysotska(query_descs: np.ndarray, ref_descs: np.ndarray,
                      places: List[List[int]],
                      n_genuine: Optional[int] = None, patch_size: int = 20) -> Dict:
    """Same as evaluate() but with Vysotska's per-query GMM thresholds."""
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


def evaluate_sequence_matcher(query_descs: np.ndarray, ref_descs: np.ndarray,
                              n_genuine: Optional[int] = None,
                              fanout: int = 3, tolerance: int = 1,
                              patch_size: int = 20) -> Dict:
    """
    Evaluate Vysotska's full pipeline: sequence matcher + adaptive threshold.

    Works for both closed-set (n_genuine=None) and open-set (n_genuine < n_query).
    In open-set, distractors appended after genuine queries should go through
    the hidden (non-matching) path.
    """
    S = query_descs @ ref_descs.T
    n_query = len(query_descs)
    if n_genuine is None:
        n_genuine = n_query

    # Compute adaptive threshold on the similarity matrix
    vysotska = VysotskaDaptiveThreshold(patch_size=patch_size)
    thresholds, _ = vysotska.compute_thresholds(S)
    median_thresh = float(np.median(thresholds))
    nmc = 1.0 - median_thresh  # convert similarity threshold to cost

    matches, all_path, path_real, path_hidden = sequence_match(
        S, non_matching_cost=nmc, fanout=fanout
    )

    # Evaluate: genuine queries matched correctly = TP, wrong = FP
    # Genuine queries unmatched (hidden) = FN
    # Distractor queries matched = FP, unmatched = TN
    TP, FP, FN, TN = 0, 0, 0, 0
    for q in range(n_query):
        is_genuine = q < n_genuine
        if q in matches:
            pred = matches[q]
            if is_genuine and abs(pred - q) <= tolerance:
                TP += 1
            else:
                FP += 1
        else:
            if is_genuine:
                FN += 1
            else:
                TN += 1

    P = TP / max(TP + FP, 1) * 100
    R = TP / max(TP + FN, 1) * 100
    F1 = 2 * P * R / max(P + R, 1e-8)
    n_dist = n_query - n_genuine
    dist_rej = TN / max(n_dist, 1) * 100 if n_dist > 0 else 0
    return {"P": P, "R": R, "F1": F1, "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "dist_rej": dist_rej, "n_matched": len(matches)}


# ── Main runner ──────────────────────────────────────────────────────

class SequentialVPRExperiment:
    """Run sequential VPR experiment with online place discovery."""

    def __init__(self, dataset_name: str, descriptor_name: str = "eigenplaces",
                 ref_condition: str = None, query_condition: str = None,
                 distractor_dataset: str = None, distractor_condition: str = None,
                 n_images: Optional[int] = None, k: float = 2.0,
                 use_cache: bool = True):
        self.dataset_config = get_dataset_config(dataset_name)
        self.descriptor_name = descriptor_name
        self.n_images = n_images
        self.k = k
        self.use_cache = use_cache

        # Determine conditions
        conditions = self.dataset_config.conditions or []
        if ref_condition:
            self.ref_condition = ref_condition
        elif len(conditions) >= 2:
            self.ref_condition = conditions[0]
        else:
            raise ValueError(f"Must specify --ref-condition for {dataset_name}")

        if query_condition:
            self.query_condition = query_condition
        elif len(conditions) >= 2:
            self.query_condition = conditions[1]
        else:
            raise ValueError(f"Must specify --query-condition for {dataset_name}")

        self.distractor_dataset = distractor_dataset
        self.distractor_condition = distractor_condition

        # Initialize feature extractor (lazy — only used if cache miss)
        self._feature_extractor = None

    @property
    def feature_extractor(self):
        if self._feature_extractor is None:
            from data_utils import DatasetLoader
            # Use DatasetLoader's init logic to get the right extractor
            dummy = DatasetLoader.__new__(DatasetLoader)
            self._feature_extractor = dummy._init_feature_extractor(self.descriptor_name)
        return self._feature_extractor

    def run(self) -> Dict:
        """Run full experiment. Returns results dict."""
        dataset_path = self.dataset_config.path
        ext = self.dataset_config.image_extension or "*.png"

        print(f"\n{'='*70}")
        print(f"SEQUENTIAL VPR EXPERIMENT")
        print(f"{'='*70}")
        print(f"Dataset:    {self.dataset_config.name}")
        print(f"Descriptor: {self.descriptor_name}")
        print(f"Reference:  {self.ref_condition}")
        print(f"Query:      {self.query_condition}")
        print(f"Threshold:  k={self.k}")

        # Load descriptors (pass None for extractor; only instantiate on cache miss)
        def _get_extractor():
            return self.feature_extractor

        print(f"\nLoading reference descriptors ({self.ref_condition})...")
        ref_descs = load_condition_descriptors(
            dataset_path, self.ref_condition, self.descriptor_name,
            image_ext=ext, n=self.n_images,
            feature_extractor=None, use_cache=self.use_cache,
            feature_extractor_factory=_get_extractor,
        )
        print(f"  Shape: {ref_descs.shape}")

        print(f"Loading query descriptors ({self.query_condition})...")
        query_descs = load_condition_descriptors(
            dataset_path, self.query_condition, self.descriptor_name,
            image_ext=ext, n=self.n_images,
            feature_extractor=None, use_cache=self.use_cache,
            feature_extractor_factory=_get_extractor,
        )
        print(f"  Shape: {query_descs.shape}")

        # Load distractors if specified
        dist_descs = None
        if self.distractor_dataset and self.distractor_condition:
            dist_config = get_dataset_config(self.distractor_dataset)
            print(f"Loading distractors ({self.distractor_dataset}/{self.distractor_condition})...")
            dist_descs = load_condition_descriptors(
                dist_config.path, self.distractor_condition, self.descriptor_name,
                image_ext=dist_config.image_extension or "*.jpg",
                feature_extractor=None, use_cache=self.use_cache,
                feature_extractor_factory=_get_extractor,
            )
            print(f"  Shape: {dist_descs.shape}")

        # Discover places
        print(f"\nDiscovering places from reference sequence...")
        places = discover_places(ref_descs)
        sizes = [len(p) for p in places]
        print(f"  {len(places)} places discovered (sizes: {min(sizes)}-{max(sizes)}, mean={np.mean(sizes):.1f})")

        # Compute thresholds
        thresholds_ours = compute_thresholds(ref_descs, places, k=self.k)
        no_thresh = {p: -np.inf for p in range(len(places))}

        results = {
            "dataset": self.dataset_config.name,
            "descriptor": self.descriptor_name,
            "ref_condition": self.ref_condition,
            "query_condition": self.query_condition,
            "n_ref": len(ref_descs),
            "n_query": len(query_descs),
            "n_places": len(places),
            "k": self.k,
        }

        # ── Closed-set ────────────────────────────────────────────────
        print(f"\n{'─'*70}")
        print("CLOSED-SET EVALUATION")
        print(f"{'─'*70}")

        r_base = evaluate(query_descs, ref_descs, places, no_thresh)
        r_ours = evaluate(query_descs, ref_descs, places, thresholds_ours)
        r_vyso = evaluate_vysotska(query_descs, ref_descs, places)
        r_sm = evaluate_sequence_matcher(query_descs, ref_descs)

        print(f"  {'Method':<25} {'Prec':>6} {'Rec':>6} {'F1':>6}")
        print(f"  {'─'*45}")
        for label, r in [("Baseline", r_base), (f"Ours (k={self.k})", r_ours),
                          ("Vysotska thresh", r_vyso), ("Vysotska seq.match", r_sm)]:
            print(f"  {label:<25} {r['P']:5.1f}% {r['R']:5.1f}% {r['F1']:5.1f}%")

        results["closed_set"] = {
            "baseline": r_base, "ours": r_ours, "vysotska": r_vyso,
            "vysotska_seq_match": r_sm,
        }

        # ── Open-set ─────────────────────────────────────────────────
        if dist_descs is not None:
            n_genuine = len(query_descs)
            n_dist = len(dist_descs)
            query_open = np.vstack([query_descs, dist_descs])

            print(f"\n{'─'*70}")
            print(f"OPEN-SET EVALUATION ({n_genuine} genuine + {n_dist} distractors)")
            print(f"{'─'*70}")

            r_base_o = evaluate(query_open, ref_descs, places, no_thresh, n_genuine=n_genuine)
            r_ours_o = evaluate(query_open, ref_descs, places, thresholds_ours, n_genuine=n_genuine)
            r_vyso_o = evaluate_vysotska(query_open, ref_descs, places, n_genuine=n_genuine)
            r_sm_o = evaluate_sequence_matcher(query_open, ref_descs, n_genuine=n_genuine)

            print(f"  {'Method':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Rej':>6}")
            print(f"  {'─'*51}")
            for label, r in [("Baseline", r_base_o), (f"Ours (k={self.k})", r_ours_o),
                              ("Vysotska thresh", r_vyso_o), ("Vysotska seq.match", r_sm_o)]:
                print(f"  {label:<25} {r['P']:5.1f}% {r['R']:5.1f}% {r['F1']:5.1f}% {r['dist_rej']:5.0f}%")

            results["open_set"] = {
                "n_distractors": n_dist,
                "distractor_source": f"{self.distractor_dataset}/{self.distractor_condition}",
                "baseline": r_base_o, "ours": r_ours_o, "vysotska": r_vyso_o,
                "vysotska_seq_match": r_sm_o,
            }

        # ── Save results ─────────────────────────────────────────────
        out_dir = os.path.join("results", self.dataset_config.name, self.descriptor_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "sequential_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # ── Summary ──────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"  Closed-set: Baseline F1={r_base['F1']:.1f}%, "
              f"Ours F1={r_ours['F1']:.1f}%, Vysotska thresh F1={r_vyso['F1']:.1f}%, "
              f"Vysotska seq.match F1={r_sm['F1']:.1f}%")
        if dist_descs is not None:
            print(f"  Open-set:   Baseline F1={r_base_o['F1']:.1f}%, "
                  f"Ours F1={r_ours_o['F1']:.1f}%, Vysotska thresh F1={r_vyso_o['F1']:.1f}%, "
                  f"Vysotska seq.match F1={r_sm_o['F1']:.1f}%")
        print(f"  Results saved to {out_path}")
        print(f"{'='*70}")

        return results


# ── Presets for common experiments ───────────────────────────────────

SEQUENTIAL_PRESETS = {
    "gardens_point": {
        "ref_condition": "day_left",
        "query_condition": "day_right",
        "distractor_dataset": "sfu",
        "distractor_condition": "dry",
    },
    "nordland_mini": {
        "ref_condition": "winter",
        "query_condition": "summer",
        "distractor_dataset": "gardens_point",
        "distractor_condition": "day_left",
        "n_images": 500,
    },
    "sfu": {
        "ref_condition": "dry",
        "query_condition": "jan",
    },
}


def run_sequential_experiment(dataset_name: str, descriptor_name: str = "eigenplaces",
                               k: float = 2.0, use_cache: bool = True,
                               **overrides) -> Dict:
    """Convenience function to run a sequential experiment with preset defaults."""
    preset = SEQUENTIAL_PRESETS.get(dataset_name, {}).copy()
    preset.update(overrides)

    exp = SequentialVPRExperiment(
        dataset_name=dataset_name,
        descriptor_name=descriptor_name,
        k=k,
        use_cache=use_cache,
        **preset
    )
    return exp.run()


def main():
    parser = argparse.ArgumentParser(description="Sequential VPR experiment runner")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name from config.py (e.g., gardens_point, nordland_mini)")
    parser.add_argument("--descriptor", default="eigenplaces",
                        choices=["eigenplaces", "cosplace", "alexnet", "dinov2-salad"],
                        help="Descriptor to use")
    parser.add_argument("--ref-condition", default=None,
                        help="Reference condition (default: from preset)")
    parser.add_argument("--query-condition", default=None,
                        help="Query condition (default: from preset)")
    parser.add_argument("--distractor-dataset", default=None,
                        help="Distractor dataset name (default: from preset)")
    parser.add_argument("--distractor-condition", default=None,
                        help="Distractor condition (default: from preset)")
    parser.add_argument("--n-images", type=int, default=None,
                        help="Number of images to use (default: all)")
    parser.add_argument("--k", type=float, default=2.0,
                        help="Threshold multiplier (default: 2.0)")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.ref_condition:
        overrides["ref_condition"] = args.ref_condition
    if args.query_condition:
        overrides["query_condition"] = args.query_condition
    if args.distractor_dataset:
        overrides["distractor_dataset"] = args.distractor_dataset
    if args.distractor_condition:
        overrides["distractor_condition"] = args.distractor_condition
    if args.n_images:
        overrides["n_images"] = args.n_images

    run_sequential_experiment(
        dataset_name=args.dataset,
        descriptor_name=args.descriptor,
        k=args.k,
        use_cache=not args.no_cache,
        **overrides
    )


if __name__ == "__main__":
    main()
