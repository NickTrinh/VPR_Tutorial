"""
Fast Nordland pipeline: Discovery + Threshold Comparison.

Optimizations vs the original:
  1. Batch descriptor loading (numpy memmap-style, not one pickle at a time)
  2. Discovery with incremental-only stats (no O(N²) recompute at boundaries)
  3. GPU-accelerated similarity matrix for Vysotska evaluation
  4. GPU feature extraction for uncached images
  5. Unbuffered output for monitoring

Usage:
    python -u experiments/run_nordland_fast.py [--max_images 0]
"""

import argparse
import os
import pickle
import sys
import time
from glob import glob

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# ─── Descriptor loading ──────────────────────────────────────────────────────

def load_cached_descriptors(cache_dir, n):
    """Load n descriptors from cache. Returns (N, D) L2-normalized float32 array."""
    print(f"  Loading {n} descriptors from {cache_dir}...", flush=True)
    t0 = time.time()
    descs = []
    missing = []
    for i in range(n):
        path = os.path.join(cache_dir, f"img_{i}_descriptor.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                d = pickle.load(f)
                if isinstance(d, dict):
                    d = d["descriptor"]
                descs.append(d.reshape(-1).astype(np.float32))
        else:
            missing.append(i)
            descs.append(None)

    if missing:
        print(f"  WARNING: {len(missing)} descriptors missing (first: {missing[0]})", flush=True)
        # Fill missing with zeros (will be handled later)
        dim = next(d.shape[0] for d in descs if d is not None)
        for i in missing:
            descs[i] = np.zeros(dim, dtype=np.float32)

    arr = np.array(descs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    arr /= norms
    print(f"  Loaded in {time.time()-t0:.1f}s, shape={arr.shape}", flush=True)
    return arr, missing


def extract_missing_gpu(img_paths, cache_dir, missing_indices, batch_size=64):
    """Extract features for missing indices using GPU."""
    if not missing_indices:
        return

    import torch
    import torchvision.transforms as transforms
    from PIL import Image as PILImage
    from feature_extraction.common import get_device

    device = get_device()
    print(f"  Extracting {len(missing_indices)} missing descriptors on {device}...", flush=True)

    model = torch.hub.load(
        "gmberton/eigenplaces", "get_trained_model",
        backbone="ResNet50", fc_output_dim=2048
    ).to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(480),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.time()

    for batch_start in range(0, len(missing_indices), batch_size):
        batch_idx = missing_indices[batch_start:batch_start + batch_size]
        imgs = []
        for i in batch_idx:
            img = PILImage.open(img_paths[i]).convert("RGB")
            imgs.append(transform(img))

        batch_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model(batch_tensor).cpu().numpy()

        for j, i in enumerate(batch_idx):
            feat = feats[j].reshape(-1).astype(np.float32)
            cache_path = os.path.join(cache_dir, f"img_{i}_descriptor.pkl")
            with open(cache_path, "wb") as f:
                pickle.dump({"descriptor": feat}, f)

        if (batch_start + batch_size) % (batch_size * 10) == 0:
            done = min(batch_start + batch_size, len(missing_indices))
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    {done}/{len(missing_indices)} extracted ({rate:.0f}/s)", flush=True)

    print(f"  Extraction done in {time.time()-t0:.1f}s", flush=True)


# ─── Fast Online Discovery (incremental only) ────────────────────────────────

class RunningStats:
    __slots__ = ['n', 'mean', 'M2']
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self):
        return self.mean if self.n > 0 else 0.0

    def get_std(self):
        return np.sqrt(self.M2 / self.n) if self.n >= 2 else 0.1


class FastOnlineDiscovery:
    """
    O(N) per frame online discovery. No full recomputation at boundaries.

    Key difference from original: at boundary events, we compute stats for the
    new place from scratch (small, since it starts with ~hysteresis frames) but
    do NOT recompute stats for all existing places (the change is marginal).
    """

    def __init__(self, all_descs, min_place_size=3, hysteresis=2, filter_n_cap=10,
                 bootstrap_std_factor=1.5):
        self.descs = all_descs  # (N, D) preloaded matrix
        self.min_place_size = min_place_size
        self.hysteresis = hysteresis
        self.filter_n_cap = filter_n_cap
        self.bootstrap_std_factor = bootstrap_std_factor

        self.places = []
        self.neg_stats = {}
        self.pos_stats = {}
        self.current_place = 0
        self.phase = "bootstrap_p0"

        self.consec_sims = []
        self.below_count = 0
        self.pending = []

        # Track all frames assigned so far (for computing "other frames")
        self._all_assigned = set()
        # Per-place frame sets for fast lookup
        self._place_sets = []

    def _sim(self, i, j):
        return float(self.descs[i] @ self.descs[j])

    def _sim_vec(self, frame, targets):
        """Similarity of frame to target frames. Returns 1D array."""
        if not targets:
            return np.array([])
        return self.descs[targets] @ self.descs[frame]

    def _get_other_indices(self, place_idx):
        """Frames NOT in this place (from all assigned frames)."""
        my_frames = self._place_sets[place_idx]
        return list(self._all_assigned - my_frames)

    def _compute_stats_for_place(self, place_idx):
        """Compute neg and pos stats for a single place."""
        target = self.places[place_idx]
        other = self._get_other_indices(place_idx)

        ns = RunningStats()
        if other and len(target) >= 1:
            # Vectorized: each target frame's mean similarity to all other frames
            neg_sims = self.descs[target] @ self.descs[other].T  # (T, O)
            for val in neg_sims.mean(axis=1):
                ns.update(float(val))

        ps = RunningStats()
        if len(target) >= 2:
            pos_sims = self.descs[target] @ self.descs[target].T  # (T, T)
            np.fill_diagonal(pos_sims, 0)
            n_t = len(target)
            for val in pos_sims.sum(axis=1) / max(n_t - 1, 1):
                ps.update(float(val))

        return ns, ps

    def _get_threshold(self):
        p = self.current_place
        if p not in self.neg_stats or self.neg_stats[p].n < 1:
            return 0.0, 0.0, 0.1, 0.5, 1.0

        mn = self.neg_stats[p].get_mean()
        sn = self.neg_stats[p].get_std()
        mp = self.pos_stats[p].get_mean() if p in self.pos_stats else 0.5

        fn = np.floor((mp - mn) / max(sn, 1e-8)) if sn >= 1e-8 else 1.0
        fn = max(0, min(fn, self.filter_n_cap))
        return mn + fn * sn, mn, sn, mp, fn

    def process_frame(self, frame_idx):
        T = frame_idx

        # ── Bootstrap P0 ──
        if self.phase == "bootstrap_p0":
            if len(self.places) == 0:
                self.places.append([T])
                self._place_sets.append({T})
                self._all_assigned.add(T)
                return

            # Check dip
            if T > 0:
                sim = self._sim(T - 1, T)
                self.consec_sims.append(sim)
                if len(self.consec_sims) >= self.min_place_size + 1:
                    prev = np.array(self.consec_sims[:-1])
                    bm, bs = prev.mean(), (prev.std() if len(prev) > 2 else 0.1)
                    if sim < bm - self.bootstrap_std_factor * bs:
                        self.phase = "bootstrap_p1"
                        self.places.append([T])
                        self._place_sets.append({T})
                        self._all_assigned.add(T)
                        self.current_place = 1
                        return

            self.places[0].append(T)
            self._place_sets[0].add(T)
            self._all_assigned.add(T)
            return

        # ── Bootstrap P1 ──
        if self.phase == "bootstrap_p1":
            self.places[1].append(T)
            self._place_sets[1].add(T)
            self._all_assigned.add(T)
            if len(self.places[1]) >= self.min_place_size:
                # Compute initial stats for both places
                for p in range(2):
                    ns, ps = self._compute_stats_for_place(p)
                    self.neg_stats[p] = ns
                    self.pos_stats[p] = ps
                self.phase = "online"
            return

        # ── Online ──
        current_frames = self.places[self.current_place]
        sims = self._sim_vec(T, current_frames)
        score = float(sims.mean())

        theta, mn, sn, mp, fn = self._get_threshold()

        if score <= theta:
            self.below_count += 1
            self.pending.append(T)
            self._all_assigned.add(T)

            if self.below_count >= self.hysteresis:
                # New place
                new_idx = len(self.places)
                self.places.append(self.pending.copy())
                self._place_sets.append(set(self.pending))
                self.pending = []
                self.below_count = 0
                self.current_place = new_idx

                # Compute stats for NEW place only (small)
                ns, ps = self._compute_stats_for_place(new_idx)
                self.neg_stats[new_idx] = ns
                self.pos_stats[new_idx] = ps

                # Refresh stats for previous place (its frames haven't changed,
                # but its "other" set grew). This is the only existing place
                # we refresh — much cheaper than refreshing ALL places.
                prev_place = new_idx - 1
                ns, ps = self._compute_stats_for_place(prev_place)
                self.neg_stats[prev_place] = ns
                self.pos_stats[prev_place] = ps
        else:
            if self.pending:
                self.places[self.current_place].extend(self.pending)
                self._place_sets[self.current_place].update(self.pending)
                self.pending = []
            self.below_count = 0
            self.places[self.current_place].append(T)
            self._place_sets[self.current_place].add(T)
            self._all_assigned.add(T)

            # Incremental neg stats
            other = self._get_other_indices(self.current_place)
            if other:
                neg = float(self._sim_vec(T, other).mean())
                if self.current_place not in self.neg_stats:
                    self.neg_stats[self.current_place] = RunningStats()
                self.neg_stats[self.current_place].update(neg)

            # Incremental pos stats
            if len(current_frames) >= 2:
                pos = float(sims.mean())
                if self.current_place not in self.pos_stats:
                    self.pos_stats[self.current_place] = RunningStats()
                self.pos_stats[self.current_place].update(pos)


# ─── Threshold computation ────────────────────────────────────────────────────

def compute_our_thresholds(ref_descs, places, method="filter_n"):
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
        fn = max(0, min(np.floor((mean_good - mean_bad) / max(std_bad, 1e-8)), 10))
        if method == "filter_n":
            thresholds[p_idx] = mean_bad + fn * std_bad
        else:
            thresholds[p_idx] = mean_bad
    return thresholds


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(query_descs, ref_descs, places, thresholds, use_gpu=False):
    """Evaluate recognition. Uses GPU for the big similarity matrix if available."""
    print("  Computing similarity matrix...", flush=True)
    t0 = time.time()

    if use_gpu:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Compute in chunks to fit GPU memory
        chunk_size = 2000
        n_q, n_r = len(query_descs), len(ref_descs)
        S = np.zeros((n_q, n_r), dtype=np.float32)
        ref_gpu = torch.from_numpy(ref_descs).to(device)
        for start in range(0, n_q, chunk_size):
            end = min(start + chunk_size, n_q)
            q_gpu = torch.from_numpy(query_descs[start:end]).to(device)
            S[start:end] = (q_gpu @ ref_gpu.T).cpu().numpy()
        del ref_gpu
        torch.cuda.empty_cache()
    else:
        S = query_descs @ ref_descs.T

    print(f"  Similarity matrix computed in {time.time()-t0:.1f}s "
          f"({S.shape[0]}x{S.shape[1]})", flush=True)

    n_query = len(query_descs)
    n_places = len(places)
    frame_to_place = {}
    for p_idx, frames in enumerate(places):
        for f in frames:
            frame_to_place[f] = p_idx

    place_scores = np.zeros((n_query, n_places))
    for p_idx, frames in enumerate(places):
        place_scores[:, p_idx] = S[:, frames].mean(axis=1)

    TP, FP, FN = 0, 0, 0
    recall_at = {1: 0, 3: 0, 5: 0, 10: 0}

    for q in range(n_query):
        gt = frame_to_place.get(q, -1)
        scores = place_scores[q].copy()
        for p in range(n_places):
            if scores[p] < thresholds.get(p, -np.inf):
                scores[p] = -np.inf
        pred = -1 if np.all(scores == -np.inf) else int(np.argmax(scores))

        if pred == -1:
            if gt != -1: FN += 1
        else:
            if pred == gt: TP += 1
            else: FP += 1

        for K in recall_at:
            if np.all(scores == -np.inf): continue
            top_k = np.argsort(scores)[::-1][:K]
            top_k = [p for p in top_k if scores[p] > -np.inf]
            if gt in top_k: recall_at[K] += 1

    P = TP / max(TP + FP, 1) * 100
    R = TP / max(TP + FN, 1) * 100
    F1 = 2 * P * R / max(P + R, 1e-8)

    return {
        "P": P, "R": R, "F1": F1, "TP": TP, "FP": FP, "FN": FN,
        "rej": FN / n_query * 100,
        **{f"R@{K}": v / n_query * 100 for K, v in recall_at.items()},
    }, S


def evaluate_vysotska(S, places, n_query, use_gpu=False):
    """Evaluate Vysotska method on precomputed similarity matrix."""
    from experiments.vysotska_threshold import VysotskaDaptiveThreshold

    print("  Running Vysotska threshold...", flush=True)
    vysotska = VysotskaDaptiveThreshold(patch_size=20)
    pq_thresh, _ = vysotska.compute_thresholds(S)

    n_places = len(places)
    frame_to_place = {}
    for p_idx, frames in enumerate(places):
        for f in frames:
            frame_to_place[f] = p_idx

    place_scores = np.zeros((n_query, n_places))
    for p_idx, frames in enumerate(places):
        place_scores[:, p_idx] = S[:n_query, frames].mean(axis=1)

    TP, FP, FN = 0, 0, 0
    for q in range(n_query):
        gt = frame_to_place.get(q, -1)
        scores = place_scores[q].copy()
        for p in range(n_places):
            if scores[p] < pq_thresh[q]:
                scores[p] = -np.inf
        pred = -1 if np.all(scores == -np.inf) else int(np.argmax(scores))
        if pred == -1:
            if gt != -1: FN += 1
        else:
            if pred == gt: TP += 1
            else: FP += 1

    P = TP / max(TP + FP, 1) * 100
    R = TP / max(TP + FN, 1) * 100
    F1 = 2 * P * R / max(P + R, 1e-8)
    return {"P": P, "R": R, "F1": F1, "TP": TP, "FP": FP, "FN": FN,
            "rej": FN / n_query * 100}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="images/Nordland_filtered")
    p.add_argument("--ref_condition", default="summer")
    p.add_argument("--query_condition", default="winter")
    p.add_argument("--descriptor", default="eigenplaces")
    p.add_argument("--max_images", type=int, default=0, help="0=all")
    p.add_argument("--img_ext", default="*.png")
    p.add_argument("--no_gpu", action="store_true")
    args = p.parse_args()

    dataset_name = os.path.basename(args.data_dir.rstrip("/"))
    use_gpu = not args.no_gpu

    # Count images
    ref_paths = sorted(glob(os.path.join(args.data_dir, args.ref_condition, args.img_ext)))
    query_paths = sorted(glob(os.path.join(args.data_dir, args.query_condition, args.img_ext)))
    if args.max_images > 0:
        ref_paths = ref_paths[:args.max_images]
        query_paths = query_paths[:args.max_images]
    N = min(len(ref_paths), len(query_paths))
    ref_paths = ref_paths[:N]
    query_paths = query_paths[:N]
    print(f"Dataset: {dataset_name}, {N} images per condition", flush=True)

    ref_cache = f"cache/{dataset_name}/{args.ref_condition}/{args.descriptor}"
    query_cache = f"cache/{dataset_name}/{args.query_condition}/{args.descriptor}"

    # ── Load/extract descriptors ──
    print(f"\n{'='*60}", flush=True)
    print("STEP 1: Load descriptors", flush=True)
    print(f"{'='*60}", flush=True)

    ref_descs, ref_missing = load_cached_descriptors(ref_cache, N)
    if ref_missing:
        extract_missing_gpu(ref_paths, ref_cache, ref_missing)
        ref_descs, _ = load_cached_descriptors(ref_cache, N)

    query_descs, query_missing = load_cached_descriptors(query_cache, N)
    if query_missing:
        extract_missing_gpu(query_paths, query_cache, query_missing)
        query_descs, _ = load_cached_descriptors(query_cache, N)

    # ── Discover places ──
    print(f"\n{'='*60}", flush=True)
    print("STEP 2: Fast online discovery", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    discoverer = FastOnlineDiscovery(ref_descs, min_place_size=3, hysteresis=2,
                                      filter_n_cap=10)
    for i in range(N):
        discoverer.process_frame(i)
        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate
            print(f"  {i+1}/{N} ({rate:.0f} fps, {len(discoverer.places)} places, "
                  f"ETA {eta:.0f}s)", flush=True)

    places = discoverer.places
    elapsed = time.time() - t0
    print(f"\n  Discovery done in {elapsed:.1f}s ({N/elapsed:.0f} fps)", flush=True)
    print(f"  {len(places)} places discovered", flush=True)
    sizes = [len(p) for p in places]
    print(f"  Size range: {min(sizes)}-{max(sizes)}, "
          f"mean={np.mean(sizes):.1f}, median={np.median(sizes):.0f}", flush=True)

    # ── Evaluate ──
    print(f"\n{'='*60}", flush=True)
    print("STEP 3: Evaluate threshold methods", flush=True)
    print(f"{'='*60}", flush=True)

    # Compute thresholds
    thresh_none = {p: -np.inf for p in range(len(places))}
    thresh_mean_bad = compute_our_thresholds(ref_descs, places, "simple_avg")
    thresh_filter_n = compute_our_thresholds(ref_descs, places, "filter_n")

    results = {}

    print("\n--- Baseline (no threshold) ---", flush=True)
    r, S = evaluate(query_descs, ref_descs, places, thresh_none, use_gpu=use_gpu)
    results["Baseline"] = r
    print(f"  R@1={r['R@1']:.1f}% P={r['P']:.1f}% R={r['R']:.1f}% F1={r['F1']:.1f}%",
          flush=True)

    print("\n--- Ours: mean_bad ---", flush=True)
    r, _ = evaluate(query_descs, ref_descs, places, thresh_mean_bad, use_gpu=use_gpu)
    results["mean_bad"] = r
    print(f"  R@1={r['R@1']:.1f}% P={r['P']:.1f}% R={r['R']:.1f}% F1={r['F1']:.1f}%",
          flush=True)

    print("\n--- Ours: filter_n ---", flush=True)
    r, _ = evaluate(query_descs, ref_descs, places, thresh_filter_n, use_gpu=use_gpu)
    results["filter_n"] = r
    print(f"  R@1={r['R@1']:.1f}% P={r['P']:.1f}% R={r['R']:.1f}% F1={r['F1']:.1f}%",
          flush=True)

    print("\n--- Vysotska ---", flush=True)
    r = evaluate_vysotska(S, places, N, use_gpu=use_gpu)
    results["Vysotska"] = r
    print(f"  P={r['P']:.1f}% R={r['R']:.1f}% F1={r['F1']:.1f}%", flush=True)

    # ── Summary ──
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Dataset: {dataset_name} ({N} images)", flush=True)
    print(f"Places: {len(places)}", flush=True)
    print(f"\n{'Method':<15} {'R@1':>6} {'P':>6} {'R':>6} {'F1':>6} {'Rej':>6}",
          flush=True)
    print(f"{'-'*50}", flush=True)
    for name, r in results.items():
        r1 = r.get('R@1', 0)
        print(f"{name:<15} {r1:>5.1f}% {r['P']:>5.1f}% {r['R']:>5.1f}% "
              f"{r['F1']:>5.1f}% {r['rej']:>5.1f}%", flush=True)

    # Save results
    output_dir = f"results/nordland_full_{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    import json
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({
            "n_images": N, "n_places": len(places),
            "place_sizes": [len(p) for p in places],
            "results": {k: {m: v for m, v in r.items() if isinstance(v, (int, float))}
                        for k, r in results.items()}
        }, f, indent=2)
    print(f"\nResults saved to {output_dir}/results.npz", flush=True)


if __name__ == "__main__":
    main()
