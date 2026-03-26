"""
Dynamic Place Recognition: Discover + Recognize + Create New Places.

Extends the discovery-then-recognition pipeline:
  1. Discover places on reference condition
  2. Compute per-place thresholds
  3. Query with a different condition:
     - If query matches an existing place -> recognition (accept)
     - If query doesn't match any place -> create a new place
  4. Evaluate: how well does the system classify known places
     AND discover genuinely new ones?

Usage:
    python experiments/dynamic_place_recognition.py
    python experiments/dynamic_place_recognition.py --with_distractors
"""

import os
import sys
import argparse
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from experiments.experiment_utils import load_cached_descriptors, discover_places
from experiments.online_place_discovery import OnlineFeatureExtractor, OnlinePlaceDiscovery


def compute_place_stats(ref_descs, places):
    """Compute per-place stats from reference descriptors."""
    stats = {}
    for p_idx in range(len(places)):
        target = places[p_idx]
        other = [f for i, p in enumerate(places) if i != p_idx for f in p]

        if not other or len(target) < 2:
            stats[p_idx] = {
                "mean_bad": 0.0, "std_bad": 0.1, "mean_good": 0.5,
                "filter_n": 1.0, "theta": 0.0
            }
            continue

        neg_sims = ref_descs[target] @ ref_descs[other].T
        per_img_mean_bad = neg_sims.mean(axis=1)
        mean_bad = float(per_img_mean_bad.mean())
        std_bad = float(per_img_mean_bad.std()) if len(per_img_mean_bad) > 1 else 0.1

        pos_sims = ref_descs[target] @ ref_descs[target].T
        np.fill_diagonal(pos_sims, 0)
        per_img_mean_good = pos_sims.sum(axis=1) / max(len(target) - 1, 1)
        mean_good = float(per_img_mean_good.mean())

        if std_bad < 1e-8:
            filter_n = 1.0
        else:
            filter_n = float(np.floor((mean_good - mean_bad) / std_bad))
        filter_n = max(0, min(filter_n, 10))

        theta = mean_bad + filter_n * std_bad

        stats[p_idx] = {
            "mean_bad": mean_bad, "std_bad": std_bad,
            "mean_good": mean_good, "filter_n": filter_n,
            "theta": theta
        }
    return stats


# ─── Dynamic Place Recognition ────────────────────────────────────────────────

class DynamicPlaceRecognizer:
    """
    Recognizes known places and creates new ones for unrecognized queries.

    Maintains a growing database of places with per-place thresholds.
    When a query doesn't match any existing place, it accumulates in a
    "pending new place" buffer. Once enough consecutive rejections occur,
    a new place is created.
    """

    def __init__(self, ref_descs, places, stats, new_place_min_size=3,
                 new_place_sim_threshold=0.5):
        """
        Args:
            ref_descs: reference descriptors (n_ref, dim)
            places: list of lists of frame indices (from discovery)
            stats: per-place statistics dict
            new_place_min_size: min queries to form a new place
            new_place_sim_threshold: min similarity between pending queries
                                     to be considered same new place
        """
        self.ref_descs = ref_descs
        self.places = [list(p) for p in places]  # copy
        self.stats = dict(stats)  # copy
        self.n_original_places = len(places)

        self.new_place_min_size = new_place_min_size
        self.new_place_sim_threshold = new_place_sim_threshold

        # Pending buffer for potential new place
        self.pending_descs = []
        self.pending_query_indices = []

        # All descriptors (ref + new place descriptors)
        self.all_descs = list(ref_descs)

        # Track decisions
        self.decisions = []

    def _score_query(self, query_desc):
        """Compute per-place scores for a query."""
        scores = {}
        for p_idx, frames in enumerate(self.places):
            place_descs = np.array([self.all_descs[f] for f in frames])
            sim = query_desc @ place_descs.T
            scores[p_idx] = float(sim.mean())
        return scores

    def _create_new_place(self):
        """Create a new place from pending descriptors."""
        new_p_idx = len(self.places)
        new_frames = []

        for desc in self.pending_descs:
            frame_idx = len(self.all_descs)
            self.all_descs.append(desc)
            new_frames.append(frame_idx)

        self.places.append(new_frames)

        # Compute stats for the new place
        all_descs_arr = np.array(self.all_descs)
        target = new_frames
        other = [f for i, p in enumerate(self.places) if i != new_p_idx for f in p]

        if other and len(target) >= 2:
            neg_sims = all_descs_arr[target] @ all_descs_arr[other].T
            per_img_mean_bad = neg_sims.mean(axis=1)
            mean_bad = float(per_img_mean_bad.mean())
            std_bad = float(per_img_mean_bad.std()) if len(per_img_mean_bad) > 1 else 0.1

            pos_sims = all_descs_arr[target] @ all_descs_arr[target].T
            np.fill_diagonal(pos_sims, 0)
            per_img_mean_good = pos_sims.sum(axis=1) / max(len(target) - 1, 1)
            mean_good = float(per_img_mean_good.mean())

            if std_bad < 1e-8:
                filter_n = 1.0
            else:
                filter_n = float(np.floor((mean_good - mean_bad) / std_bad))
            filter_n = max(0, min(filter_n, 10))

            self.stats[new_p_idx] = {
                "mean_bad": mean_bad, "std_bad": std_bad,
                "mean_good": mean_good, "filter_n": filter_n,
                "theta": mean_bad + filter_n * std_bad
            }
        else:
            self.stats[new_p_idx] = {
                "mean_bad": 0.0, "std_bad": 0.1, "mean_good": 0.5,
                "filter_n": 1.0, "theta": 0.1
            }

        query_indices = list(self.pending_query_indices)
        self.pending_descs = []
        self.pending_query_indices = []

        return new_p_idx, query_indices

    def process_query(self, query_desc, query_idx):
        """
        Process a single query.

        Returns:
            dict with decision info
        """
        scores = self._score_query(query_desc)

        # Find best place above threshold
        best_place = None
        best_score = -np.inf

        for p_idx, score in scores.items():
            theta = self.stats[p_idx]["theta"]
            if score >= theta and score > best_score:
                best_place = p_idx
                best_score = score

        if best_place is not None:
            # Recognized — flush pending buffer if any
            new_place_created = None
            if len(self.pending_descs) >= self.new_place_min_size:
                new_place_created = self._create_new_place()
            else:
                self.pending_descs = []
                self.pending_query_indices = []

            decision = {
                "query_idx": query_idx,
                "decision": "recognized",
                "place": best_place,
                "score": best_score,
                "is_original_place": best_place < self.n_original_places,
                "new_place_created": new_place_created,
            }
        else:
            # Not recognized — add to pending
            # Check if this query is similar enough to pending queries
            if self.pending_descs:
                pending_arr = np.array(self.pending_descs)
                sims = query_desc @ pending_arr.T
                avg_sim = float(sims) if np.ndim(sims) == 0 else float(sims.mean())
                if avg_sim < self.new_place_sim_threshold:
                    # Too different from pending — flush and start new pending
                    if len(self.pending_descs) >= self.new_place_min_size:
                        new_place_created = self._create_new_place()
                    else:
                        self.pending_descs = []
                        self.pending_query_indices = []
                        new_place_created = None

            self.pending_descs.append(query_desc)
            self.pending_query_indices.append(query_idx)

            new_place_created = None
            if len(self.pending_descs) >= self.new_place_min_size:
                new_place_created = self._create_new_place()

            decision = {
                "query_idx": query_idx,
                "decision": "new_place" if new_place_created else "pending",
                "place": new_place_created[0] if new_place_created else None,
                "score": max(scores.values()) if scores else 0,
                "is_original_place": False,
                "new_place_created": new_place_created,
            }

        self.decisions.append(decision)
        return decision

    def flush_pending(self):
        """Create a new place from any remaining pending queries."""
        if len(self.pending_descs) >= self.new_place_min_size:
            return self._create_new_place()
        return None


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_dynamic(decisions, places_original, n_genuine, n_distractor,
                     frame_to_place):
    """
    Evaluate dynamic recognition.

    For genuine queries (query_idx < n_genuine):
      - Recognized into correct original place -> TP
      - Recognized into wrong original place -> FP
      - Recognized into new place -> FP (should have matched original)
      - Pending/unrecognized -> FN

    For distractor queries (query_idx >= n_genuine):
      - Recognized into any original place -> FP
      - Assigned to a NEW place -> TN (correctly identified as new)
      - Pending/unrecognized -> TN (correctly rejected)
    """
    TP, FP, FN, TN = 0, 0, 0, 0
    new_places_created = 0
    distractor_new_places = 0

    for d in decisions:
        q_idx = d["query_idx"]
        is_genuine = q_idx < n_genuine
        is_distractor = q_idx >= n_genuine

        if d["new_place_created"]:
            new_places_created += 1

        if is_genuine:
            gt_place = frame_to_place.get(q_idx, -1)
            if d["decision"] == "recognized":
                if d["place"] == gt_place:
                    TP += 1
                else:
                    FP += 1
            elif d["decision"] == "new_place":
                # Genuine query put into new place — it should have matched original
                FN += 1
            else:  # pending
                FN += 1
        else:  # distractor
            if d["decision"] == "recognized":
                if d["is_original_place"]:
                    FP += 1  # distractor accepted into original place
                else:
                    TN += 1  # distractor put into new place (correctly separated)
            elif d["decision"] == "new_place":
                TN += 1  # distractor created new place (correctly identified as new)
                distractor_new_places += 1
            else:  # pending
                TN += 1  # not yet assigned, effectively rejected

    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "Precision": precision, "Recall": recall, "F1": f1,
        "new_places_created": new_places_created,
        "distractor_new_places": distractor_new_places,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_distractors", action="store_true",
                        help="Mix in SFU distractor queries")
    parser.add_argument("--n_genuine", type=int, default=100)
    parser.add_argument("--output_dir", default="results/paper_figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading cached descriptors...")
    ref_descs = load_cached_descriptors("cache/GardensPoint/day_left/eigenplaces", 200)
    query_descs = load_cached_descriptors("cache/GardensPoint/day_right/eigenplaces", 200)

    # Discover places
    print("Discovering places on reference...")
    places = discover_places(ref_descs)
    stats = compute_place_stats(ref_descs, places)
    print(f"  {len(places)} places discovered")

    # Build frame-to-place mapping
    frame_to_place = {}
    for p_idx, frames in enumerate(places):
        for f in frames:
            frame_to_place[f] = p_idx

    # ── Test 1: Genuine queries only (closed-set + dynamic) ──
    print("\n=== Test 1: Genuine queries only ===")
    recognizer = DynamicPlaceRecognizer(
        ref_descs, places, stats,
        new_place_min_size=3, new_place_sim_threshold=0.3
    )

    n_genuine = min(args.n_genuine, len(query_descs))
    for i in range(n_genuine):
        recognizer.process_query(query_descs[i], i)
    recognizer.flush_pending()

    results = evaluate_dynamic(
        recognizer.decisions, places, n_genuine, 0, frame_to_place
    )
    print(f"  P={results['Precision']:.1f}%  R={results['Recall']:.1f}%  F1={results['F1']:.1f}%")
    print(f"  TP={results['TP']}  FP={results['FP']}  FN={results['FN']}  TN={results['TN']}")
    print(f"  New places created: {results['new_places_created']}")
    print(f"  Total places now: {len(recognizer.places)} "
          f"(original: {recognizer.n_original_places})")

    if not args.with_distractors:
        print("\nRun with --with_distractors for open-set test")
        return

    # ── Test 2: Mixed genuine + distractors ──
    print("\n=== Test 2: Genuine + SFU Distractors (dynamic) ===")
    n_sfu = len(glob("cache/SFU/dry/eigenplaces/img_*_descriptor.pkl"))
    sfu_descs = load_cached_descriptors("cache/SFU/dry/eigenplaces", n_sfu)
    n_dist = min(100, n_sfu)

    recognizer2 = DynamicPlaceRecognizer(
        ref_descs, places, stats,
        new_place_min_size=3, new_place_sim_threshold=0.3
    )

    # Interleave genuine and distractor queries (more realistic)
    genuine_indices = list(range(n_genuine))
    distractor_indices = list(range(n_genuine, n_genuine + n_dist))
    all_queries = np.vstack([query_descs[:n_genuine], sfu_descs[:n_dist]])

    # Process in order
    for i in range(len(all_queries)):
        recognizer2.process_query(all_queries[i], i)
    recognizer2.flush_pending()

    results2 = evaluate_dynamic(
        recognizer2.decisions, places, n_genuine, n_dist, frame_to_place
    )
    print(f"  P={results2['Precision']:.1f}%  R={results2['Recall']:.1f}%  F1={results2['F1']:.1f}%")
    print(f"  TP={results2['TP']}  FP={results2['FP']}  FN={results2['FN']}  TN={results2['TN']}")
    print(f"  New places created: {results2['new_places_created']}")
    print(f"  Distractor new places: {results2['distractor_new_places']}")
    print(f"  Total places now: {len(recognizer2.places)} "
          f"(original: {recognizer2.n_original_places})")

    # ── Test 3: Compare all methods on same mixed query set ──
    print("\n=== Comparison: Static vs Dynamic (with distractors) ===")

    # Static filter_n (just rejects, no new places)
    thresh_filter_n = {p: stats[p]["theta"] for p in range(len(places))}
    S = all_queries @ ref_descs.T
    n_places = len(places)
    place_scores = np.zeros((len(all_queries), n_places))
    for p_idx, frames in enumerate(places):
        place_scores[:, p_idx] = S[:, frames].mean(axis=1)

    static_TP, static_FP, static_FN, static_TN = 0, 0, 0, 0
    for q_idx in range(len(all_queries)):
        is_genuine = q_idx < n_genuine
        gt_place = frame_to_place.get(q_idx, -1) if is_genuine else -1
        scores = place_scores[q_idx].copy()
        for p_idx in range(n_places):
            if scores[p_idx] < thresh_filter_n[p_idx]:
                scores[p_idx] = -np.inf
        if np.all(scores == -np.inf):
            pred = -1
        else:
            pred = int(np.argmax(scores))

        if pred == -1:
            if gt_place == -1:
                static_TN += 1
            else:
                static_FN += 1
        else:
            if is_genuine and pred == gt_place:
                static_TP += 1
            else:
                static_FP += 1

    sp = static_TP / (static_TP + static_FP) * 100 if (static_TP + static_FP) > 0 else 0
    sr = static_TP / (static_TP + static_FN) * 100 if (static_TP + static_FN) > 0 else 0
    sf1 = 2 * sp * sr / (sp + sr) if (sp + sr) > 0 else 0

    print(f"\n  {'Method':<30} {'P':>6} {'R':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'New':>4}")
    print(f"  {'-'*80}")
    print(f"  {'Static filter_n (reject only)':<30} {sp:>5.1f}% {sr:>5.1f}% {sf1:>5.1f}% "
          f"{static_TP:>4} {static_FP:>4} {static_FN:>4} {static_TN:>4} {'0':>4}")
    print(f"  {'Dynamic filter_n (new places)':<30} {results2['Precision']:>5.1f}% {results2['Recall']:>5.1f}% "
          f"{results2['F1']:>5.1f}% {results2['TP']:>4} {results2['FP']:>4} {results2['FN']:>4} "
          f"{results2['TN']:>4} {results2['new_places_created']:>4}")

    # ── Visualize decisions ──
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Top: decision timeline
    ax = axes[0]
    for d in recognizer2.decisions:
        q = d["query_idx"]
        is_genuine = q < n_genuine
        color = "#27ae60" if d["decision"] == "recognized" and is_genuine else \
                "#e74c3c" if d["decision"] == "recognized" and not is_genuine else \
                "#3498db" if d["decision"] == "new_place" else "#95a5a6"
        marker = "o" if d["decision"] == "recognized" else "s" if d["decision"] == "new_place" else "x"
        place = d["place"] if d["place"] is not None else -1
        ax.scatter(q, place, c=color, marker=marker, s=20, alpha=0.7, zorder=3)

    ax.axvline(n_genuine - 0.5, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.text(n_genuine / 2, ax.get_ylim()[1] * 0.95, "Genuine queries",
            ha="center", fontsize=9, color="#27ae60")
    ax.text(n_genuine + n_dist / 2, ax.get_ylim()[1] * 0.95, "Distractors",
            ha="center", fontsize=9, color="#e74c3c")
    ax.axhline(recognizer2.n_original_places - 0.5, color="gray", linewidth=1,
               linestyle=":", alpha=0.5)
    ax.text(-5, recognizer2.n_original_places + 0.5, "New places -->",
            fontsize=8, color="#3498db", va="bottom")
    ax.set_xlabel("Query index", fontsize=10)
    ax.set_ylabel("Assigned place", fontsize=10)
    ax.set_title("Dynamic Place Recognition: Decision Timeline", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2)

    # Bottom: comparison bars
    ax2 = axes[1]
    methods = ["Static filter_n\n(reject only)", "Dynamic filter_n\n(create new places)"]
    f1_vals = [sf1, results2["F1"]]
    colors = ["#e74c3c", "#3498db"]
    bars = ax2.bar(range(2), f1_vals, color=colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, f1_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}%",
                 ha="center", fontsize=11, fontweight="bold")
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(methods, fontsize=10)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("F1 Score (%)", fontsize=11)
    ax2.set_title("Static Rejection vs Dynamic Place Creation", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    path = os.path.join(args.output_dir, "fig10_dynamic_recognition.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
