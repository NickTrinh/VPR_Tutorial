"""
Truly Online Sequential Place Discovery.

Processes one frame at a time — no precomputed similarity matrix.
Each frame is:
  1. Feature-extracted (or loaded from cache)
  2. Compared to all previously stored descriptors
  3. Scored against the current place
  4. Thresholded using incrementally-updated negative statistics

The algorithm never looks ahead — it only uses information available
up to the current frame.

Usage:
    python online_place_discovery.py \
        --condition day_left \
        --output_dir results/visualizations/online_discovery
"""

import argparse
import os
import pickle
import sys
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import normalize_l2


# ──────────────────────────────────────────────────────────────────────────────
# Online feature extraction / cache
# ──────────────────────────────────────────────────────────────────────────────

class OnlineFeatureExtractor:
    """Extracts (or loads from cache) one descriptor at a time."""

    def __init__(self, cache_dir, device=None):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.model = None
        self.device = device
        self._transform = None

    def _load_model(self):
        """Lazy-load the model only if we actually need to extract."""
        if self.model is not None:
            return
        import torch
        import torchvision.transforms as transforms
        from feature_extraction.common import get_device

        self.device = self.device or get_device()
        self.model = torch.hub.load(
            "gmberton/eigenplaces", "get_trained_model",
            backbone="ResNet50", fc_output_dim=2048
        ).to(self.device).eval()
        self._transform = transforms.Compose([
            transforms.Resize(480),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def get_descriptor(self, img_path, idx):
        """
        Get L2-normalized descriptor for a single image.
        Returns a 1D float32 numpy array of shape (2048,).
        """
        cache_path = os.path.join(self.cache_dir, f"img_{idx}_descriptor.pkl")

        # Try cache first
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                d = pickle.load(f)
            desc = d["descriptor"].reshape(-1).astype(np.float32)
            norm = np.linalg.norm(desc)
            if norm > 0:
                desc /= norm
            return desc

        # Extract fresh
        import torch
        self._load_model()
        img = PILImage.open(img_path).convert("RGB")
        tensor = self._transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor).cpu().numpy().reshape(-1).astype(np.float32)

        # Cache it
        with open(cache_path, "wb") as f:
            pickle.dump({"descriptor": feat}, f)

        norm = np.linalg.norm(feat)
        if norm > 0:
            feat /= norm
        return feat


# ──────────────────────────────────────────────────────────────────────────────
# Running statistics tracker
# ──────────────────────────────────────────────────────────────────────────────

class RunningStats:
    """
    Incrementally compute mean and std of a stream of values.
    Uses Welford's online algorithm for numerical stability.
    """
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # sum of squared deviations

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self):
        return self.mean if self.n > 0 else 0.0

    def get_std(self):
        if self.n < 2:
            return 0.1  # fallback
        return np.sqrt(self.M2 / self.n)

    def __repr__(self):
        return f"RunningStats(n={self.n}, mean={self.get_mean():.4f}, std={self.get_std():.4f})"


# ──────────────────────────────────────────────────────────────────────────────
# Online place discovery
# ──────────────────────────────────────────────────────────────────────────────

class OnlinePlaceDiscovery:
    """
    Truly online sequential place discovery.

    State maintained:
      - descriptors: list of L2-normed descriptors seen so far
      - places: list of lists of frame indices
      - neg_stats[p]: RunningStats for per-image mean-negative-score of place p
      - pos_stats[p]: RunningStats for within-place similarity of place p
    """

    def __init__(self, min_place_size=3, hysteresis=2, filter_n_cap=10,
                 bootstrap_std_factor=1.5):
        self.min_place_size = min_place_size
        self.hysteresis = hysteresis
        self.filter_n_cap = filter_n_cap
        self.bootstrap_std_factor = bootstrap_std_factor

        # State
        self.descriptors = []       # list of 1D numpy arrays
        self.places = []            # list of lists of frame indices
        self.neg_stats = {}         # place_idx -> RunningStats (mean-neg per image)
        self.pos_stats = {}         # place_idx -> RunningStats (within-place sim)
        self.current_place = 0
        self.phase = "bootstrap_p0" # bootstrap_p0 -> bootstrap_p1 -> online
        self.history = []

        # Bootstrap state
        self.prev_sim = None        # consecutive similarity for bootstrap
        self.consec_sims = []       # running list of consecutive sims
        self.below_count = 0        # hysteresis counter
        self.pending = []           # frames in hysteresis limbo

    def _sim(self, i, j):
        """Cosine similarity between stored descriptors i and j."""
        return float(np.dot(self.descriptors[i], self.descriptors[j]))

    def _sim_to_frame(self, frame_idx, target_frames):
        """Similarity of one frame to a set of frames. Returns array."""
        desc = self.descriptors[frame_idx]
        sims = np.array([np.dot(desc, self.descriptors[t]) for t in target_frames])
        return sims

    def _mean_sim_to_others(self, frame_idx, other_frames):
        """Mean similarity of frame to a set of other frames."""
        if len(other_frames) == 0:
            return 0.0
        sims = self._sim_to_frame(frame_idx, other_frames)
        return float(sims.mean())

    def _get_other_frames(self, place_idx):
        """All frames NOT in the given place."""
        other = []
        for i, p in enumerate(self.places):
            if i != place_idx:
                other.extend(p)
        return other

    def _compute_neg_stats_for_place(self, place_idx):
        """
        Recompute neg stats for a place from scratch.
        Called when a new place is created (previous places' stats change).
        """
        stats = RunningStats()
        target_frames = self.places[place_idx]
        other_frames = self._get_other_frames(place_idx)
        if not other_frames:
            return stats
        for t in target_frames:
            mean_neg = self._mean_sim_to_others(t, other_frames)
            stats.update(mean_neg)
        return stats

    def _compute_pos_stats_for_place(self, place_idx):
        """Recompute pos stats for a place from scratch."""
        stats = RunningStats()
        frames = self.places[place_idx]
        if len(frames) < 2:
            return stats
        for i, t in enumerate(frames):
            others = frames[:i] + frames[i+1:]
            if others:
                mean_pos = self._mean_sim_to_others(t, others)
                stats.update(mean_pos)
        return stats

    def _update_neg_stat_incremental(self, place_idx, new_frame):
        """
        Incrementally update neg stats for place_idx when new_frame
        is added to a DIFFERENT place.

        Each image in place_idx now has one more negative neighbor,
        so its mean-neg changes. But recomputing all is expensive.

        Compromise: update the stat for the new frame (if it's in this
        place's "others"), and note that existing per-image means shift
        slightly. For accuracy, we recompute from scratch when a new
        place boundary is created (infrequent event).
        """
        # This is called when new_frame is NOT in place_idx.
        # We just need to know the neg stat for place_idx.
        # Full recompute is fine since boundaries are infrequent.
        pass  # handled by full recompute at boundary events

    def _get_threshold(self):
        """
        Compute threshold for current place using filter_n formula:
          θ = mean_neg + min(filter_n, cap) × std_neg
          filter_n = floor((mean_pos - mean_neg) / std_neg)
        """
        p = self.current_place
        if p not in self.neg_stats or self.neg_stats[p].n < 1:
            return 0.0  # permissive during bootstrap

        mean_neg = self.neg_stats[p].get_mean()
        std_neg = self.neg_stats[p].get_std()
        mean_pos = self.pos_stats[p].get_mean() if p in self.pos_stats else 0.5

        if std_neg < 1e-8:
            filter_n = 1.0
        else:
            filter_n = np.floor((mean_pos - mean_neg) / std_neg)

        filter_n = min(filter_n, self.filter_n_cap)
        filter_n = max(filter_n, 0)  # don't go negative

        theta = mean_neg + filter_n * std_neg
        return theta, mean_neg, std_neg, mean_pos, filter_n

    def _recompute_all_stats(self):
        """Recompute neg and pos stats for all places from scratch.
        Called when a new place boundary is created."""
        for p_idx in range(len(self.places)):
            self.neg_stats[p_idx] = self._compute_neg_stats_for_place(p_idx)
            self.pos_stats[p_idx] = self._compute_pos_stats_for_place(p_idx)

    def _bootstrap_check_dip(self, T):
        """
        Check if frame T is a bootstrap boundary (first significant dip).
        Uses running mean/std of consecutive similarities.
        """
        if T == 0:
            return False

        sim = self._sim(T - 1, T)
        self.consec_sims.append(sim)

        if len(self.consec_sims) < self.min_place_size + 1:
            return False

        # Check if current consecutive sim is significantly below running mean
        prev_sims = np.array(self.consec_sims[:-1])
        baseline_mean = prev_sims.mean()
        baseline_std = prev_sims.std() if len(prev_sims) > 2 else 0.1

        return sim < baseline_mean - self.bootstrap_std_factor * baseline_std

    def process_frame(self, descriptor, frame_idx, verbose=True):
        """
        Process a single new frame. This is the main entry point.

        Args:
            descriptor: L2-normalized 1D numpy array
            frame_idx: integer frame index
            verbose: print decisions

        Returns:
            dict with decision info
        """
        self.descriptors.append(descriptor)
        T = frame_idx

        # ── Phase: building Place 0 ──────────────────────────────────
        if self.phase == "bootstrap_p0":
            if len(self.places) == 0:
                self.places.append([T])
                return {"frame": T, "decision": "bootstrap_p0", "place": 0}

            # Check for dip
            if self._bootstrap_check_dip(T):
                # This frame starts Place 1
                self.phase = "bootstrap_p1"
                self.places.append([T])
                self.current_place = 1
                if verbose:
                    print(f"  Frame {T}: Bootstrap split — "
                          f"Place 0 = [0..{T-1}] ({T} frames), "
                          f"Place 1 starts")
                return {"frame": T, "decision": "bootstrap_split", "place": 1}
            else:
                self.places[0].append(T)
                return {"frame": T, "decision": "bootstrap_p0", "place": 0}

        # ── Phase: building Place 1 (need min_place_size) ────────────
        if self.phase == "bootstrap_p1":
            self.places[1].append(T)
            if len(self.places[1]) >= self.min_place_size:
                # We now have 2 places — compute initial stats and go online
                self._recompute_all_stats()
                self.phase = "online"
                if verbose:
                    print(f"  Frame {T}: Bootstrap complete — "
                          f"Place 1 has {len(self.places[1])} frames. "
                          f"Going online.")
                    for p_idx in range(len(self.places)):
                        ns = self.neg_stats[p_idx]
                        ps = self.pos_stats[p_idx]
                        print(f"    Place {p_idx}: neg(μ={ns.get_mean():.3f}, "
                              f"σ={ns.get_std():.3f}), "
                              f"pos(μ={ps.get_mean():.3f})")
            return {"frame": T, "decision": "bootstrap_p1", "place": 1}

        # ── Phase: online ────────────────────────────────────────────
        # Score frame T against current place (mean similarity)
        current_frames = self.places[self.current_place]
        score = float(self._sim_to_frame(T, current_frames).mean())

        # Get threshold
        thresh_info = self._get_threshold()
        theta, mean_neg, std_neg, mean_pos, filter_n = thresh_info

        below = score <= theta

        record = {
            "frame": T,
            "place": self.current_place,
            "score": score,
            "theta": theta,
            "mean_neg": mean_neg,
            "std_neg": std_neg,
            "mean_pos": mean_pos,
            "filter_n": filter_n,
        }

        if below:
            self.below_count += 1
            self.pending.append(T)

            if self.below_count >= self.hysteresis:
                # NEW PLACE confirmed
                new_place_idx = len(self.places)
                self.places.append(self.pending.copy())
                self.pending = []
                self.below_count = 0
                self.current_place = new_place_idx

                # Recompute all stats (new boundary changes everyone's negatives)
                self._recompute_all_stats()

                record["decision"] = "new_place"
                if verbose:
                    ns = self.neg_stats.get(new_place_idx)
                    print(f"  Frame {T}: NEW place {new_place_idx} "
                          f"(score={score:.3f} ≤ θ={theta:.3f}, "
                          f"filter_n={filter_n:.0f})")
            else:
                record["decision"] = "pending"
        else:
            # Frame stays in current place
            if self.pending:
                # False alarm — pending frames go back to current place
                self.places[self.current_place].extend(self.pending)
                self.pending = []
            self.below_count = 0
            self.places[self.current_place].append(T)

            # Incrementally update pos stats for current place
            # (new frame added, so within-place similarities change)
            if len(current_frames) >= 2:
                new_pos = float(self._sim_to_frame(T, current_frames).mean())
                if self.current_place not in self.pos_stats:
                    self.pos_stats[self.current_place] = RunningStats()
                self.pos_stats[self.current_place].update(new_pos)

            # Incrementally update neg stats for current place
            # (new frame's mean-neg score)
            other_frames = self._get_other_frames(self.current_place)
            if other_frames:
                new_neg = self._mean_sim_to_others(T, other_frames)
                if self.current_place not in self.neg_stats:
                    self.neg_stats[self.current_place] = RunningStats()
                self.neg_stats[self.current_place].update(new_neg)

            record["decision"] = "stay"

        self.history.append(record)
        return record


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_online_results(discoverer, output_dir, img_paths=None):
    """Generate visualization of online discovery results."""
    places = discoverer.places
    history = discoverer.history
    descriptors = np.array(discoverer.descriptors)
    N = len(descriptors)

    # Reconstruct similarity matrix for visualization only
    # (not used by the algorithm — purely for the plot)
    S = descriptors @ descriptors.T

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # ── Top-left: similarity matrix with place boundaries ────────────
    ax = axes[0, 0]
    ax.imshow(S, cmap="hot", vmin=0, vmax=1, aspect="auto")
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(places), 1)))
    for i, place in enumerate(places):
        start, end = place[0], place[-1]
        rect = plt.Rectangle((start - 0.5, start - 0.5),
                              end - start + 1, end - start + 1,
                              linewidth=2, edgecolor=colors[i % len(colors)],
                              facecolor="none")
        ax.add_patch(rect)
        if i > 0:
            ax.axvline(x=start - 0.5, color="white", linewidth=1, alpha=0.7)
            ax.axhline(y=start - 0.5, color="white", linewidth=1, alpha=0.7)
    sizes = [len(p) for p in places]
    ax.set_title(f"Online discovery: {len(places)} places\nsizes={sizes}", fontsize=10)
    ax.set_xlabel("Frame j")
    ax.set_ylabel("Frame i")

    # ── Top-right: score vs threshold over time ──────────────────────
    ax2 = axes[0, 1]
    online_history = [h for h in history if "score" in h]
    if online_history:
        frames = [h["frame"] for h in online_history]
        scores = [h["score"] for h in online_history]
        thetas = [h["theta"] for h in online_history]

        ax2.plot(frames, scores, color="#2980b9", linewidth=1.0,
                 label="score (mean sim)", alpha=0.8)
        ax2.plot(frames, thetas, color="#e74c3c", linewidth=1.5,
                 label="threshold (θ)", linestyle="--")

        # Mark boundaries
        new_frames = [h["frame"] for h in online_history
                      if h["decision"] == "new_place"]
        new_scores = [h["score"] for h in online_history
                      if h["decision"] == "new_place"]
        ax2.scatter(new_frames, new_scores, color="#e74c3c",
                    zorder=5, s=50, label="New place")

        # Shade places
        boundaries = [places[0][0]] + [p[0] for p in places[1:]] + [N]
        for i in range(len(boundaries) - 1):
            ax2.axvspan(boundaries[i], boundaries[i + 1],
                        alpha=0.08, color=colors[i % len(colors)])

    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("Similarity / Threshold")
    ax2.set_title("Online score vs threshold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    # ── Bottom-left: filter_n over time ──────────────────────────────
    ax3 = axes[1, 0]
    if online_history and "filter_n" in online_history[0]:
        frames = [h["frame"] for h in online_history]
        fns = [h["filter_n"] for h in online_history]
        ax3.plot(frames, fns, color="#27ae60", linewidth=1.0, alpha=0.8)
        ax3.axhline(y=discoverer.filter_n_cap, color="gray",
                     linestyle=":", label=f"cap={discoverer.filter_n_cap}")

        boundaries = [places[0][0]] + [p[0] for p in places[1:]] + [N]
        for i in range(len(boundaries) - 1):
            ax3.axvspan(boundaries[i], boundaries[i + 1],
                        alpha=0.08, color=colors[i % len(colors)])

    ax3.set_xlabel("Frame index")
    ax3.set_ylabel("filter_n")
    ax3.set_title("Adaptive filter_n over time")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25)

    # ── Bottom-right: neg/pos stats over time ────────────────────────
    ax4 = axes[1, 1]
    if online_history and "mean_neg" in online_history[0]:
        frames = [h["frame"] for h in online_history]
        mean_negs = [h["mean_neg"] for h in online_history]
        mean_poss = [h["mean_pos"] for h in online_history]
        std_negs = [h["std_neg"] for h in online_history]

        ax4.plot(frames, mean_poss, color="#2980b9", linewidth=1.0,
                 label="mean_pos (within-place)", alpha=0.8)
        ax4.plot(frames, mean_negs, color="#e74c3c", linewidth=1.0,
                 label="mean_neg (cross-place)", alpha=0.8)
        ax4.fill_between(frames,
                         np.array(mean_negs) - np.array(std_negs),
                         np.array(mean_negs) + np.array(std_negs),
                         color="#e74c3c", alpha=0.15, label="±std_neg")

        boundaries = [places[0][0]] + [p[0] for p in places[1:]] + [N]
        for i in range(len(boundaries) - 1):
            ax4.axvspan(boundaries[i], boundaries[i + 1],
                        alpha=0.08, color=colors[i % len(colors)])

    ax4.set_xlabel("Frame index")
    ax4.set_ylabel("Similarity")
    ax4.set_title("Running negative & positive statistics")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.25)

    plt.suptitle("Truly Online Place Discovery\n"
                 "(no precomputed similarity matrix — frame-by-frame processing)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "online_discovery.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ── Place summary with sample images ─────────────────────────────
    if img_paths:
        n_places = len(places)
        cols = min(n_places, 12)
        rows = 3
        fig, axes_img = plt.subplots(rows, cols, figsize=(cols * 2, rows * 1.5))
        if cols == 1:
            axes_img = axes_img.reshape(-1, 1)

        for pi in range(min(n_places, cols)):
            frames = places[pi]
            samples = [frames[0], frames[len(frames) // 2], frames[-1]]
            for ri, t in enumerate(samples):
                img = PILImage.open(img_paths[t]).resize((160, 90))
                axes_img[ri, pi].imshow(np.array(img))
                axes_img[ri, pi].set_title(f"t={t}", fontsize=7)
                axes_img[ri, pi].axis("off")
            axes_img[0, pi].set_title(f"P{pi} (n={len(frames)})\nt={samples[0]}",
                                      fontsize=7, fontweight="bold")

        row_labels = ["First", "Mid", "Last"]
        for ri in range(rows):
            axes_img[ri, 0].set_ylabel(row_labels[ri], fontsize=9, rotation=0,
                                        ha="right", va="center")

        fig.suptitle("Online discovered places — sample frames",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(output_dir, "online_place_summary.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Truly online sequential place discovery")
    p.add_argument("--data_dir", default="images/GardensPoint")
    p.add_argument("--condition", default="day_left")
    p.add_argument("--descriptor", default="eigenplaces")
    p.add_argument("--min_place_size", type=int, default=3)
    p.add_argument("--hysteresis", type=int, default=2)
    p.add_argument("--filter_n_cap", type=int, default=10)
    p.add_argument("--max_images", type=int, default=200,
                   help="Limit number of images to process")
    p.add_argument("--output_dir",
                   default="results/visualizations/online_discovery")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load image paths
    img_dir = os.path.join(args.data_dir, args.condition)
    img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    if args.max_images > 0:
        img_paths = img_paths[:args.max_images]
    N = len(img_paths)
    print(f"Processing {N} images from {img_dir}")

    # Set up online extractor
    cache_dir = os.path.join("cache", "GardensPoint", args.condition,
                             args.descriptor)
    extractor = OnlineFeatureExtractor(cache_dir)

    # Set up online discoverer
    discoverer = OnlinePlaceDiscovery(
        min_place_size=args.min_place_size,
        hysteresis=args.hysteresis,
        filter_n_cap=args.filter_n_cap,
    )

    # Process frames one at a time
    print("\n--- Processing frames online ---")
    for i in range(N):
        desc = extractor.get_descriptor(img_paths[i], i)
        result = discoverer.process_frame(desc, i, verbose=True)

        # Progress every 50 frames
        if (i + 1) % 50 == 0:
            print(f"  ... processed {i + 1}/{N} frames, "
                  f"{len(discoverer.places)} places so far")

    # Summary
    print(f"\n{'='*60}")
    print(f"ONLINE DISCOVERY COMPLETE")
    print(f"{'='*60}")
    print(f"Frames processed: {N}")
    print(f"Places discovered: {len(discoverer.places)}")
    for i, p in enumerate(discoverer.places):
        print(f"  Place {i}: frames [{p[0]}..{p[-1]}], size={len(p)}")
    print(f"Filter_n cap: {args.filter_n_cap}")
    print(f"Hysteresis: {args.hysteresis}")

    # Visualize
    print("\n--- Generating plots ---")
    plot_online_results(discoverer, args.output_dir, img_paths)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
