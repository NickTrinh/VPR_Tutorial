"""
Sequential Place Discovery using Online Negative Statistics.

Walks through a sequential image stream and discovers place boundaries
incrementally, using the negative statistics method from the RCC 2025 paper
applied online.

Algorithm:
  1. Bootstrap: find first significant dip in consecutive similarity to
     split the sequence into initial places P1 and P2.
  2. For each subsequent frame T:
     - Compute similarity of T to images in current place N
     - Compute threshold θ_N from negative statistics (mean similarity of
       place N's images to all images in places P1..P(N-1))
     - If score(T, place_N) > θ_N: T belongs to place N
     - Else: start new place N+1
  3. Two scoring variants: max (best-matching frame) vs mean (average).
  4. Two negative stats variants: current-place-only vs all-places.

Usage:
    python sequential_place_discovery.py \
        --condition day_left \
        --output_dir results/visualizations/place_discovery
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
from scipy.signal import argrelmin

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import normalize_l2


# ──────────────────────────────────────────────────────────────────────────────
# Feature loading (reuse cache from explore_sequential_similarity.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_features(cache_dir, n_images):
    """Load pre-extracted features from cache."""
    descriptors = []
    for i in range(n_images):
        path = os.path.join(cache_dir, f"img_{i}_descriptor.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing cached descriptor {path}. "
                "Run explore_sequential_similarity.py first to extract features."
            )
        with open(path, "rb") as f:
            d = pickle.load(f)
        desc = d["descriptor"].reshape(-1).astype(np.float32)
        descriptors.append(desc)
    return np.stack(descriptors, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Core algorithm
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap(S, min_place_size=3):
    """
    Find the first two places using the first significant dip in
    consecutive similarity.

    Returns (boundary_idx, ) where place 1 = [0, boundary_idx) and
    place 2 starts at boundary_idx.
    """
    N = S.shape[0]
    csim = np.array([S[t, t + 1] for t in range(N - 1)])

    # Find local minima with some smoothing
    mins = argrelmin(csim, order=3)[0]
    if len(mins) == 0:
        # Fallback: just split in half
        return N // 2

    # Pick the first minimum that is significantly below the running mean
    # Use the mean of the first few frames as baseline
    for m in mins:
        if m < min_place_size:
            continue
        baseline = csim[:m].mean()
        baseline_std = csim[:m].std() if m > 2 else 0.1
        if csim[m] < baseline - 1.5 * baseline_std:
            return m + 1  # boundary is at m+1 (first frame of new place)

    # Fallback: use the deepest minimum
    valid = mins[mins >= min_place_size]
    if len(valid) > 0:
        return valid[np.argmin(csim[valid])] + 1
    return mins[0] + 1


def compute_negative_threshold(S, places, place_idx, mode="current_only"):
    """
    Compute threshold θ for a given place using negative statistics.

    For each image in the target place, compute its mean similarity to
    all images in OTHER places. θ = mean of these mean-negative-scores.

    Args:
        S: full similarity matrix
        places: list of lists, where places[i] = list of frame indices
        place_idx: which place to compute threshold for
        mode: "current_only" — compute negatives only for current place
              "all_places" — compute negatives for all places, return
                             the threshold for the current place

    Returns:
        θ (float) for the specified place
    """
    target_frames = places[place_idx]
    other_frames = []
    for i, p in enumerate(places):
        if i != place_idx:
            other_frames.extend(p)

    if not other_frames or not target_frames:
        return 0.0  # can't compute, return permissive threshold

    other_frames = np.array(other_frames)
    mean_neg_scores = []
    for t in target_frames:
        # Similarity of frame t to all frames in other places
        neg_sims = S[t, other_frames]
        mean_neg_scores.append(neg_sims.mean())

    return np.mean(mean_neg_scores)


def compute_all_thresholds(S, places):
    """Compute negative-stats threshold for every place."""
    thresholds = {}
    for i in range(len(places)):
        thresholds[i] = compute_negative_threshold(S, places, i, mode="all_places")
    return thresholds


def score_frame_vs_place(S, frame_idx, place_frames, method="max"):
    """
    Compute how well frame T matches a place.

    method="max":  max similarity to any frame in the place
    method="mean": mean similarity to all frames in the place
    """
    sims = S[frame_idx, place_frames]
    if method == "max":
        return sims.max()
    else:
        return sims.mean()


def discover_places(S, scoring="max", neg_stats_mode="current_only",
                    min_place_size=3, verbose=True):
    """
    Main sequential place discovery algorithm.

    Args:
        S: NxN similarity matrix
        scoring: "max" or "mean" — how to score a frame against a place
        neg_stats_mode: "current_only" or "all_places"
        min_place_size: minimum frames before a place can be "left"
        verbose: print progress

    Returns:
        places: list of lists of frame indices
        history: list of dicts with per-frame decision info
    """
    N = S.shape[0]

    # Step 1: Bootstrap — find first two places
    split = bootstrap(S, min_place_size=min_place_size)
    places = [list(range(0, split))]  # Place 0

    if verbose:
        print(f"  Bootstrap: Place 0 = frames [0, {split})")

    # Now we need to find where Place 1 ends to have two places
    # Start Place 1 and process frame by frame
    # But we need N >= 2 places for negative stats.
    # So: build Place 1 with at least min_place_size frames, then start the loop.
    place1_start = split
    place1_end = min(place1_start + min_place_size, N)
    places.append(list(range(place1_start, place1_end)))

    if verbose:
        print(f"  Bootstrap: Place 1 = frames [{place1_start}, {place1_end})")

    # Step 2: Process remaining frames
    history = []
    current_place_idx = 1

    for T in range(place1_end, N):
        # Compute threshold for current place
        if neg_stats_mode == "current_only":
            theta = compute_negative_threshold(S, places, current_place_idx)
        else:
            # Recompute for all places, use current place's threshold
            theta = compute_negative_threshold(S, places, current_place_idx)

        # Score frame T against current place
        score = score_frame_vs_place(
            S, T, np.array(places[current_place_idx]), method=scoring
        )

        # Decision
        in_current = score > theta

        history.append({
            "frame": T,
            "place": current_place_idx,
            "score": score,
            "theta": theta,
            "decision": "stay" if in_current else "new_place",
        })

        if in_current:
            # T belongs to current place
            places[current_place_idx].append(T)
        else:
            # Start new place
            current_place_idx += 1
            places.append([T])
            if verbose:
                print(f"  Frame {T}: NEW place {current_place_idx} "
                      f"(score={score:.3f} ≤ θ={theta:.3f})")

    if verbose:
        print(f"\n  Discovered {len(places)} places total")
        for i, p in enumerate(places):
            print(f"    Place {i}: frames [{p[0]}..{p[-1]}], "
                  f"size={len(p)}")

    return places, history


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_discovered_places(S, places, history, scoring, neg_mode, output_dir):
    """Plot the similarity matrix with discovered place boundaries overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Left: similarity matrix with place boundaries
    ax = axes[0]
    ax.imshow(S, cmap="hot", vmin=0, vmax=1, aspect="auto")

    # Draw place boundaries
    colors = plt.cm.Set2(np.linspace(0, 1, min(len(places), 8)))
    for i, place in enumerate(places):
        start, end = place[0], place[-1]
        color = colors[i % len(colors)]
        # Draw rectangle
        rect = plt.Rectangle((start - 0.5, start - 0.5),
                              end - start + 1, end - start + 1,
                              linewidth=2, edgecolor=color,
                              facecolor="none", linestyle="-")
        ax.add_patch(rect)
        # Vertical boundary lines
        if i > 0:
            ax.axvline(x=start - 0.5, color="white", linewidth=1, alpha=0.8)
            ax.axhline(y=start - 0.5, color="white", linewidth=1, alpha=0.8)

    ax.set_xlabel("Frame j")
    ax.set_ylabel("Frame i")
    ax.set_title(f"Similarity matrix with discovered places\n"
                 f"scoring={scoring}, neg_stats={neg_mode}")

    # Right: score and threshold over time
    ax2 = axes[1]
    if history:
        frames = [h["frame"] for h in history]
        scores = [h["score"] for h in history]
        thetas = [h["theta"] for h in history]
        decisions = [h["decision"] for h in history]

        ax2.plot(frames, scores, color="#2980b9", linewidth=1.0,
                 label=f"score ({scoring})", alpha=0.8)
        ax2.plot(frames, thetas, color="#e74c3c", linewidth=1.5,
                 label="θ (neg stats threshold)", linestyle="--")

        # Mark new-place events
        new_place_frames = [h["frame"] for h in history
                           if h["decision"] == "new_place"]
        new_place_scores = [h["score"] for h in history
                           if h["decision"] == "new_place"]
        ax2.scatter(new_place_frames, new_place_scores, color="#e74c3c",
                    zorder=5, s=50, label="New place detected")

        # Shade place regions
        all_boundaries = [places[0][0]] + [p[0] for p in places[1:]] + [S.shape[0]]
        for i in range(len(all_boundaries) - 1):
            color = colors[i % len(colors)]
            ax2.axvspan(all_boundaries[i], all_boundaries[i + 1],
                        alpha=0.1, color=color)

    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("Similarity / Threshold")
    ax2.set_title(f"Per-frame score vs threshold\n"
                  f"Red dots = boundary detections")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    fname = f"places_{scoring}_{neg_mode}.png"
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_comparison(all_results, S, output_dir):
    """Side-by-side comparison of all 4 variants."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (key, (places, history)) in enumerate(all_results.items()):
        scoring, neg_mode = key
        ax = axes[idx]

        ax.imshow(S, cmap="hot", vmin=0, vmax=1, aspect="auto")

        colors = plt.cm.Set2(np.linspace(0, 1, min(len(places), 8)))
        for i, place in enumerate(places):
            start, end = place[0], place[-1]
            color = colors[i % len(colors)]
            rect = plt.Rectangle((start - 0.5, start - 0.5),
                                  end - start + 1, end - start + 1,
                                  linewidth=2, edgecolor=color,
                                  facecolor="none")
            ax.add_patch(rect)
            if i > 0:
                ax.axvline(x=start - 0.5, color="white", linewidth=1, alpha=0.7)
                ax.axhline(y=start - 0.5, color="white", linewidth=1, alpha=0.7)

        ax.set_title(f"scoring={scoring}, neg_stats={neg_mode}\n"
                     f"{len(places)} places discovered",
                     fontsize=11)
        ax.set_xlabel("Frame j")
        ax.set_ylabel("Frame i")

    fig.suptitle("Comparison of place discovery variants\n"
                 "Boxes = discovered places overlaid on self-similarity matrix",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_place_summary(places, img_paths, output_dir, label=""):
    """Show first, middle, last frame from each discovered place."""
    n_places = len(places)
    cols = min(n_places, 10)
    rows = 3  # first, mid, last

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 1.5))
    if cols == 1:
        axes = axes.reshape(-1, 1)

    for pi in range(min(n_places, cols)):
        frames = places[pi]
        samples = [frames[0], frames[len(frames) // 2], frames[-1]]
        for ri, t in enumerate(samples):
            from PIL import Image as PILImage
            img = PILImage.open(img_paths[t]).resize((160, 90))
            axes[ri, pi].imshow(np.array(img))
            axes[ri, pi].set_title(f"t={t}", fontsize=7)
            axes[ri, pi].axis("off")
        axes[0, pi].set_title(f"P{pi} (n={len(frames)})\nt={samples[0]}",
                              fontsize=7, fontweight="bold")

    row_labels = ["First", "Middle", "Last"]
    for ri in range(rows):
        axes[ri, 0].set_ylabel(row_labels[ri], fontsize=9, rotation=0,
                               ha="right", va="center")

    fig.suptitle(f"Discovered places — {label}\n"
                 f"First / middle / last frame from each place",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    safe_label = label.replace(" ", "_").replace(",", "")
    path = os.path.join(output_dir, f"place_summary_{safe_label}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="images/GardensPoint")
    p.add_argument("--condition", default="day_left")
    p.add_argument("--descriptor", default="eigenplaces")
    p.add_argument("--min_place_size", type=int, default=3)
    p.add_argument("--output_dir",
                   default="results/visualizations/place_discovery")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load images and features
    img_dir = os.path.join(args.data_dir, args.condition)
    img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    N = len(img_paths)
    print(f"Found {N} images in {img_dir}")

    cache_dir = os.path.join("cache", "GardensPoint", args.condition,
                             args.descriptor)
    features = load_features(cache_dir, N)
    print(f"Loaded features: {features.shape}")

    # Compute similarity matrix
    F = normalize_l2(features)
    S = F @ F.T
    print(f"Similarity matrix: {S.shape}")

    # Run all 4 variants
    variants = [
        ("max", "current_only"),
        ("mean", "current_only"),
        ("max", "all_places"),
        ("mean", "all_places"),
    ]

    all_results = {}
    for scoring, neg_mode in variants:
        print(f"\n{'='*60}")
        print(f"Running: scoring={scoring}, neg_stats={neg_mode}")
        print(f"{'='*60}")
        places, history = discover_places(
            S, scoring=scoring, neg_stats_mode=neg_mode,
            min_place_size=args.min_place_size
        )
        all_results[(scoring, neg_mode)] = (places, history)

        # Per-variant plots
        plot_discovered_places(S, places, history, scoring, neg_mode,
                               args.output_dir)
        plot_place_summary(places, img_paths, args.output_dir,
                           label=f"{scoring} {neg_mode}")

    # Comparison plot
    plot_comparison(all_results, S, args.output_dir)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
