"""
Test each refinement individually against the baseline place discovery.

Baseline: mean scoring, current_only neg stats, no refinements (6 places).

Refinements tested one at a time:
  1. Sliding window (W=10) — only compare to last W frames in the place
  2. Min place size — merge places smaller than k frames into previous
  3. Top-k scoring (k=3) — mean of top-3 most similar frames
  4. Hysteresis (h=3) — require h consecutive frames below threshold

Usage:
    python test_refinements.py
"""

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
# Feature loading
# ──────────────────────────────────────────────────────────────────────────────

def load_features(cache_dir, n_images):
    descriptors = []
    for i in range(n_images):
        path = os.path.join(cache_dir, f"img_{i}_descriptor.pkl")
        with open(path, "rb") as f:
            d = pickle.load(f)
        desc = d["descriptor"].reshape(-1).astype(np.float32)
        descriptors.append(desc)
    return np.stack(descriptors, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap (shared across all variants)
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap(S, min_place_size=3):
    N = S.shape[0]
    csim = np.array([S[t, t + 1] for t in range(N - 1)])
    mins = argrelmin(csim, order=3)[0]
    if len(mins) == 0:
        return N // 2
    for m in mins:
        if m < min_place_size:
            continue
        baseline = csim[:m].mean()
        baseline_std = csim[:m].std() if m > 2 else 0.1
        if csim[m] < baseline - 1.5 * baseline_std:
            return m + 1
    valid = mins[mins >= min_place_size]
    if len(valid) > 0:
        return valid[np.argmin(csim[valid])] + 1
    return mins[0] + 1


# ──────────────────────────────────────────────────────────────────────────────
# Negative threshold computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_negative_threshold(S, places, place_idx):
    target_frames = places[place_idx]
    other_frames = []
    for i, p in enumerate(places):
        if i != place_idx:
            other_frames.extend(p)
    if not other_frames or not target_frames:
        return 0.0
    other_frames = np.array(other_frames)
    mean_neg_scores = []
    for t in target_frames:
        neg_sims = S[t, other_frames]
        mean_neg_scores.append(neg_sims.mean())
    return np.mean(mean_neg_scores)


def compute_negative_threshold_windowed(S, places, place_idx, window=10):
    """Same as above but only use last `window` frames of the target place."""
    target_frames = places[place_idx][-window:]  # only last W frames
    other_frames = []
    for i, p in enumerate(places):
        if i != place_idx:
            other_frames.extend(p)
    if not other_frames or not target_frames:
        return 0.0
    other_frames = np.array(other_frames)
    mean_neg_scores = []
    for t in target_frames:
        neg_sims = S[t, other_frames]
        mean_neg_scores.append(neg_sims.mean())
    return np.mean(mean_neg_scores)


# ──────────────────────────────────────────────────────────────────────────────
# Scoring functions
# ──────────────────────────────────────────────────────────────────────────────

def score_mean(S, frame_idx, place_frames):
    return S[frame_idx, place_frames].mean()


def score_mean_windowed(S, frame_idx, place_frames, window=10):
    """Mean similarity to only the last W frames in the place."""
    recent = place_frames[-window:]
    return S[frame_idx, recent].mean()


def score_topk(S, frame_idx, place_frames, k=3):
    """Mean of top-k most similar frames in the place."""
    sims = S[frame_idx, place_frames]
    if len(sims) <= k:
        return sims.mean()
    topk = np.sort(sims)[-k:]
    return topk.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Place discovery variants
# ──────────────────────────────────────────────────────────────────────────────

def discover_baseline(S, min_place_size=3):
    """Baseline: mean scoring, no refinements."""
    N = S.shape[0]
    split = bootstrap(S, min_place_size)
    places = [list(range(0, split))]
    place1_end = min(split + min_place_size, N)
    places.append(list(range(split, place1_end)))
    history = []
    current = 1

    for T in range(place1_end, N):
        theta = compute_negative_threshold(S, places, current)
        score = score_mean(S, T, np.array(places[current]))
        history.append({"frame": T, "score": score, "theta": theta,
                        "decision": "stay" if score > theta else "new_place"})
        if score > theta:
            places[current].append(T)
        else:
            current += 1
            places.append([T])

    return places, history


def discover_sliding_window(S, min_place_size=3, window=10):
    """Refinement 1: sliding window for both scoring and threshold."""
    N = S.shape[0]
    split = bootstrap(S, min_place_size)
    places = [list(range(0, split))]
    place1_end = min(split + min_place_size, N)
    places.append(list(range(split, place1_end)))
    history = []
    current = 1

    for T in range(place1_end, N):
        theta = compute_negative_threshold_windowed(S, places, current, window)
        score = score_mean_windowed(S, T, np.array(places[current]), window)
        history.append({"frame": T, "score": score, "theta": theta,
                        "decision": "stay" if score > theta else "new_place"})
        if score > theta:
            places[current].append(T)
        else:
            current += 1
            places.append([T])

    return places, history


def discover_min_place_merge(S, min_place_size=3, min_merge=3):
    """Refinement 2: run baseline, then merge small places into previous."""
    places, history = discover_baseline(S, min_place_size)

    # Post-process: merge small places
    merged = [places[0]]
    for p in places[1:]:
        if len(p) < min_merge:
            merged[-1].extend(p)
        else:
            merged.append(p)

    return merged, history


def discover_topk(S, min_place_size=3, k=3):
    """Refinement 3: top-k scoring instead of mean."""
    N = S.shape[0]
    split = bootstrap(S, min_place_size)
    places = [list(range(0, split))]
    place1_end = min(split + min_place_size, N)
    places.append(list(range(split, place1_end)))
    history = []
    current = 1

    for T in range(place1_end, N):
        theta = compute_negative_threshold(S, places, current)
        score = score_topk(S, T, np.array(places[current]), k)
        history.append({"frame": T, "score": score, "theta": theta,
                        "decision": "stay" if score > theta else "new_place"})
        if score > theta:
            places[current].append(T)
        else:
            current += 1
            places.append([T])

    return places, history


def discover_hysteresis(S, min_place_size=3, h=3):
    """Refinement 4: require h consecutive frames below threshold."""
    N = S.shape[0]
    split = bootstrap(S, min_place_size)
    places = [list(range(0, split))]
    place1_end = min(split + min_place_size, N)
    places.append(list(range(split, place1_end)))
    history = []
    current = 1
    below_count = 0  # consecutive frames below threshold
    pending = []     # frames tentatively below threshold

    for T in range(place1_end, N):
        theta = compute_negative_threshold(S, places, current)
        score = score_mean(S, T, np.array(places[current]))
        below = score <= theta

        if below:
            below_count += 1
            pending.append(T)
            decision = "pending"

            if below_count >= h:
                # Confirmed boundary: the first pending frame starts the new place
                current += 1
                places.append(pending.copy())
                pending = []
                below_count = 0
                decision = "new_place"
        else:
            # Reset: add any pending frames back to current place
            if pending:
                places[current].extend(pending)
                pending = []
            below_count = 0
            places[current].append(T)
            decision = "stay"

        history.append({"frame": T, "score": score, "theta": theta,
                        "decision": decision})

    # Handle any remaining pending frames
    if pending:
        places[current].extend(pending)

    return places, history


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_single_result(S, places, history, title, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: similarity matrix with boxes
    ax = axes[0]
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

    place_sizes = [len(p) for p in places]
    ax.set_title(f"{title}\n{len(places)} places, sizes: {place_sizes}",
                 fontsize=10)
    ax.set_xlabel("Frame j")
    ax.set_ylabel("Frame i")

    # Right: score vs threshold
    ax2 = axes[1]
    if history:
        frames = [h["frame"] for h in history]
        scores = [h["score"] for h in history]
        thetas = [h["theta"] for h in history]

        ax2.plot(frames, scores, color="#2980b9", linewidth=1.0,
                 label="score", alpha=0.8)
        ax2.plot(frames, thetas, color="#e74c3c", linewidth=1.5,
                 label="θ (threshold)", linestyle="--")

        new_frames = [h["frame"] for h in history if h["decision"] == "new_place"]
        new_scores = [h["score"] for h in history if h["decision"] == "new_place"]
        ax2.scatter(new_frames, new_scores, color="#e74c3c", zorder=5, s=50,
                    label="New place")

        boundaries = [places[0][0]] + [p[0] for p in places[1:]] + [S.shape[0]]
        for i in range(len(boundaries) - 1):
            ax2.axvspan(boundaries[i], boundaries[i + 1],
                        alpha=0.08, color=colors[i % len(colors)])

    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("Similarity / Threshold")
    ax2.set_title("Score vs threshold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_summary_comparison(all_results, S, output_path):
    """All variants side by side — similarity matrices only."""
    n = len(all_results)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows))
    axes = np.array(axes).flatten()

    for idx, (name, (places, history)) in enumerate(all_results.items()):
        ax = axes[idx]
        ax.imshow(S, cmap="hot", vmin=0, vmax=1, aspect="auto")
        colors = plt.cm.Set2(np.linspace(0, 1, min(len(places), 8)))
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
        ax.set_title(f"{name}\n{len(places)} places, sizes={sizes}", fontsize=9)
        ax.set_xlabel("Frame j", fontsize=8)
        ax.set_ylabel("Frame i", fontsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Refinement comparison — each tested independently against baseline",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    output_dir = "results/visualizations/refinement_tests"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    img_dir = "images/GardensPoint/day_left"
    img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    N = len(img_paths)
    print(f"Found {N} images")

    cache_dir = "cache/GardensPoint/day_left/eigenplaces"
    features = load_features(cache_dir, N)
    F = normalize_l2(features)
    S = F @ F.T
    print(f"Similarity matrix: {S.shape}")

    # Run each variant
    all_results = {}

    # Baseline
    print("\n--- Baseline (mean, no refinements) ---")
    places, hist = discover_baseline(S)
    all_results["Baseline"] = (places, hist)
    print(f"  {len(places)} places: {[len(p) for p in places]}")
    plot_single_result(S, places, hist, "Baseline",
                       os.path.join(output_dir, "baseline.png"))

    # R1: Sliding window
    for W in [5, 10, 15]:
        print(f"\n--- R1: Sliding window (W={W}) ---")
        places, hist = discover_sliding_window(S, window=W)
        label = f"R1: Window W={W}"
        all_results[label] = (places, hist)
        print(f"  {len(places)} places: {[len(p) for p in places]}")
        plot_single_result(S, places, hist, label,
                           os.path.join(output_dir, f"r1_window_{W}.png"))

    # R2: Min place size merge
    for k in [3, 5]:
        print(f"\n--- R2: Min place merge (k={k}) ---")
        places, hist = discover_min_place_merge(S, min_merge=k)
        label = f"R2: Min merge k={k}"
        all_results[label] = (places, hist)
        print(f"  {len(places)} places: {[len(p) for p in places]}")
        plot_single_result(S, places, hist, label,
                           os.path.join(output_dir, f"r2_merge_{k}.png"))

    # R3: Top-k scoring
    for k in [3, 5]:
        print(f"\n--- R3: Top-k scoring (k={k}) ---")
        places, hist = discover_topk(S, k=k)
        label = f"R3: Top-k k={k}"
        all_results[label] = (places, hist)
        print(f"  {len(places)} places: {[len(p) for p in places]}")
        plot_single_result(S, places, hist, label,
                           os.path.join(output_dir, f"r3_topk_{k}.png"))

    # R4: Hysteresis
    for h in [2, 3]:
        print(f"\n--- R4: Hysteresis (h={h}) ---")
        places, hist = discover_hysteresis(S, h=h)
        label = f"R4: Hysteresis h={h}"
        all_results[label] = (places, hist)
        print(f"  {len(places)} places: {[len(p) for p in places]}")
        plot_single_result(S, places, hist, label,
                           os.path.join(output_dir, f"r4_hysteresis_{h}.png"))

    # Summary comparison
    plot_summary_comparison(all_results, S,
                           os.path.join(output_dir, "all_refinements.png"))

    # Print summary table
    print(f"\n{'='*65}")
    print(f"{'Variant':<30} {'Places':>6} {'Sizes'}")
    print(f"{'-'*65}")
    for name, (places, _) in all_results.items():
        sizes = [len(p) for p in places]
        print(f"{name:<30} {len(places):>6}   {sizes}")
    print(f"{'='*65}")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
