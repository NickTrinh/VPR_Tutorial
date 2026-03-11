"""
Test approaches to discover more (finer-grained) places.

Problem: baseline discovers only 5-6 places with very large segments.
The threshold is too permissive because negative stats from few places
give a low mean.

Approaches tested:
  A. Raised threshold: θ = mean_neg + α * std_neg  (α = 0.5, 1.0, 1.5, 2.0)
  B. Recursive splitting: run baseline, then re-apply within each large place
  C. Percentile threshold: θ = percentile(neg_scores, q)  (q = 50, 75, 90)

Usage:
    python test_more_places.py
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
# Bootstrap
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap(S, frame_indices, min_place_size=3):
    """Bootstrap on a subsequence defined by frame_indices."""
    n = len(frame_indices)
    if n < 2 * min_place_size:
        return None  # too small to split

    csim = np.array([S[frame_indices[t], frame_indices[t + 1]]
                     for t in range(n - 1)])

    mins = argrelmin(csim, order=3)[0]
    if len(mins) == 0:
        return None

    for m in mins:
        if m < min_place_size:
            continue
        baseline_mean = csim[:m].mean()
        baseline_std = csim[:m].std() if m > 2 else 0.1
        if csim[m] < baseline_mean - 1.5 * baseline_std:
            return m + 1

    valid = mins[mins >= min_place_size]
    if len(valid) > 0:
        return valid[np.argmin(csim[valid])] + 1
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Threshold variants
# ──────────────────────────────────────────────────────────────────────────────

def compute_neg_stats(S, places, place_idx):
    """Return (mean, std) of negative scores for a place."""
    target_frames = places[place_idx]
    other_frames = []
    for i, p in enumerate(places):
        if i != place_idx:
            other_frames.extend(p)
    if not other_frames or not target_frames:
        return 0.0, 0.0

    other_frames = np.array(other_frames)
    mean_neg_scores = []
    for t in target_frames:
        neg_sims = S[t, other_frames]
        mean_neg_scores.append(neg_sims.mean())
    return np.mean(mean_neg_scores), np.std(mean_neg_scores)


def compute_neg_percentile(S, places, place_idx, percentile=75):
    """Return a percentile of the negative scores for a place."""
    target_frames = places[place_idx]
    other_frames = []
    for i, p in enumerate(places):
        if i != place_idx:
            other_frames.extend(p)
    if not other_frames or not target_frames:
        return 0.0

    other_frames = np.array(other_frames)
    all_neg = []
    for t in target_frames:
        all_neg.extend(S[t, other_frames].tolist())
    return np.percentile(all_neg, percentile)


# ──────────────────────────────────────────────────────────────────────────────
# Approach A: Raised threshold (mean + α * std)
# ──────────────────────────────────────────────────────────────────────────────

def discover_raised_threshold(S, alpha=1.0, min_place_size=3, hysteresis=2):
    N = S.shape[0]
    split = bootstrap(S, list(range(N)), min_place_size)
    if split is None:
        return [list(range(N))], []
    places = [list(range(0, split))]
    place1_end = min(split + min_place_size, N)
    places.append(list(range(split, place1_end)))
    history = []
    current = 1
    below_count = 0
    pending = []

    for T in range(place1_end, N):
        mean_neg, std_neg = compute_neg_stats(S, places, current)
        theta = mean_neg + alpha * std_neg
        score = S[T, np.array(places[current])].mean()
        below = score <= theta

        if below:
            below_count += 1
            pending.append(T)
            if below_count >= hysteresis:
                current += 1
                places.append(pending.copy())
                pending = []
                below_count = 0
        else:
            if pending:
                places[current].extend(pending)
                pending = []
            below_count = 0
            places[current].append(T)

        history.append({"frame": T, "score": score, "theta": theta})

    if pending:
        places[current].extend(pending)

    return places, history


# ──────────────────────────────────────────────────────────────────────────────
# Approach B: Recursive splitting
# ──────────────────────────────────────────────────────────────────────────────

def discover_baseline_on_indices(S, frame_indices, min_place_size=3, hysteresis=2):
    """Run baseline discovery on a subsequence defined by frame_indices."""
    n = len(frame_indices)
    if n < 2 * min_place_size:
        return [list(frame_indices)]

    split = bootstrap(S, frame_indices, min_place_size)
    if split is None:
        return [list(frame_indices)]

    places = [list(frame_indices[:split])]
    place1_end = min(split + min_place_size, n)
    places.append(list(frame_indices[split:place1_end]))
    current = 1
    below_count = 0
    pending = []

    for t_local in range(place1_end, n):
        T = frame_indices[t_local]
        # Compute neg stats within this subsequence's places
        target = places[current]
        other = []
        for i, p in enumerate(places):
            if i != current:
                other.extend(p)
        if not other:
            places[current].append(T)
            continue

        other_arr = np.array(other)
        mean_negs = [S[f, other_arr].mean() for f in target]
        theta = np.mean(mean_negs)

        score = S[T, np.array(target)].mean()
        below = score <= theta

        if below:
            below_count += 1
            pending.append(T)
            if below_count >= hysteresis:
                current += 1
                places.append(pending.copy())
                pending = []
                below_count = 0
        else:
            if pending:
                places[current].extend(pending)
                pending = []
            below_count = 0
            places[current].append(T)

    if pending:
        places[current].extend(pending)

    return places


def discover_recursive(S, min_place_size=3, max_depth=3, min_split_size=8):
    """Recursively split places."""
    # First pass: baseline on full sequence
    all_indices = list(range(S.shape[0]))
    top_places = discover_baseline_on_indices(S, all_indices, min_place_size)

    for depth in range(max_depth):
        new_places = []
        did_split = False
        for place in top_places:
            if len(place) >= min_split_size:
                sub = discover_baseline_on_indices(
                    S, place, min_place_size, hysteresis=2
                )
                if len(sub) > 1:
                    new_places.extend(sub)
                    did_split = True
                else:
                    new_places.append(place)
            else:
                new_places.append(place)
        top_places = new_places
        if not did_split:
            break

    return top_places


# ──────────────────────────────────────────────────────────────────────────────
# Approach C: Percentile threshold
# ──────────────────────────────────────────────────────────────────────────────

def discover_percentile(S, percentile=75, min_place_size=3, hysteresis=2):
    N = S.shape[0]
    split = bootstrap(S, list(range(N)), min_place_size)
    if split is None:
        return [list(range(N))], []
    places = [list(range(0, split))]
    place1_end = min(split + min_place_size, N)
    places.append(list(range(split, place1_end)))
    history = []
    current = 1
    below_count = 0
    pending = []

    for T in range(place1_end, N):
        theta = compute_neg_percentile(S, places, current, percentile)
        score = S[T, np.array(places[current])].mean()
        below = score <= theta

        if below:
            below_count += 1
            pending.append(T)
            if below_count >= hysteresis:
                current += 1
                places.append(pending.copy())
                pending = []
                below_count = 0
        else:
            if pending:
                places[current].extend(pending)
                pending = []
            below_count = 0
            places[current].append(T)

        history.append({"frame": T, "score": score, "theta": theta})

    if pending:
        places[current].extend(pending)

    return places, history


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_result(S, places, title, output_path, img_paths=None):
    if img_paths:
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
        ax = fig.add_subplot(gs[0])
        ax_imgs = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax_imgs = None

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
    ax.set_title(f"{title}\n{len(places)} places, sizes={sizes}", fontsize=10)
    ax.set_xlabel("Frame j")
    ax.set_ylabel("Frame i")

    # Show first frame of each place as thumbnails
    if ax_imgs and img_paths:
        from PIL import Image as PILImage
        n_show = min(len(places), 20)
        for i in range(n_show):
            frame = places[i][0]
            img = PILImage.open(img_paths[frame]).resize((80, 45))
            # Position in the axis
            extent = [places[i][0] - 0.5, places[i][0] + 8.5,
                      0, 1]
            ax_imgs.imshow(np.array(img),
                          extent=extent, aspect="auto")
        ax_imgs.set_xlim(0, S.shape[0])
        ax_imgs.set_ylim(0, 1)
        ax_imgs.set_xlabel("Frame index")
        ax_imgs.set_yticks([])
        ax_imgs.set_title("First frame of each place", fontsize=9)
        # Mark boundaries
        for i, place in enumerate(places):
            if i > 0:
                ax_imgs.axvline(x=place[0] - 0.5, color="red",
                               linewidth=1, alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_all_comparison(all_results, S, output_path):
    n = len(all_results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    for idx, (name, places) in enumerate(all_results.items()):
        ax = axes[idx]
        ax.imshow(S, cmap="hot", vmin=0, vmax=1, aspect="auto")
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(places), 1)))
        for i, place in enumerate(places):
            start, end = place[0], place[-1]
            rect = plt.Rectangle((start - 0.5, start - 0.5),
                                  end - start + 1, end - start + 1,
                                  linewidth=2,
                                  edgecolor=colors[i % len(colors)],
                                  facecolor="none")
            ax.add_patch(rect)
            if i > 0:
                ax.axvline(x=start - 0.5, color="white", linewidth=0.8, alpha=0.6)
                ax.axhline(y=start - 0.5, color="white", linewidth=0.8, alpha=0.6)
        sizes = [len(p) for p in places]
        ax.set_title(f"{name}\n{len(places)} places", fontsize=9)
        ax.set_xlabel("Frame j", fontsize=8)
        ax.set_ylabel("Frame i", fontsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Approaches to discover more places — comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    output_dir = "results/visualizations/more_places"
    os.makedirs(output_dir, exist_ok=True)

    img_dir = "images/GardensPoint/day_left"
    img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    N = len(img_paths)
    print(f"Found {N} images")

    cache_dir = "cache/GardensPoint/day_left/eigenplaces"
    features = load_features(cache_dir, N)
    F = normalize_l2(features)
    S = F @ F.T
    print(f"Similarity matrix: {S.shape}")

    all_results = {}

    # Baseline (with hysteresis)
    print("\n--- Baseline (mean, θ=mean_neg, hysteresis=2) ---")
    places, _ = discover_raised_threshold(S, alpha=0.0)
    all_results["Baseline\n(θ=mean_neg)"] = places
    print(f"  {len(places)} places: {[len(p) for p in places]}")

    # A: Raised threshold
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        print(f"\n--- A: Raised threshold (α={alpha}) ---")
        places, _ = discover_raised_threshold(S, alpha=alpha)
        label = f"A: α={alpha}\n(θ=μ+{alpha}σ)"
        all_results[label] = places
        print(f"  {len(places)} places: {[len(p) for p in places]}")
        plot_result(S, places, f"Raised threshold α={alpha}",
                    os.path.join(output_dir, f"a_alpha_{alpha}.png"), img_paths)

    # B: Recursive
    print(f"\n--- B: Recursive splitting ---")
    places = discover_recursive(S, min_place_size=3, max_depth=3, min_split_size=8)
    all_results["B: Recursive\n(depth≤3)"] = places
    print(f"  {len(places)} places: {[len(p) for p in places]}")
    plot_result(S, places, "Recursive splitting",
                os.path.join(output_dir, "b_recursive.png"), img_paths)

    # C: Percentile
    for q in [50, 75, 90]:
        print(f"\n--- C: Percentile threshold (q={q}) ---")
        places, _ = discover_percentile(S, percentile=q)
        label = f"C: P{q}\n(θ=P{q}(neg))"
        all_results[label] = places
        print(f"  {len(places)} places: {[len(p) for p in places]}")
        plot_result(S, places, f"Percentile threshold q={q}",
                    os.path.join(output_dir, f"c_percentile_{q}.png"), img_paths)

    # Comparison
    plot_all_comparison(all_results, S,
                        os.path.join(output_dir, "comparison.png"))

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Variant':<30} {'Places':>6}  {'Sizes'}")
    print(f"{'-'*70}")
    for name, places in all_results.items():
        clean_name = name.replace('\n', ' ')
        sizes = [len(p) for p in places]
        print(f"{clean_name:<30} {len(places):>6}   {sizes}")
    print(f"{'='*70}")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
