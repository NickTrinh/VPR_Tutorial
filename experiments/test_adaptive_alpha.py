"""
Test data-driven α (filter_n) for sequential place discovery.

Instead of manually setting α, compute it from the data:
  α = (mean_positive - mean_negative) / std_negative

where mean_positive = mean within-place similarity (how similar
frames in the current place are to each other).

This mirrors the RCC paper's filter_n computation, applied online.

Usage:
    python test_adaptive_alpha.py
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

def bootstrap(S, min_place_size=3):
    N = S.shape[0]
    csim = np.array([S[t, t + 1] for t in range(N - 1)])
    mins = argrelmin(csim, order=3)[0]
    if len(mins) == 0:
        return N // 2
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
    return mins[0] + 1


# ──────────────────────────────────────────────────────────────────────────────
# Statistics computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_place_stats(S, places, place_idx):
    """
    Compute positive and negative statistics for a place.

    Returns:
        mean_pos: mean within-place similarity (excluding self-matches)
        mean_neg: mean similarity to frames in other places
        std_neg:  std of per-image mean-negative-scores
    """
    target_frames = np.array(places[place_idx])
    other_frames = []
    for i, p in enumerate(places):
        if i != place_idx:
            other_frames.extend(p)

    if not other_frames or len(target_frames) < 2:
        return 0.5, 0.0, 0.1  # fallback

    other_frames = np.array(other_frames)

    # Positive: within-place similarity (exclude diagonal / self-matches)
    pos_sims = []
    for i, t in enumerate(target_frames):
        others_in_place = np.concatenate([target_frames[:i], target_frames[i+1:]])
        if len(others_in_place) > 0:
            pos_sims.append(S[t, others_in_place].mean())
    mean_pos = np.mean(pos_sims) if pos_sims else 0.5

    # Negative: cross-place similarity
    mean_neg_per_image = []
    for t in target_frames:
        neg_sims = S[t, other_frames]
        mean_neg_per_image.append(neg_sims.mean())
    mean_neg = np.mean(mean_neg_per_image)
    std_neg = np.std(mean_neg_per_image) if len(mean_neg_per_image) > 1 else 0.1

    return mean_pos, mean_neg, std_neg


def compute_filter_n(mean_pos, mean_neg, std_neg):
    """
    Compute filter_n exactly as in the RCC paper:
    filter_n = floor((good_score - mean_bad) / std_bad)
    """
    if std_neg < 1e-8:
        return 1.0
    return np.floor((mean_pos - mean_neg) / std_neg)


# ──────────────────────────────────────────────────────────────────────────────
# Discovery variants
# ──────────────────────────────────────────────────────────────────────────────

def discover_adaptive_alpha(S, min_place_size=3, hysteresis=2,
                            alpha_scale=1.0, verbose=True):
    """
    Sequential place discovery with data-driven α.

    θ = mean_neg + α * std_neg
    where α = alpha_scale * filter_n
          filter_n = floor((mean_pos - mean_neg) / std_neg)

    alpha_scale lets us test fractions of the full filter_n:
      1.0 = full filter_n (as in the paper)
      0.5 = half of filter_n (more conservative)
    """
    N = S.shape[0]
    split = bootstrap(S, min_place_size)
    places = [list(range(0, split))]
    place1_end = min(split + min_place_size, N)
    places.append(list(range(split, place1_end)))
    history = []
    current = 1
    below_count = 0
    pending = []

    for T in range(place1_end, N):
        mean_pos, mean_neg, std_neg = compute_place_stats(S, places, current)
        fn = compute_filter_n(mean_pos, mean_neg, std_neg)
        alpha = alpha_scale * fn
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
                if verbose:
                    print(f"  Frame {T}: NEW place {current} "
                          f"(score={score:.3f} ≤ θ={theta:.3f}, "
                          f"α={alpha:.2f}, filter_n={fn:.0f})")
        else:
            if pending:
                places[current].extend(pending)
                pending = []
            below_count = 0
            places[current].append(T)

        history.append({
            "frame": T, "score": score, "theta": theta,
            "mean_pos": mean_pos, "mean_neg": mean_neg,
            "std_neg": std_neg, "filter_n": fn, "alpha": alpha,
        })

    if pending:
        places[current].extend(pending)

    if verbose:
        print(f"\n  {len(places)} places: {[len(p) for p in places]}")

    return places, history


def discover_fixed_alpha(S, alpha, min_place_size=3, hysteresis=2):
    """Fixed α for comparison."""
    N = S.shape[0]
    split = bootstrap(S, min_place_size)
    places = [list(range(0, split))]
    place1_end = min(split + min_place_size, N)
    places.append(list(range(split, place1_end)))
    history = []
    current = 1
    below_count = 0
    pending = []

    for T in range(place1_end, N):
        other = []
        for i, p in enumerate(places):
            if i != current:
                other.extend(p)
        other_arr = np.array(other)
        target = np.array(places[current])

        mean_negs = [S[f, other_arr].mean() for f in target]
        mean_neg = np.mean(mean_negs)
        std_neg = np.std(mean_negs) if len(mean_negs) > 1 else 0.1

        theta = mean_neg + alpha * std_neg
        score = S[T, target].mean()
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

        history.append({"frame": T, "score": score, "theta": theta,
                        "alpha": alpha})

    if pending:
        places[current].extend(pending)

    return places, history


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_result_detailed(S, places, history, title, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Left: similarity matrix
    ax = axes[0]
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

    # Right: score, threshold, and alpha over time
    ax2 = axes[1]
    if history:
        frames = [h["frame"] for h in history]
        scores = [h["score"] for h in history]
        thetas = [h["theta"] for h in history]

        ax2.plot(frames, scores, color="#2980b9", linewidth=1.0,
                 label="score (mean)", alpha=0.8)
        ax2.plot(frames, thetas, color="#e74c3c", linewidth=1.5,
                 label="θ (threshold)", linestyle="--")

        # Plot alpha on secondary axis if available
        if "alpha" in history[0] and "filter_n" in history[0]:
            ax3 = ax2.twinx()
            alphas = [h["alpha"] for h in history]
            ax3.plot(frames, alphas, color="#27ae60", linewidth=1.0,
                     alpha=0.6, label="α (data-driven)")
            ax3.set_ylabel("α value", color="#27ae60", fontsize=9)
            ax3.tick_params(axis="y", labelcolor="#27ae60")
            ax3.legend(loc="upper right", fontsize=8)

        # Shade places
        boundaries = [places[0][0]] + [p[0] for p in places[1:]] + [S.shape[0]]
        for i in range(len(boundaries) - 1):
            ax2.axvspan(boundaries[i], boundaries[i + 1],
                        alpha=0.08, color=colors[i % len(colors)])

    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("Similarity / Threshold")
    ax2.set_title("Score vs threshold (+ adaptive α in green)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_comparison(all_results, S, output_path):
    n = len(all_results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    for idx, (name, (places, _)) in enumerate(all_results.items()):
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
        ax.set_title(f"{name}\n{len(places)} places, sizes={sizes}", fontsize=8)
        ax.set_xlabel("Frame j", fontsize=8)
        ax.set_ylabel("Frame i", fontsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Adaptive α (data-driven) vs fixed α — comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    output_dir = "results/visualizations/adaptive_alpha"
    os.makedirs(output_dir, exist_ok=True)

    img_paths = sorted(glob("images/GardensPoint/day_left/*.jpg"))
    N = len(img_paths)
    print(f"Found {N} images")

    features = load_features("cache/GardensPoint/day_left/eigenplaces", N)
    F = normalize_l2(features)
    S = F @ F.T
    print(f"Similarity matrix: {S.shape}")

    all_results = {}

    # Baseline (α=0, just mean_neg)
    print("\n--- Baseline (α=0) ---")
    places, hist = discover_fixed_alpha(S, alpha=0.0)
    all_results["Baseline\n(θ=mean_neg)"] = (places, hist)
    print(f"  {len(places)} places: {[len(p) for p in places]}")

    # Best fixed α from previous test
    print("\n--- Fixed α=1.0 ---")
    places, hist = discover_fixed_alpha(S, alpha=1.0)
    all_results["Fixed α=1.0"] = (places, hist)
    print(f"  {len(places)} places: {[len(p) for p in places]}")

    # Adaptive α with different scales of filter_n
    for scale in [0.25, 0.5, 0.75, 1.0]:
        print(f"\n--- Adaptive α (scale={scale} × filter_n) ---")
        places, hist = discover_adaptive_alpha(S, alpha_scale=scale)
        label = f"Adaptive\n(α={scale}×filter_n)"
        all_results[label] = (places, hist)
        plot_result_detailed(S, places, hist,
                             f"Adaptive α (scale={scale} × filter_n)",
                             os.path.join(output_dir,
                                          f"adaptive_scale_{scale}.png"))

    # Comparison
    plot_comparison(all_results, S,
                    os.path.join(output_dir, "comparison.png"))

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Variant':<35} {'Places':>6}  Sizes")
    print(f"{'-'*70}")
    for name, (places, hist) in all_results.items():
        clean = name.replace('\n', ' ')
        sizes = [len(p) for p in places]

        # Show alpha stats if available
        alpha_info = ""
        if hist and "alpha" in hist[0] and "filter_n" in hist[0]:
            alphas = [h["alpha"] for h in hist]
            fns = [h["filter_n"] for h in hist]
            alpha_info = f"  α={np.mean(alphas):.1f}±{np.std(alphas):.1f}"

        print(f"{clean:<35} {len(places):>6}   {sizes}{alpha_info}")
    print(f"{'='*70}")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
