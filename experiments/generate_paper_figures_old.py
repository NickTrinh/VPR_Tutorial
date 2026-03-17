"""
Generate figures to improve the original RCC 2025 paper.

Figures:
  A. Dataset sample images (conditions side-by-side)
  B. Gaussian mixture visualization for example places
  C. Recall@1 improvement bar chart (baseline vs simple avg)
  D. Filter-then-rank illustration (similarity matrix before/after)

Usage:
    python experiments/generate_paper_figures_old.py
"""

import os
import sys
import pickle
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

OUTPUT_DIR = "results/paper_figures_rcc"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_cached_descriptors(cache_dir, n):
    descs = []
    for i in range(n):
        path = os.path.join(cache_dir, f"img_{i}_descriptor.pkl")
        with open(path, "rb") as f:
            d = pickle.load(f)
            if isinstance(d, dict):
                d = d["descriptor"]
            descs.append(d)
    return np.array(descs)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE A: Dataset sample images
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating Figure A: Dataset Sample Images...")

fig, axes = plt.subplots(3, 6, figsize=(15, 7.5))

# GardensPoint: day_left, day_right, night_right
gp_conditions = [
    ("images/GardensPoint/day_left", "Day Left"),
    ("images/GardensPoint/day_right", "Day Right"),
]
# Pick 2 sample frames to show condition variation
sample_frames = [10, 80]

for row, (cond_path, cond_name) in enumerate(gp_conditions):
    imgs = sorted(glob(os.path.join(cond_path, "*.jpg")))
    for col, frame_idx in enumerate(sample_frames):
        if frame_idx < len(imgs):
            try:
                img = Image.open(imgs[frame_idx])
                axes[row, col].imshow(np.array(img))
                axes[row, col].set_title(f"GP {cond_name}\nFrame {frame_idx}", fontsize=8)
            except Exception:
                pass
        axes[row, col].axis("off")

# Nordland: summer, winter
nord_conditions = [
    ("images/Nordland_filtered/summer", "Summer"),
    ("images/Nordland_filtered/winter", "Winter"),
]
for row, (cond_path, cond_name) in enumerate(nord_conditions):
    imgs = sorted(glob(os.path.join(cond_path, "*.png")))
    for col, frame_idx in enumerate(sample_frames):
        if frame_idx < len(imgs):
            try:
                img = Image.open(imgs[frame_idx])
                axes[row, col + 2].imshow(np.array(img))
                axes[row, col + 2].set_title(f"Nordland {cond_name}\nFrame {frame_idx}", fontsize=8)
            except Exception:
                pass
        axes[row, col + 2].axis("off")

# SFU: dry
sfu_conditions = [
    ("images/SFU/dry", "Dry"),
    ("images/SFU/dusk", "Dusk"),
]
for row, (cond_path, cond_name) in enumerate(sfu_conditions):
    imgs = sorted(glob(os.path.join(cond_path, "*.jpg")))
    if not imgs:
        imgs = sorted(glob(os.path.join(cond_path, "*.png")))
    for col, frame_idx in enumerate(sample_frames):
        if frame_idx < len(imgs):
            try:
                img = Image.open(imgs[frame_idx])
                axes[row, col + 4].imshow(np.array(img))
                axes[row, col + 4].set_title(f"SFU {cond_name}\nFrame {frame_idx}", fontsize=8)
            except Exception:
                pass
        axes[row, col + 4].axis("off")

# Third row: hide or use for more samples
for col in range(6):
    axes[2, col].axis("off")

# Row 3: Show same place under different conditions
# GardensPoint frame 50: day_left vs day_right
gp_dl = sorted(glob("images/GardensPoint/day_left/*.jpg"))
gp_dr = sorted(glob("images/GardensPoint/day_right/*.jpg"))
nord_sum = sorted(glob("images/Nordland_filtered/summer/*.png"))
nord_win = sorted(glob("images/Nordland_filtered/winter/*.png"))

pairs = [
    (gp_dl, 50, "GP Day Left\nFrame 50"),
    (gp_dr, 50, "GP Day Right\nFrame 50"),
    (nord_sum, 50, "Nordland Summer\nFrame 50"),
    (nord_win, 50, "Nordland Winter\nFrame 50"),
]

for col, (img_list, idx, title) in enumerate(pairs):
    if idx < len(img_list):
        try:
            img = Image.open(img_list[idx])
            axes[2, col].imshow(np.array(img))
            axes[2, col].set_title(title, fontsize=8)
        except Exception:
            pass
    axes[2, col].axis("off")

# Label rows
fig.text(0.01, 0.78, "Condition 1", rotation=90, va="center", fontsize=10, fontweight="bold")
fig.text(0.01, 0.50, "Condition 2", rotation=90, va="center", fontsize=10, fontweight="bold")
fig.text(0.01, 0.22, "Same Place,\nDiff. Condition", rotation=90, va="center", fontsize=10, fontweight="bold")

plt.suptitle("Dataset Samples: Appearance Variation Across Conditions",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0.03, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "figA_dataset_samples.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "figA_dataset_samples.pdf"), bbox_inches="tight")
plt.close()
print("  Saved figA_dataset_samples")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE B: Gaussian Mixture Visualization
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating Figure B: Gaussian Mixture Visualization...")

# Load GardensPoint Mini data (the dataset used in the RCC paper)
# Use the landmark_grouped format
gp_mini_path = "images/GardensPoint_Mini"
gp_mini_conditions = ["day_left", "day_right", "night_right"]

# Load reference descriptors from day_left
ref_cache = "cache/GardensPoint/day_left/eigenplaces"
ref_descs = load_cached_descriptors(ref_cache, 200)

# Use the group-and-step approach from the paper: group=3, step=10
# This creates places from consecutive frames
group = 3
step = 10
places = []
idx = 0
while idx + group <= 200:
    places.append(list(range(idx, idx + group)))
    idx += group + step

print(f"  Created {len(places)} places with group={group}, step={step}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Pick 3 places at different positions
example_places = [0, len(places) // 2, len(places) - 1]

for ax_idx, p_idx in enumerate(example_places):
    ax = axes[ax_idx]
    target = places[p_idx]
    other = [f for i, p in enumerate(places) if i != p_idx for f in p]

    if not other or len(target) < 2:
        continue

    # Negative similarities: each image in this place vs all other-place images
    neg_sims = ref_descs[target] @ ref_descs[other].T

    # Per-image mean bad scores (these are the Gaussian components)
    per_img_mean_bad = neg_sims.mean(axis=1)
    per_img_std_bad = neg_sims.std(axis=1)

    # Plot individual per-image distributions as thin Gaussians
    x_range = np.linspace(-0.1, 0.5, 300)
    colors_comp = plt.cm.Set2(np.linspace(0, 1, len(target)))

    # All negative scores histogram
    all_neg = neg_sims.flatten()
    ax.hist(all_neg, bins=60, alpha=0.3, color="#bdc3c7", density=True,
            edgecolor="none", label="All negative scores")

    # Individual Gaussian components
    mixture_pdf = np.zeros_like(x_range)
    n_images = len(target)
    for img_idx in range(n_images):
        mu = per_img_mean_bad[img_idx]
        sigma = per_img_std_bad[img_idx]
        if sigma < 1e-8:
            sigma = 0.01
        weight = 1.0 / n_images
        component = weight * norm.pdf(x_range, mu, sigma)
        mixture_pdf += component
        ax.plot(x_range, component, color=colors_comp[img_idx], linewidth=1.5, alpha=0.7,
                label=f"Image {target[img_idx]}: $\\mu$={mu:.3f}" if img_idx < 4 else None)

    # Total mixture
    ax.plot(x_range, mixture_pdf, color="#2c3e50", linewidth=2.5, linestyle="-",
            label=f"Mixture M(p)")

    # Threshold: simple average = mean of per-image means
    theta_simple = float(per_img_mean_bad.mean())
    ax.axvline(theta_simple, color="#e74c3c", linewidth=2, linestyle="--",
               label=f"$\\theta_p$ = {theta_simple:.3f}")

    # Weighted average (precision-weighted)
    precisions = 1.0 / (per_img_std_bad ** 2 + 1e-8)
    theta_weighted = float(np.sum(precisions * per_img_mean_bad) / np.sum(precisions))
    ax.axvline(theta_weighted, color="#3498db", linewidth=2, linestyle=":",
               label=f"$\\theta_p^w$ = {theta_weighted:.3f}")

    ax.set_xlabel("Cosine similarity", fontsize=10)
    if ax_idx == 0:
        ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"Place {p_idx} (frames {target[0]}-{target[-1]})\n"
                 f"{n_images} components", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)

plt.suptitle("Negative Gaussian Mixture M(p): Per-Image Components and Threshold",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figB_gaussian_mixture.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "figB_gaussian_mixture.pdf"), bbox_inches="tight")
plt.close()
print("  Saved figB_gaussian_mixture")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE C: Recall@1 Improvement Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating Figure C: Recall@1 Improvement Chart...")

# Data from Table I in the paper
data = {
    "GP Mini\n(20 places)": {
        "EigenPlaces": {"Baseline": 61.67, "Simple Avg": 93.33},
        "CosPlace": {"Baseline": 53.33, "Simple Avg": 93.33},
        "AlexNet": {"Baseline": 43.33, "Simple Avg": 85.00},
    },
    "SFU Mini\n(192 places)": {
        "EigenPlaces": {"Baseline": 76.04, "Simple Avg": 81.25},
        "CosPlace": {"Baseline": 62.76, "Simple Avg": 69.27},
        "AlexNet": {"Baseline": 57.55, "Simple Avg": 62.50},
    },
    "Nordland g3s3\n(9,197 places)": {
        "EigenPlaces": {"Baseline": 69.04, "Simple Avg": 76.72},
        "CosPlace": {"Baseline": 68.15, "Simple Avg": 78.21},
        "AlexNet": {"Baseline": 39.61, "Simple Avg": 42.85},
    },
    "Nordland g2s2\n(13,796 places)": {
        "EigenPlaces": {"Baseline": 69.04, "Simple Avg": 74.42},
        "CosPlace": {"Baseline": 68.15, "Simple Avg": 78.21},
        "AlexNet": {"Baseline": 39.61, "Simple Avg": 42.85},
    },
}

fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
descriptors = ["EigenPlaces", "CosPlace", "AlexNet"]
desc_colors_base = ["#bdc3c7", "#bdc3c7", "#bdc3c7"]
desc_colors_ours = ["#3498db", "#e74c3c", "#f39c12"]

for ax_idx, (dataset, descs) in enumerate(data.items()):
    ax = axes[ax_idx]
    x = np.arange(len(descriptors))
    width = 0.35

    baselines = [descs[d]["Baseline"] for d in descriptors]
    ours = [descs[d]["Simple Avg"] for d in descriptors]

    bars1 = ax.bar(x - width/2, baselines, width, label="Baseline",
                    color="#bdc3c7", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, ours, width, label="Ours (Simple Avg)",
                    color=desc_colors_ours, edgecolor="white", linewidth=0.5, alpha=0.85)

    # Value labels
    for bar, val in zip(bars1, baselines):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}",
                ha="center", fontsize=7, color="#7f8c8d")
    for bar, val in zip(bars2, ours):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}",
                ha="center", fontsize=7, fontweight="bold")

    # Improvement arrows
    for i in range(len(descriptors)):
        diff = ours[i] - baselines[i]
        ax.annotate(f"+{diff:.1f}", xy=(x[i], max(baselines[i], ours[i]) + 5),
                    fontsize=7, ha="center", color="#27ae60", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(descriptors, fontsize=8)
    ax.set_title(dataset, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.2, axis="y")
    if ax_idx == 0:
        ax.set_ylabel("Recall@1 (%)", fontsize=11)

axes[0].legend(fontsize=9, loc="upper left")

plt.suptitle("Recall@1 Improvement: Baseline vs Our Adaptive Threshold",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figC_recall_improvement.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "figC_recall_improvement.pdf"), bbox_inches="tight")
plt.close()
print("  Saved figC_recall_improvement")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE D: Filter-then-Rank Illustration
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating Figure D: Filter-then-Rank Illustration...")

# Use GardensPoint with group-and-step places
query_descs = load_cached_descriptors("cache/GardensPoint/day_right/eigenplaces", 200)

# Compute similarity matrix (query vs ref, averaged by place)
S_full = query_descs @ ref_descs.T

n_query = len(query_descs)
n_places = len(places)
place_scores = np.zeros((n_query, n_places))
for p_idx, frames in enumerate(places):
    place_scores[:, p_idx] = S_full[:, frames].mean(axis=1)

# Compute thresholds (simple average)
thresholds = {}
for p_idx in range(n_places):
    target = places[p_idx]
    other = [f for i, p in enumerate(places) if i != p_idx for f in p]
    if not other or len(target) < 2:
        thresholds[p_idx] = 0.0
        continue
    neg_sims = ref_descs[target] @ ref_descs[other].T
    per_img_mean_bad = neg_sims.mean(axis=1)
    thresholds[p_idx] = float(per_img_mean_bad.mean())

# Create filtered version
place_scores_filtered = place_scores.copy()
for q_idx in range(n_query):
    for p_idx in range(n_places):
        if place_scores_filtered[q_idx, p_idx] < thresholds[p_idx]:
            place_scores_filtered[q_idx, p_idx] = 0

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Raw place-level similarity
im1 = ax1.imshow(place_scores.T, aspect="auto", cmap="hot", vmin=0, vmax=1, origin="upper")
ax1.set_xlabel("Query frame", fontsize=10)
ax1.set_ylabel("Place index", fontsize=10)
ax1.set_title("(a) Raw place scores\n(before thresholding)", fontsize=11, fontweight="bold")
plt.colorbar(im1, ax=ax1, shrink=0.8)

# Panel 2: Threshold overlay
im2 = ax2.imshow(place_scores.T, aspect="auto", cmap="hot", vmin=0, vmax=1, origin="upper")
# Draw threshold lines per place
for p_idx in range(n_places):
    theta = thresholds[p_idx]
    ax2.text(n_query + 2, p_idx, f"{theta:.2f}", fontsize=6, va="center", color="#e74c3c")
ax2.set_xlabel("Query frame", fontsize=10)
ax2.set_ylabel("Place index", fontsize=10)
ax2.set_title("(b) With per-place thresholds $\\theta_p$\n(shown on right)", fontsize=11, fontweight="bold")
plt.colorbar(im2, ax=ax2, shrink=0.8)

# Panel 3: After filtering
im3 = ax3.imshow(place_scores_filtered.T, aspect="auto", cmap="hot", vmin=0, vmax=1, origin="upper")
ax3.set_xlabel("Query frame", fontsize=10)
ax3.set_ylabel("Place index", fontsize=10)
ax3.set_title("(c) After filter-then-rank\n(sub-threshold entries zeroed)", fontsize=11, fontweight="bold")
plt.colorbar(im3, ax=ax3, shrink=0.8)

plt.suptitle("Filter-then-Rank: Eliminating Low-Confidence Matches Before Ranking",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figD_filter_then_rank.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "figD_filter_then_rank.pdf"), bbox_inches="tight")
plt.close()
print("  Saved figD_filter_then_rank")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nAll RCC paper figures saved to: {OUTPUT_DIR}/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")
