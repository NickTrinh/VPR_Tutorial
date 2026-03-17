"""
Generate paper figures for the extended adaptive thresholding paper.

Figures:
  1. Place discovery: consecutive similarity + boundaries + sample images
  2. Per-place score distributions (good vs bad) with thresholds marked
  3. Closed-set vs open-set comparison (the "flip" figure)
  4. Distractor score distributions showing why filter_n wins
  5. Rejection-recall tradeoff curve

Uses cached descriptors — no feature extraction needed.

Usage:
    python experiments/generate_paper_figures.py
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
from matplotlib.patches import FancyBboxPatch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from experiments.online_place_discovery import OnlineFeatureExtractor, OnlinePlaceDiscovery
from experiments.vysotska_threshold import VysotskaDaptiveThreshold

OUTPUT_DIR = "results/paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_cached_descriptors(cache_dir, n):
    """Load n cached descriptors."""
    descs = []
    for i in range(n):
        path = os.path.join(cache_dir, f"img_{i}_descriptor.pkl")
        with open(path, "rb") as f:
            d = pickle.load(f)
            if isinstance(d, dict):
                d = d["descriptor"]
            descs.append(d)
    return np.array(descs)


def discover_places(ref_descs, filter_n_cap=10):
    """Run place discovery on preloaded descriptors."""
    discoverer = OnlinePlaceDiscovery(
        min_place_size=3, hysteresis=2, filter_n_cap=filter_n_cap
    )
    for i in range(len(ref_descs)):
        discoverer.process_frame(ref_descs[i], i, verbose=False)
    return discoverer.places


def compute_place_stats(ref_descs, places):
    """Compute per-place statistics: mean_bad, std_bad, mean_good, filter_n, thresholds."""
    stats = {}
    for p_idx in range(len(places)):
        target = places[p_idx]
        other = [f for i, p in enumerate(places) if i != p_idx for f in p]

        if not other or len(target) < 2:
            stats[p_idx] = {"mean_bad": 0, "std_bad": 0.1, "mean_good": 0.5,
                            "filter_n": 1, "theta_mean_bad": 0, "theta_filter_n": 0}
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

        stats[p_idx] = {
            "mean_bad": mean_bad,
            "std_bad": std_bad,
            "mean_good": mean_good,
            "filter_n": filter_n,
            "theta_mean_bad": mean_bad,
            "theta_filter_n": mean_bad + filter_n * std_bad,
            "neg_sims_flat": neg_sims.flatten(),
            "pos_sims_flat": pos_sims[np.triu_indices(len(target), k=1)],
        }
    return stats


def evaluate_method(query_descs, ref_descs, places, thresholds):
    """Evaluate with per-place thresholds. Returns TP, FP, FN, predictions."""
    S = query_descs @ ref_descs.T
    n_query = len(query_descs)
    n_places = len(places)

    frame_to_place = {}
    for p_idx, frames in enumerate(places):
        for f in frames:
            frame_to_place[f] = p_idx

    place_scores = np.zeros((n_query, n_places))
    for p_idx, frames in enumerate(places):
        place_scores[:, p_idx] = S[:, frames].mean(axis=1)

    TP, FP, FN, TN = 0, 0, 0, 0
    for q_idx in range(n_query):
        gt_place = frame_to_place.get(q_idx, -1)
        scores = place_scores[q_idx].copy()

        for p_idx in range(n_places):
            if scores[p_idx] < thresholds.get(p_idx, -np.inf):
                scores[p_idx] = -np.inf

        if np.all(scores == -np.inf):
            pred = -1
        else:
            pred = int(np.argmax(scores))

        if pred == -1:
            if gt_place == -1:
                TN += 1
            else:
                FN += 1
        else:
            if pred == gt_place:
                TP += 1
            else:
                FP += 1

    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    dist_rej = TN / max(TN + FP, 1) * 100 if TN + FP > 0 else None

    return {"P": precision, "R": recall, "F1": f1, "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "dist_rej": dist_rej}


def evaluate_vysotska(query_descs, ref_descs, places, patch_size=20):
    """Evaluate Vysotska method."""
    S = query_descs @ ref_descs.T
    n_query = len(query_descs)
    n_places = len(places)

    frame_to_place = {}
    for p_idx, frames in enumerate(places):
        for f in frames:
            frame_to_place[f] = p_idx

    vysotska = VysotskaDaptiveThreshold(patch_size=patch_size)
    per_query_thresholds, _ = vysotska.compute_thresholds(S)

    place_scores = np.zeros((n_query, n_places))
    for p_idx, frames in enumerate(places):
        place_scores[:, p_idx] = S[:, frames].mean(axis=1)

    TP, FP, FN, TN = 0, 0, 0, 0
    for q_idx in range(n_query):
        gt_place = frame_to_place.get(q_idx, -1)
        scores = place_scores[q_idx].copy()
        theta = per_query_thresholds[q_idx]

        for p_idx in range(n_places):
            if scores[p_idx] < theta:
                scores[p_idx] = -np.inf

        if np.all(scores == -np.inf):
            pred = -1
        else:
            pred = int(np.argmax(scores))

        if pred == -1:
            if gt_place == -1:
                TN += 1
            else:
                FN += 1
        else:
            if pred == gt_place:
                TP += 1
            else:
                FP += 1

    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    dist_rej = TN / max(TN + FP, 1) * 100 if TN + FP > 0 else None

    return {"P": precision, "R": recall, "F1": f1, "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "dist_rej": dist_rej}, per_query_thresholds


# ─── Load all data ────────────────────────────────────────────────────────────

print("Loading cached descriptors...")
ref_descs = load_cached_descriptors("cache/GardensPoint/day_left/eigenplaces", 200)
query_descs = load_cached_descriptors("cache/GardensPoint/day_right/eigenplaces", 200)
n_sfu = len(glob("cache/SFU/dry/eigenplaces/img_*_descriptor.pkl"))
sfu_descs = load_cached_descriptors("cache/SFU/dry/eigenplaces", n_sfu)

print(f"  Reference: {len(ref_descs)}, Query: {len(query_descs)}, SFU distractors: {len(sfu_descs)}")

print("Discovering places...")
places = discover_places(ref_descs)
stats = compute_place_stats(ref_descs, places)
print(f"  {len(places)} places discovered")

# Reference image paths for sample images
ref_img_paths = sorted(glob("images/GardensPoint/day_left/*.jpg"))


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Place Discovery — consecutive similarity + boundaries + samples
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGenerating Figure 1: Place Discovery...")

consec_sims = np.array([
    float(ref_descs[i] @ ref_descs[i + 1]) for i in range(len(ref_descs) - 1)
])

fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.05)

# Top: sample images at place midpoints
ax_img = fig.add_subplot(gs[0])
ax_img.set_xlim(-0.5, len(ref_descs) - 0.5)
ax_img.axis("off")

colors = plt.cm.tab20(np.linspace(0, 1, len(places)))
n_samples = min(len(places), 19)  # show up to 19 places
for p_idx in range(n_samples):
    frames = places[p_idx]
    mid = frames[len(frames) // 2]
    if mid < len(ref_img_paths):
        try:
            img = Image.open(ref_img_paths[mid]).resize((60, 45))
            x_pos = mid / len(ref_descs)
            ax_img.imshow(np.array(img),
                         extent=[mid - 3, mid + 3, 0, 1],
                         aspect="auto", zorder=5)
            # Colored border
            rect = plt.Rectangle((mid - 3.2, -0.02), 6.4, 1.04,
                                linewidth=2, edgecolor=colors[p_idx],
                                facecolor="none", zorder=6)
            ax_img.add_patch(rect)
        except Exception:
            pass

ax_img.set_title("Sample images from discovered place midpoints", fontsize=11, pad=10)

# Bottom: consecutive similarity with place boundaries
ax = fig.add_subplot(gs[1])
ax.plot(range(len(consec_sims)), consec_sims, color="#2c3e50", linewidth=0.8, alpha=0.9)

# Shade each place
for p_idx, frames in enumerate(places):
    start, end = frames[0], frames[-1]
    ax.axvspan(start, end, alpha=0.15, color=colors[p_idx])
    # Boundary lines
    if p_idx > 0:
        ax.axvline(start, color="#e74c3c", linewidth=1.0, alpha=0.7, linestyle="--")

# Labels
ax.set_xlabel("Frame index", fontsize=11)
ax.set_ylabel("Cosine similarity to next frame", fontsize=11)
ax.set_xlim(-0.5, len(ref_descs) - 0.5)
ax.grid(True, alpha=0.2)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    plt.Line2D([0], [0], color="#2c3e50", linewidth=1, label="sim(t, t+1)"),
    plt.Line2D([0], [0], color="#e74c3c", linewidth=1, linestyle="--", label="Place boundary"),
    Patch(facecolor="gray", alpha=0.15, label="Discovered place"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=9)

plt.savefig(os.path.join(OUTPUT_DIR, "fig1_place_discovery.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_place_discovery.pdf"), bbox_inches="tight")
plt.close()
print("  Saved fig1_place_discovery")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Per-place score distributions with thresholds
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating Figure 2: Score Distributions...")

# Pick 3 interesting places: well-separated, medium, hard
place_separations = []
for p_idx in range(len(places)):
    s = stats[p_idx]
    if "pos_sims_flat" in s:
        gap = s["mean_good"] - s["mean_bad"]
        place_separations.append((p_idx, gap, len(places[p_idx])))

place_separations.sort(key=lambda x: x[1], reverse=True)
# Pick best, middle, worst (that have enough data)
valid = [p for p in place_separations if p[2] >= 4]
if len(valid) >= 3:
    example_places = [valid[0][0], valid[len(valid)//2][0], valid[-1][0]]
else:
    example_places = [p[0] for p in valid[:3]]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
labels = ["Well-separated place", "Medium-separated place", "Hard place"]

for ax_idx, (p_idx, label) in enumerate(zip(example_places, labels)):
    ax = axes[ax_idx]
    s = stats[p_idx]

    if "neg_sims_flat" not in s:
        continue

    # Histograms
    ax.hist(s["neg_sims_flat"], bins=50, alpha=0.6, color="#e74c3c", density=True,
            label=f"Cross-place (negative)", edgecolor="none")
    ax.hist(s["pos_sims_flat"], bins=30, alpha=0.6, color="#27ae60", density=True,
            label=f"Within-place (positive)", edgecolor="none")

    # Threshold lines
    ax.axvline(s["theta_mean_bad"], color="#3498db", linewidth=2, linestyle="-",
               label=f"$\\theta_{{mean\\_bad}}$ = {s['theta_mean_bad']:.3f}")
    ax.axvline(s["theta_filter_n"], color="#8e44ad", linewidth=2, linestyle="--",
               label=f"$\\theta_{{filter\\_n}}$ = {s['theta_filter_n']:.3f}")

    ax.set_xlabel("Cosine similarity", fontsize=10)
    if ax_idx == 0:
        ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"{label}\n(Place {p_idx}, {len(places[p_idx])} frames, "
                 f"filter_n={s['filter_n']:.0f})", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.2)

plt.suptitle("Per-Place Score Distributions with Adaptive Thresholds",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_score_distributions.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_score_distributions.pdf"), bbox_inches="tight")
plt.close()
print("  Saved fig2_score_distributions")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Closed-set vs Open-set comparison (the "flip" figure)
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating Figure 3: Closed-set vs Open-set...")

# Closed-set evaluation (query = day_right, all genuine)
thresh_none = {p: -np.inf for p in range(len(places))}
thresh_mean_bad = {p: stats[p]["theta_mean_bad"] for p in range(len(places))}
thresh_filter_n = {p: stats[p]["theta_filter_n"] for p in range(len(places))}

closed_baseline = evaluate_method(query_descs, ref_descs, places, thresh_none)
closed_mean_bad = evaluate_method(query_descs, ref_descs, places, thresh_mean_bad)
closed_filter_n = evaluate_method(query_descs, ref_descs, places, thresh_filter_n)
closed_vysotska, _ = evaluate_vysotska(query_descs, ref_descs, places)

# Open-set evaluation (100 genuine + distractors)
genuine_query = query_descs[:100]
n_dist = min(100, len(sfu_descs))
mixed_query = np.vstack([genuine_query, sfu_descs[:n_dist]])

# Build frame_to_place mapping for mixed queries
# First 100 are genuine (frame 0-99), rest are distractors (no place)
frame_to_place_mixed = {}
for p_idx, frames in enumerate(places):
    for f in frames:
        frame_to_place_mixed[f] = p_idx

# For open-set, evaluate manually to handle distractors
def evaluate_open_set(query_descs_mixed, ref_descs, places, thresholds, n_genuine):
    S = query_descs_mixed @ ref_descs.T
    n_query = len(query_descs_mixed)
    n_places = len(places)

    frame_to_place = {}
    for p_idx, frames in enumerate(places):
        for f in frames:
            frame_to_place[f] = p_idx

    place_scores = np.zeros((n_query, n_places))
    for p_idx, frames in enumerate(places):
        place_scores[:, p_idx] = S[:, frames].mean(axis=1)

    TP, FP, FN, TN = 0, 0, 0, 0
    for q_idx in range(n_query):
        is_genuine = q_idx < n_genuine
        gt_place = frame_to_place.get(q_idx, -1) if is_genuine else -1
        scores = place_scores[q_idx].copy()

        for p_idx in range(n_places):
            if scores[p_idx] < thresholds.get(p_idx, -np.inf):
                scores[p_idx] = -np.inf

        if np.all(scores == -np.inf):
            pred = -1
        else:
            pred = int(np.argmax(scores))

        if pred == -1:
            if gt_place == -1:
                TN += 1
            else:
                FN += 1
        else:
            if is_genuine and pred == gt_place:
                TP += 1
            else:
                FP += 1

    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"P": precision, "R": recall, "F1": f1, "TP": TP, "FP": FP, "FN": FN, "TN": TN}


def evaluate_vysotska_open_set(query_descs_mixed, ref_descs, places, n_genuine, patch_size=20):
    S = query_descs_mixed @ ref_descs.T
    n_query = len(query_descs_mixed)
    n_places = len(places)

    frame_to_place = {}
    for p_idx, frames in enumerate(places):
        for f in frames:
            frame_to_place[f] = p_idx

    vysotska = VysotskaDaptiveThreshold(patch_size=patch_size)
    per_query_thresholds, _ = vysotska.compute_thresholds(S)

    place_scores = np.zeros((n_query, n_places))
    for p_idx, frames in enumerate(places):
        place_scores[:, p_idx] = S[:, frames].mean(axis=1)

    TP, FP, FN, TN = 0, 0, 0, 0
    for q_idx in range(n_query):
        is_genuine = q_idx < n_genuine
        gt_place = frame_to_place.get(q_idx, -1) if is_genuine else -1
        scores = place_scores[q_idx].copy()
        theta = per_query_thresholds[q_idx]

        for p_idx in range(n_places):
            if scores[p_idx] < theta:
                scores[p_idx] = -np.inf

        if np.all(scores == -np.inf):
            pred = -1
        else:
            pred = int(np.argmax(scores))

        if pred == -1:
            if gt_place == -1:
                TN += 1
            else:
                FN += 1
        else:
            if is_genuine and pred == gt_place:
                TP += 1
            else:
                FP += 1

    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"P": precision, "R": recall, "F1": f1, "TP": TP, "FP": FP, "FN": FN, "TN": TN}


open_baseline = evaluate_open_set(mixed_query, ref_descs, places, thresh_none, 100)
open_mean_bad = evaluate_open_set(mixed_query, ref_descs, places, thresh_mean_bad, 100)
open_filter_n = evaluate_open_set(mixed_query, ref_descs, places, thresh_filter_n, 100)
open_vysotska = evaluate_vysotska_open_set(mixed_query, ref_descs, places, 100)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

methods = ["Baseline", "mean_bad\n(RCC'25)", "filter_n\n(ours)", "Vysotska\n(ICRA'25)"]
method_colors = ["#95a5a6", "#3498db", "#e74c3c", "#f39c12"]

# Left: closed-set
closed_f1 = [closed_baseline["F1"], closed_mean_bad["F1"],
             closed_filter_n["F1"], closed_vysotska["F1"]]
bars1 = ax1.bar(range(4), closed_f1, color=method_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars1, closed_f1):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}%",
             ha="center", fontsize=10, fontweight="bold")
ax1.set_xticks(range(4))
ax1.set_xticklabels(methods, fontsize=9)
ax1.set_ylim(0, 110)
ax1.set_ylabel("F1 Score (%)", fontsize=11)
ax1.set_title("Closed-Set\n(all queries have a match)", fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.2, axis="y")

# Right: open-set
open_f1 = [open_baseline["F1"], open_mean_bad["F1"],
           open_filter_n["F1"], open_vysotska["F1"]]
bars2 = ax2.bar(range(4), open_f1, color=method_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars2, open_f1):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}%",
             ha="center", fontsize=10, fontweight="bold")
ax2.set_xticks(range(4))
ax2.set_xticklabels(methods, fontsize=9)
ax2.set_ylim(0, 110)
ax2.set_ylabel("F1 Score (%)", fontsize=11)
ax2.set_title("Open-Set\n(50% genuine + 50% distractors)", fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.2, axis="y")

plt.suptitle("The Impact of Open-Set Evaluation on Threshold Methods",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_closed_vs_open.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_closed_vs_open.pdf"), bbox_inches="tight")
plt.close()
print("  Saved fig3_closed_vs_open")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Distractor score distributions — why filter_n wins
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating Figure 4: Distractor Score Distributions...")

# Compute per-place scores for genuine queries, distractors, and cross-place ref
S_genuine = genuine_query @ ref_descs.T
S_distractor = sfu_descs[:n_dist] @ ref_descs.T

# Pick 3 example places (well-separated, medium, hard)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax_idx, p_idx in enumerate(example_places):
    ax = axes[ax_idx]
    frames = places[p_idx]
    s = stats[p_idx]

    # Genuine query scores for this place
    genuine_scores = S_genuine[:100, :][:, frames].mean(axis=1)

    # Distractor scores for this place
    distractor_scores = S_distractor[:, frames].mean(axis=1)

    # Cross-place reference scores (what mean_bad is based on)
    other_frames = [f for i, p in enumerate(places) if i != p_idx for f in p]
    cross_ref_scores = ref_descs[other_frames] @ ref_descs[frames].T
    cross_ref_mean = cross_ref_scores.mean(axis=1)

    ax.hist(cross_ref_mean, bins=40, alpha=0.5, color="#3498db", density=True,
            label="Cross-place ref", edgecolor="none")
    ax.hist(distractor_scores, bins=40, alpha=0.5, color="#e74c3c", density=True,
            label="Distractor queries", edgecolor="none")
    ax.hist(genuine_scores, bins=40, alpha=0.5, color="#27ae60", density=True,
            label="Genuine queries", edgecolor="none")

    # Thresholds
    ax.axvline(s["theta_mean_bad"], color="#3498db", linewidth=2.5, linestyle="-",
               label=f"$\\theta_{{mean\\_bad}}$ = {s['theta_mean_bad']:.3f}")
    ax.axvline(s["theta_filter_n"], color="#8e44ad", linewidth=2.5, linestyle="--",
               label=f"$\\theta_{{filter\\_n}}$ = {s['theta_filter_n']:.3f}")

    ax.set_xlabel("Cosine similarity to place", fontsize=10)
    if ax_idx == 0:
        ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"Place {p_idx} ({len(frames)} frames)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.2)

plt.suptitle("Why filter_n Rejects Distractors: Score Distributions per Place",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_distractor_distributions.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_distractor_distributions.pdf"), bbox_inches="tight")
plt.close()
print("  Saved fig4_distractor_distributions")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Rejection-Recall tradeoff curve
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating Figure 5: Rejection-Recall Tradeoff...")

# Sweep threshold multiplier: theta = mean_bad + alpha * std_bad
alphas = np.linspace(0, 15, 60)
our_recalls = []
our_dist_rejs = []
our_f1s = []

for alpha in alphas:
    thresh = {}
    for p_idx in range(len(places)):
        s = stats[p_idx]
        thresh[p_idx] = s["mean_bad"] + alpha * s["std_bad"]

    result = evaluate_open_set(mixed_query, ref_descs, places, thresh, 100)
    our_recalls.append(result["R"])
    n_dist_rejected = result["TN"]
    our_dist_rejs.append(n_dist_rejected / n_dist * 100)
    our_f1s.append(result["F1"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: Rejection vs Recall
ax1.plot(our_dist_rejs, our_recalls, color="#2c3e50", linewidth=2, zorder=3)

# Mark key points
# alpha=0 is mean_bad
ax1.scatter([our_dist_rejs[0]], [our_recalls[0]], s=120, color="#3498db",
            zorder=5, edgecolors="white", linewidths=1.5)
ax1.annotate("mean_bad\n($\\alpha$=0)", (our_dist_rejs[0], our_recalls[0]),
             textcoords="offset points", xytext=(15, -15), fontsize=9,
             arrowprops=dict(arrowstyle="->", color="#3498db"),
             color="#3498db", fontweight="bold")

# Find filter_n point (alpha ≈ average filter_n across places)
avg_filter_n = np.mean([stats[p]["filter_n"] for p in range(len(places))])
fn_idx = np.argmin(np.abs(alphas - avg_filter_n))
ax1.scatter([our_dist_rejs[fn_idx]], [our_recalls[fn_idx]], s=120, color="#e74c3c",
            zorder=5, edgecolors="white", linewidths=1.5)
ax1.annotate(f"filter_n\n($\\alpha\\approx${avg_filter_n:.1f})",
             (our_dist_rejs[fn_idx], our_recalls[fn_idx]),
             textcoords="offset points", xytext=(15, 10), fontsize=9,
             arrowprops=dict(arrowstyle="->", color="#e74c3c"),
             color="#e74c3c", fontweight="bold")

# Mark Vysotska
ax1.scatter([open_vysotska["TN"] / n_dist * 100], [open_vysotska["R"]],
            s=120, color="#f39c12", marker="D", zorder=5,
            edgecolors="white", linewidths=1.5)
ax1.annotate("Vysotska", (open_vysotska["TN"] / n_dist * 100, open_vysotska["R"]),
             textcoords="offset points", xytext=(-15, -20), fontsize=9,
             arrowprops=dict(arrowstyle="->", color="#f39c12"),
             color="#f39c12", fontweight="bold")

ax1.set_xlabel("Distractor Rejection Rate (%)", fontsize=11)
ax1.set_ylabel("Genuine Recall (%)", fontsize=11)
ax1.set_title("Rejection-Recall Tradeoff", fontsize=12, fontweight="bold")
ax1.set_xlim(-5, 105)
ax1.set_ylim(-5, 105)
ax1.grid(True, alpha=0.2)

# Right: F1 vs alpha
ax2.plot(alphas, our_f1s, color="#2c3e50", linewidth=2)
ax2.axvline(0, color="#3498db", linewidth=1.5, linestyle=":", alpha=0.7, label="mean_bad ($\\alpha$=0)")
ax2.axvline(avg_filter_n, color="#e74c3c", linewidth=1.5, linestyle=":", alpha=0.7,
            label=f"filter_n ($\\alpha\\approx${avg_filter_n:.1f})")

# Mark Vysotska F1
ax2.axhline(open_vysotska["F1"], color="#f39c12", linewidth=1.5, linestyle="--",
            alpha=0.7, label=f"Vysotska F1={open_vysotska['F1']:.1f}%")

best_alpha_idx = np.argmax(our_f1s)
ax2.scatter([alphas[best_alpha_idx]], [our_f1s[best_alpha_idx]], s=100, color="#2c3e50",
            zorder=5, edgecolors="white", linewidths=1.5)
ax2.annotate(f"Best $\\alpha$={alphas[best_alpha_idx]:.1f}\nF1={our_f1s[best_alpha_idx]:.1f}%",
             (alphas[best_alpha_idx], our_f1s[best_alpha_idx]),
             textcoords="offset points", xytext=(15, -10), fontsize=9)

ax2.set_xlabel("Threshold multiplier $\\alpha$ (in $\\theta = \\mu_{bad} + \\alpha \\cdot \\sigma_{bad}$)",
               fontsize=10)
ax2.set_ylabel("F1 Score (%)", fontsize=11)
ax2.set_title("F1 vs Threshold Strictness", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_rejection_recall_tradeoff.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_rejection_recall_tradeoff.pdf"), bbox_inches="tight")
plt.close()
print("  Saved fig5_rejection_recall_tradeoff")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Threshold variation across places
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating Figure 6: Threshold Variation Across Places...")

fig, ax = plt.subplots(figsize=(12, 5))

place_indices = range(len(places))
mean_bads = [stats[p]["mean_bad"] for p in place_indices]
mean_goods = [stats[p]["mean_good"] for p in place_indices]
theta_fns = [stats[p]["theta_filter_n"] for p in place_indices]
filter_ns = [stats[p]["filter_n"] for p in place_indices]
place_sizes = [len(places[p]) for p in place_indices]

x = np.arange(len(places))
width = 0.6

# Shaded region between mean_bad and mean_good
ax.fill_between(x, mean_bads, mean_goods, alpha=0.15, color="#27ae60",
                label="Separability gap")
ax.plot(x, mean_goods, "o-", color="#27ae60", markersize=5, linewidth=1.5,
        label="$\\mu_{good}$ (within-place)")
ax.plot(x, mean_bads, "s-", color="#e74c3c", markersize=5, linewidth=1.5,
        label="$\\mu_{bad}$ (cross-place) = $\\theta_{mean\\_bad}$")
ax.plot(x, theta_fns, "D-", color="#8e44ad", markersize=5, linewidth=1.5,
        label="$\\theta_{filter\\_n}$")

# Annotate filter_n values
for i in range(len(places)):
    ax.annotate(f"n={filter_ns[i]:.0f}", (x[i], theta_fns[i]),
                textcoords="offset points", xytext=(0, 8), fontsize=7,
                ha="center", color="#8e44ad")

ax.set_xticks(x)
ax.set_xticklabels([f"P{i}\n({place_sizes[i]})" for i in range(len(places))],
                    fontsize=7, rotation=0)
ax.set_xlabel("Place (frame count)", fontsize=11)
ax.set_ylabel("Cosine similarity", fontsize=11)
ax.set_title("Per-Place Thresholds: Adaptivity to Place Characteristics", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig6_threshold_variation.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "fig6_threshold_variation.pdf"), bbox_inches="tight")
plt.close()
print("  Saved fig6_threshold_variation")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Similarity matrix with thresholds overlaid
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating Figure 7: Similarity Matrix with Thresholds...")

S_full = query_descs @ ref_descs.T

# Vysotska per-query thresholds on closed-set
vysotska_closed = VysotskaDaptiveThreshold(patch_size=20)
vysotska_pq_thresholds, _ = vysotska_closed.compute_thresholds(S_full)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel 1: Raw similarity matrix
ax = axes[0]
im = ax.imshow(S_full, aspect="auto", cmap="hot", vmin=0, vmax=1, origin="upper")
# Draw place boundaries
for p_idx, frames in enumerate(places):
    start, end = frames[0], frames[-1] + 1
    # Vertical lines (reference)
    ax.axvline(start - 0.5, color="cyan", linewidth=0.5, alpha=0.6)
    # Horizontal lines (query — same correspondence)
    ax.axhline(start - 0.5, color="cyan", linewidth=0.5, alpha=0.6)
ax.set_xlabel("Reference frame", fontsize=10)
ax.set_ylabel("Query frame", fontsize=10)
ax.set_title("Similarity Matrix S\nwith place boundaries", fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity")

# Panel 2: Our per-place thresholds as a mask
ax2 = axes[1]
# Build threshold matrix: for each query-ref pair, threshold is the ref's place threshold
thresh_matrix_ours = np.zeros_like(S_full)
frame_to_place_map = {}
for p_idx, frames in enumerate(places):
    for f in frames:
        frame_to_place_map[f] = p_idx

for r_idx in range(S_full.shape[1]):
    p = frame_to_place_map.get(r_idx, -1)
    if p >= 0:
        thresh_matrix_ours[:, r_idx] = stats[p]["theta_filter_n"]

# Show which entries pass threshold
pass_mask = (S_full >= thresh_matrix_ours).astype(float)
# Overlay: dim the matrix where threshold is not met
S_masked = S_full.copy()
S_masked[~(S_full >= thresh_matrix_ours)] *= 0.2

im2 = ax2.imshow(S_masked, aspect="auto", cmap="hot", vmin=0, vmax=1, origin="upper")
for p_idx, frames in enumerate(places):
    start = frames[0]
    ax2.axvline(start - 0.5, color="cyan", linewidth=0.5, alpha=0.6)
    ax2.axhline(start - 0.5, color="cyan", linewidth=0.5, alpha=0.6)

# Draw threshold level per place as horizontal annotation
for p_idx, frames in enumerate(places):
    mid_r = (frames[0] + frames[-1]) / 2
    theta = stats[p_idx]["theta_filter_n"]
    ax2.annotate(f"{theta:.2f}", (mid_r, -3), fontsize=5, ha="center",
                color="#8e44ad", fontweight="bold", clip_on=False)

ax2.set_xlabel("Reference frame", fontsize=10)
ax2.set_ylabel("Query frame", fontsize=10)
ax2.set_title("Our filter_n threshold\n(per-place, entries below threshold dimmed)",
              fontsize=11, fontweight="bold")
plt.colorbar(im2, ax=ax2, shrink=0.8, label="Cosine similarity")

# Panel 3: Vysotska per-query thresholds
ax3 = axes[2]
S_masked_v = S_full.copy()
for q_idx in range(S_full.shape[0]):
    S_masked_v[q_idx, S_full[q_idx] < vysotska_pq_thresholds[q_idx]] *= 0.2

im3 = ax3.imshow(S_masked_v, aspect="auto", cmap="hot", vmin=0, vmax=1, origin="upper")
for p_idx, frames in enumerate(places):
    start = frames[0]
    ax3.axvline(start - 0.5, color="cyan", linewidth=0.5, alpha=0.6)
    ax3.axhline(start - 0.5, color="cyan", linewidth=0.5, alpha=0.6)

# Draw Vysotska threshold as a line on the right edge
# Create a twin axis to show the per-query threshold
ax3_twin = ax3.twinx()
ax3_twin.plot(vysotska_pq_thresholds, range(len(vysotska_pq_thresholds)),
              color="#f39c12", linewidth=1.5, alpha=0.8)
ax3_twin.set_ylim(len(vysotska_pq_thresholds) - 0.5, -0.5)
ax3_twin.set_ylabel("Vysotska $\\theta$(query)", fontsize=9, color="#f39c12")
ax3_twin.tick_params(axis="y", colors="#f39c12", labelsize=8)

ax3.set_xlabel("Reference frame", fontsize=10)
ax3.set_ylabel("Query frame", fontsize=10)
ax3.set_title("Vysotska threshold\n(per-query, entries below threshold dimmed)",
              fontsize=11, fontweight="bold")
plt.colorbar(im3, ax=ax3, shrink=0.8, label="Cosine similarity")

plt.suptitle("Similarity Matrix with Threshold Overlays: Per-Place (Ours) vs Per-Query (Vysotska)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig7_similarity_matrix_thresholds.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "fig7_similarity_matrix_thresholds.pdf"), bbox_inches="tight")
plt.close()
print("  Saved fig7_similarity_matrix_thresholds")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Method pipeline diagram (our method vs Vysotska)
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating Figure 8: Method Pipeline Diagram...")

fig, (ax_ours, ax_vys) = plt.subplots(2, 1, figsize=(14, 7))

def draw_pipeline_box(ax, x, y, w, h, text, color, fontsize=8):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor="#2c3e50",
                          linewidth=1.5, alpha=0.85)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color="#2c3e50"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))

# ── Our Pipeline ──
ax_ours.set_xlim(-0.5, 10.5)
ax_ours.set_ylim(-1, 2)
ax_ours.axis("off")
ax_ours.set_title("Our Method: Per-Place Thresholds from Reference Data",
                   fontsize=12, fontweight="bold", color="#e74c3c", pad=10)

# Reference phase (top row)
boxes_ours = [
    (0.8, 1.0, 1.4, 0.7, "Reference\nImage Stream", "#dfe6e9"),
    (2.8, 1.0, 1.4, 0.7, "Online Place\nDiscovery", "#74b9ff"),
    (5.0, 1.0, 1.5, 0.7, "Compute\nPer-Place Stats\n($\\mu_{bad}, \\sigma_{bad}$)", "#a29bfe"),
    (7.3, 1.0, 1.5, 0.7, "Per-Place\nThresholds\n$\\theta_p = f(\\mu_{bad}, \\sigma_{bad})$", "#fd79a8"),
]
for x, y, w, h, text, color in boxes_ours:
    draw_pipeline_box(ax_ours, x, y, w, h, text, color)

draw_arrow(ax_ours, 1.5, 1.0, 2.1, 1.0)
draw_arrow(ax_ours, 3.5, 1.0, 4.25, 1.0)
draw_arrow(ax_ours, 5.75, 1.0, 6.55, 1.0)

# Query phase (bottom row)
boxes_query = [
    (0.8, -0.2, 1.4, 0.7, "New Query\nImage", "#dfe6e9"),
    (2.8, -0.2, 1.4, 0.7, "Compute\nSimilarity to\nAll Ref Images", "#74b9ff"),
    (5.0, -0.2, 1.5, 0.7, "Average\nScore per\nPlace", "#a29bfe"),
    (7.3, -0.2, 1.5, 0.7, "Compare\nScore vs $\\theta_p$", "#fd79a8"),
    (9.3, -0.2, 1.2, 0.7, "Accept /\nReject", "#55efc4"),
]
for x, y, w, h, text, color in boxes_query:
    draw_pipeline_box(ax_ours, x, y, w, h, text, color)

draw_arrow(ax_ours, 1.5, -0.2, 2.1, -0.2)
draw_arrow(ax_ours, 3.5, -0.2, 4.25, -0.2)
draw_arrow(ax_ours, 5.75, -0.2, 6.55, -0.2)
draw_arrow(ax_ours, 8.05, -0.2, 8.7, -0.2)

# Arrow from thresholds down to comparison
draw_arrow(ax_ours, 7.3, 0.65, 7.3, 0.15, color="#e74c3c")

# Labels
ax_ours.text(-0.3, 1.0, "Offline\n(reference)", fontsize=8, ha="center", va="center",
             style="italic", color="#636e72")
ax_ours.text(-0.3, -0.2, "Online\n(query)", fontsize=8, ha="center", va="center",
             style="italic", color="#636e72")

# ── Vysotska Pipeline ──
ax_vys.set_xlim(-0.5, 10.5)
ax_vys.set_ylim(-0.5, 1.5)
ax_vys.axis("off")
ax_vys.set_title("Vysotska et al.: Per-Query Thresholds from Similarity Matrix",
                  fontsize=12, fontweight="bold", color="#f39c12", pad=10)

boxes_vys = [
    (0.8, 0.5, 1.4, 0.7, "Full Query +\nReference Set", "#dfe6e9"),
    (2.8, 0.5, 1.4, 0.7, "Compute Full\nSimilarity\nMatrix S", "#fdcb6e"),
    (4.8, 0.5, 1.2, 0.7, "Extract\n20x20 Patch\naround match", "#ffeaa7"),
    (6.6, 0.5, 1.2, 0.7, "KS Test\n(bimodal?)", "#fab1a0"),
    (8.3, 0.5, 1.2, 0.7, "GMM Fit +\nDecision\nBoundary", "#ff7675"),
    (9.9, 0.5, 1.0, 0.7, "Kalman\nSmooth\n$\\theta_q$", "#55efc4"),
]
for x, y, w, h, text, color in boxes_vys:
    draw_pipeline_box(ax_vys, x, y, w, h, text, color)

draw_arrow(ax_vys, 1.5, 0.5, 2.1, 0.5)
draw_arrow(ax_vys, 3.5, 0.5, 4.2, 0.5)
draw_arrow(ax_vys, 5.4, 0.5, 6.0, 0.5)
draw_arrow(ax_vys, 7.2, 0.5, 7.7, 0.5)
draw_arrow(ax_vys, 8.9, 0.5, 9.4, 0.5)

# Note
ax_vys.text(5.0, -0.2, "Requires all queries upfront to build S; threshold depends on each query's similarity pattern",
            fontsize=9, ha="center", va="center", style="italic", color="#636e72",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffeaa7", alpha=0.3))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig8_method_pipeline.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "fig8_method_pipeline.pdf"), bbox_inches="tight")
plt.close()
print("  Saved fig8_method_pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: Varying distractor ratios
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating Figure 9: Varying Distractor Ratios...")

# Use first 100 genuine queries and up to 100 SFU distractors
# Vary ratio: 0%, 10%, 25%, 50%, 75%, 90%
n_genuine_fixed = 100
ratios = [0, 10, 25, 50, 75, 90]

results_by_method = {
    "Baseline": [],
    "mean_bad (RCC'25)": [],
    "filter_n (ours)": [],
    "Vysotska (ICRA'25)": [],
}

for ratio in ratios:
    n_dist_for_ratio = int(n_genuine_fixed * ratio / (100 - ratio)) if ratio < 100 else n_genuine_fixed * 9
    n_dist_for_ratio = min(n_dist_for_ratio, len(sfu_descs))
    total = n_genuine_fixed + n_dist_for_ratio

    if n_dist_for_ratio > 0:
        mixed = np.vstack([genuine_query, sfu_descs[:n_dist_for_ratio]])
    else:
        mixed = genuine_query.copy()

    # Baseline
    r = evaluate_open_set(mixed, ref_descs, places, thresh_none, n_genuine_fixed)
    results_by_method["Baseline"].append(r["F1"])

    # mean_bad
    r = evaluate_open_set(mixed, ref_descs, places, thresh_mean_bad, n_genuine_fixed)
    results_by_method["mean_bad (RCC'25)"].append(r["F1"])

    # filter_n
    r = evaluate_open_set(mixed, ref_descs, places, thresh_filter_n, n_genuine_fixed)
    results_by_method["filter_n (ours)"].append(r["F1"])

    # Vysotska
    r = evaluate_vysotska_open_set(mixed, ref_descs, places, n_genuine_fixed)
    results_by_method["Vysotska (ICRA'25)"].append(r["F1"])

    bl = results_by_method["Baseline"][-1]
    mb = results_by_method["mean_bad (RCC'25)"][-1]
    fn = results_by_method["filter_n (ours)"][-1]
    vy = results_by_method["Vysotska (ICRA'25)"][-1]
    print(f"  Ratio {ratio}%: Baseline={bl:.1f}  mean_bad={mb:.1f}  filter_n={fn:.1f}  Vysotska={vy:.1f}")

fig, ax = plt.subplots(figsize=(10, 6))

method_styles = {
    "Baseline": ("#95a5a6", "o", "-"),
    "mean_bad (RCC'25)": ("#3498db", "s", "-"),
    "filter_n (ours)": ("#e74c3c", "D", "-"),
    "Vysotska (ICRA'25)": ("#f39c12", "^", "--"),
}

for method, f1_values in results_by_method.items():
    color, marker, ls = method_styles[method]
    ax.plot(ratios, f1_values, color=color, marker=marker, markersize=8,
            linewidth=2, linestyle=ls, label=method, alpha=0.9)

ax.set_xlabel("Distractor Ratio (%)", fontsize=12)
ax.set_ylabel("F1 Score (%)", fontsize=12)
ax.set_title("Robustness to Distractor Ratio: F1 Score as Unknown Queries Increase",
             fontsize=13, fontweight="bold")
ax.set_xticks(ratios)
ax.set_xticklabels([f"{r}%" for r in ratios])
ax.legend(fontsize=10, loc="lower left")
ax.grid(True, alpha=0.2)
ax.set_ylim(0, 105)

# Annotation
ax.annotate("Closed-set\n(no distractors)", (0, results_by_method["Baseline"][0]),
            textcoords="offset points", xytext=(15, -15), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray"), color="gray")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig9_distractor_ratios.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(OUTPUT_DIR, "fig9_distractor_ratios.pdf"), bbox_inches="tight")
plt.close()
print("  Saved fig9_distractor_ratios")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nAll figures saved to: {OUTPUT_DIR}/")
print("Files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")
