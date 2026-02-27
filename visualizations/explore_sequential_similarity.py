"""
Explore sequential image similarity in GardensPoint (full, 200-image walkway).

Extracts EigenPlaces features for a single condition (day_left by default),
then computes and visualises the similarity signal at multiple temporal offsets.

Outputs (saved to results/visualizations/gardens_point_sequential/):
  - consecutive_sim.png     : sim(t, t+1) over the full sequence
  - multiscale_sim.png      : sim(t, t+k) for k = 1..max_offset, a few anchor rows
  - similarity_matrix.png   : full NxN self-similarity heatmap
  - scene_samples.png       : image thumbnails at key positions
  - offset_heatmap.png      : heatmap of sim(t, t+k) for all t and k

Usage (from project root):
    python visualizations/explore_sequential_similarity.py \
        --condition day_left --max_offset 20 --output_dir results/visualizations/gardens_point_sequential
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
import matplotlib.gridspec as gridspec
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import normalize_l2


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction (with caching)
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(img_paths, cache_dir, descriptor_name="eigenplaces"):
    """Extract or load cached EigenPlaces descriptors for img_paths."""
    os.makedirs(cache_dir, exist_ok=True)
    descriptors = []

    cached = [os.path.join(cache_dir, f"img_{i}_descriptor.pkl")
              for i in range(len(img_paths))]
    missing = [i for i, p in enumerate(cached) if not os.path.exists(p)]

    if missing:
        print(f"  Extracting {descriptor_name} features for {len(missing)} images ...")
        # Lazy-import heavy dependencies
        import torch
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader, Dataset as TorchDataset

        class _ImgDS(TorchDataset):
            def __init__(self, paths, tfm):
                self.paths = paths
                self.tfm = tfm
            def __len__(self): return len(self.paths)
            def __getitem__(self, i):
                img = Image.open(self.paths[i]).convert("RGB")
                return self.tfm(img), i

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}")

        tfm = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        if descriptor_name == "eigenplaces":
            model = torch.hub.load(
                "gmberton/eigenplaces", "get_trained_model",
                backbone="ResNet50", fc_output_dim=2048
            )
        elif descriptor_name == "cosplace":
            model = torch.hub.load(
                "gmberton/cosplace", "get_trained_model",
                backbone="ResNet50", fc_output_dim=2048
            )
        else:
            raise ValueError(f"Unknown descriptor: {descriptor_name}")

        model = model.to(device).eval()

        # Build a dataset of only the missing images
        missing_paths = [img_paths[i] for i in missing]
        ds = _ImgDS(missing_paths, tfm)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

        tmp = {}
        with torch.no_grad():
            for batch_imgs, batch_idx in loader:
                feats = model(batch_imgs.to(device)).cpu().numpy()
                for feat, orig_i in zip(feats, batch_idx.numpy()):
                    real_idx = missing[orig_i]
                    tmp[real_idx] = feat

        for idx, feat in tmp.items():
            with open(cached[idx], "wb") as f:
                pickle.dump({"descriptor": feat}, f)
        print("  Done extracting.")

    # Load all from cache in order
    for i, cp in enumerate(cached):
        with open(cp, "rb") as f:
            d = pickle.load(f)
        desc = d["descriptor"].reshape(-1).astype(np.float32)
        descriptors.append(desc)

    return np.stack(descriptors, axis=0)  # (N, D)


# ──────────────────────────────────────────────────────────────────────────────
# Similarity computations
# ──────────────────────────────────────────────────────────────────────────────

def compute_similarity_matrix(features):
    """Full NxN cosine similarity matrix."""
    F = normalize_l2(features)
    return F @ F.T  # (N, N)


def consecutive_sim(S):
    """sim(t, t+1) for all t."""
    N = S.shape[0]
    return np.array([S[t, t + 1] for t in range(N - 1)])


def offset_sim(S, max_offset):
    """
    Returns array of shape (N - max_offset, max_offset) where
    result[t, k] = S[t, t + k + 1]  (k=0 → offset 1)
    """
    N = S.shape[0]
    T = N - max_offset
    out = np.zeros((T, max_offset))
    for k in range(1, max_offset + 1):
        for t in range(T):
            out[t, k - 1] = S[t, t + k]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_consecutive(csim, output_dir):
    fig, ax = plt.subplots(figsize=(14, 4))
    t = np.arange(len(csim))
    ax.plot(t, csim, color="#2980b9", linewidth=1.2, label="sim(t, t+1)")
    ax.fill_between(t, csim, csim.min(), alpha=0.15, color="#2980b9")

    # Mark local minima (potential boundaries)
    from scipy.signal import argrelmin
    mins = argrelmin(csim, order=5)[0]
    low_mins = mins[csim[mins] < np.percentile(csim, 25)]
    ax.scatter(low_mins, csim[low_mins], color="#e74c3c", zorder=5, s=40,
               label="Notable dips (candidate boundaries)")

    ax.set_xlabel("Frame index t")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Consecutive-frame similarity  sim(t, t+1)\n"
                 "Dips suggest scene / place transitions")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = os.path.join(output_dir, "consecutive_sim.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return low_mins


def plot_similarity_matrix(S, output_dir):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(S, cmap="hot", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xlabel("Frame j")
    ax.set_ylabel("Frame i")
    ax.set_title("Self-similarity matrix S[i,j] = cos(d(i), d(j))\n"
                 "Bright diagonal = self-match; bright blocks = coherent places")
    plt.tight_layout()
    path = os.path.join(output_dir, "similarity_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_offset_heatmap(off_sim, output_dir, max_offset):
    """Heatmap: x=frame t, y=offset k, colour=similarity."""
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(off_sim.T, aspect="auto", origin="lower",
                   cmap="viridis", vmin=0, vmax=1,
                   extent=[0, off_sim.shape[0], 1, max_offset])
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xlabel("Anchor frame t")
    ax.set_ylabel("Temporal offset k")
    ax.set_title("sim(t, t+k) for all t and k\n"
                 "Dark vertical stripes = similarity drops at all scales → place boundary")
    plt.tight_layout()
    path = os.path.join(output_dir, "offset_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_multiscale_profiles(off_sim, candidate_boundaries, output_dir, max_offset):
    """For a few selected anchor frames, plot sim vs offset."""
    N = off_sim.shape[0]

    # Choose anchors: some candidates + some stable mid-place frames
    stable_frames = []
    cands = sorted(candidate_boundaries)
    if len(cands) >= 2:
        for i in range(len(cands) - 1):
            mid = (cands[i] + cands[i + 1]) // 2
            if 0 <= mid < N:
                stable_frames.append(mid)

    # Keep at most 4 candidates and 4 stable
    anchors_cand = [c for c in cands if c < N][:4]
    anchors_stable = stable_frames[:4]
    all_anchors = sorted(set(anchors_cand + anchors_stable))[:8]

    if not all_anchors:
        all_anchors = list(range(0, N, N // 6))[:6]

    offsets = np.arange(1, max_offset + 1)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    axes = axes.flatten()

    for ax_idx, (t, ax) in enumerate(zip(all_anchors, axes)):
        profile = off_sim[t, :]
        color = "#e74c3c" if t in anchors_cand else "#27ae60"
        label = "boundary candidate" if t in anchors_cand else "within-place"
        ax.plot(offsets, profile, color=color, linewidth=1.8)
        ax.fill_between(offsets, profile, profile.min(), alpha=0.2, color=color)
        ax.set_title(f"Anchor t={t}\n({label})", fontsize=9, color=color)
        ax.set_xlabel("Offset k", fontsize=8)
        ax.set_ylabel("sim(t, t+k)", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)

    for ax in axes[len(all_anchors):]:
        ax.set_visible(False)

    fig.suptitle("Multi-scale similarity profiles: sim(t, t+k) vs k\n"
                 "Red = near boundary candidate  |  Green = within stable region",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "multiscale_profiles.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_scene_samples(img_paths, candidate_boundaries, csim, output_dir, n_samples=16):
    """Show image thumbnails at candidate boundary frames and their neighbours."""
    N = len(img_paths)
    # Pick frames: each candidate ± 1
    frames = set()
    for b in sorted(candidate_boundaries)[:6]:
        for delta in [-2, -1, 0, 1, 2]:
            if 0 <= b + delta < N:
                frames.add(b + delta)
    # Fill up to n_samples with evenly spaced frames
    step = max(1, N // (n_samples - len(frames)))
    for i in range(0, N, step):
        frames.add(i)
    frames = sorted(frames)[:n_samples]

    cols = 8
    rows = (len(frames) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.0))
    axes = np.array(axes).flatten()

    for ax_idx, t in enumerate(frames):
        img = Image.open(img_paths[t]).resize((160, 90))
        sim_val = csim[t] if t < len(csim) else float("nan")
        is_cand = t in candidate_boundaries
        axes[ax_idx].imshow(np.array(img))
        axes[ax_idx].set_title(
            f"t={t}\nsim→next={sim_val:.3f}",
            fontsize=7,
            color="#e74c3c" if is_cand else "black"
        )
        if is_cand:
            for spine in axes[ax_idx].spines.values():
                spine.set_edgecolor("#e74c3c")
                spine.set_linewidth(2)
        axes[ax_idx].axis("off")

    for ax in axes[len(frames):]:
        ax.set_visible(False)

    fig.suptitle("Scene samples — red border = boundary candidate\n"
                 "sim→next: cosine similarity to the immediately following frame",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "scene_samples.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="images/GardensPoint")
    p.add_argument("--condition", default="day_left",
                   choices=["day_left", "day_right", "night_right"])
    p.add_argument("--descriptor", default="eigenplaces",
                   choices=["eigenplaces", "cosplace"])
    p.add_argument("--max_offset", type=int, default=20,
                   help="Maximum temporal offset for multi-scale profiles")
    p.add_argument("--output_dir",
                   default="results/visualizations/gardens_point_sequential")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Find images ────────────────────────────────────────────────────────
    img_dir = os.path.join(args.data_dir, args.condition)
    img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    if not img_paths:
        raise FileNotFoundError(f"No jpg images found in {img_dir}")
    print(f"Found {len(img_paths)} images in {img_dir}")

    # ── 2. Extract / load features ────────────────────────────────────────────
    cache_dir = os.path.join(
        "cache", "GardensPoint", args.condition, args.descriptor
    )
    features = extract_features(img_paths, cache_dir, args.descriptor)
    print(f"Features: {features.shape}")

    # ── 3. Compute similarity ─────────────────────────────────────────────────
    print("Computing similarity matrix ...")
    S = compute_similarity_matrix(features)
    csim = consecutive_sim(S)
    off_sim = offset_sim(S, args.max_offset)

    print(f"Consecutive sim — mean: {csim.mean():.3f}  min: {csim.min():.3f}  "
          f"max: {csim.max():.3f}  std: {csim.std():.3f}")

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    print("Generating plots ...")
    boundaries = plot_consecutive(csim, args.output_dir)
    plot_similarity_matrix(S, args.output_dir)
    plot_offset_heatmap(off_sim, args.output_dir, args.max_offset)
    plot_multiscale_profiles(off_sim, boundaries, args.output_dir, args.max_offset)
    plot_scene_samples(img_paths, set(boundaries), csim, args.output_dir)

    print(f"\nDone. Results saved to: {args.output_dir}")
    print(f"Candidate boundary frames: {sorted(boundaries)}")


if __name__ == "__main__":
    main()
