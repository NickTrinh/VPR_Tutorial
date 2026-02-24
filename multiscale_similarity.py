"""
Sequential Place Boundary Detection via Similarity Curves.

Detects place boundaries from sequential image streams by analyzing
cosine similarity drop-off between an anchor image and subsequent frames.
The hypothesis: similarity drops more steeply when crossing a place boundary.
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_extraction.common import get_device
from feature_extraction.feature_extractor_eigenplaces import EigenPlacesFeatureExtractor
from utils import normalize_l2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential place boundary detection via similarity curves"
    )
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to folder of sequential images")
    parser.add_argument("--max_offset", type=int, default=20,
                        help="Frames ahead to compare (default: 20)")
    parser.add_argument("--step", type=int, default=5,
                        help="Anchor spacing (default: 5)")
    parser.add_argument("--max_images", type=int, default=500,
                        help="Limit images loaded (default: 500)")
    parser.add_argument("--output_dir", type=str,
                        default="./results/similarity_analysis",
                        help="Save PNGs here")
    parser.add_argument("--device", type=str, default=None,
                        help="Force cuda/cpu (default: auto-detect)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Feature extraction batch size (default: 64)")
    return parser.parse_args()


def load_images(image_dir, max_images):
    """Glob for *.png + *.jpg, sort by filename, take first max_images."""
    image_dir = Path(image_dir)
    paths = sorted(
        list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    )
    paths = paths[:max_images]
    print(f"Loading {len(paths)} images from {image_dir} ...")
    imgs = []
    for p in tqdm(paths, desc="Loading images"):
        img = np.array(Image.open(p).convert("RGB"))
        imgs.append(img)
    return imgs


def extract_features(imgs, output_dir, image_dir, batch_size):
    """Extract features with caching to avoid recomputation."""
    cache_path = os.path.join(output_dir, "features_cache.pkl")
    cache_key = (os.path.basename(image_dir.rstrip("/")), len(imgs))

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if cache.get("key") == cache_key:
            print(f"Loaded cached features for {cache_key}")
            return cache["features"]

    print("Extracting features with EigenPlaces (ResNet50, dim=2048) ...")
    extractor = EigenPlacesFeatureExtractor()

    # Process in chunks to manage memory
    all_feats = []
    for i in range(0, len(imgs), batch_size):
        chunk = imgs[i : i + batch_size]
        feats = extractor.compute_features(chunk)
        all_feats.append(feats)
    features = np.concatenate(all_feats, axis=0)

    os.makedirs(output_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"key": cache_key, "features": features}, f)
    print(f"Cached features to {cache_path}")

    return features


def compute_offset_similarity(features, step, max_offset):
    """Compute cosine similarity between each anchor and subsequent frames.

    Returns (num_anchors, max_offset) array and list of anchor indices.
    """
    n = len(features)
    anchors = list(range(0, n - max_offset, step))
    sim_matrix = np.zeros((len(anchors), max_offset), dtype=np.float32)

    for i, anchor in enumerate(anchors):
        for offset in range(1, max_offset + 1):
            idx = anchor + offset
            sim_matrix[i, offset - 1] = np.dot(features[anchor], features[idx])

    return sim_matrix, anchors


def compute_drop_rate(sim_matrix):
    """Per anchor: linear fit slope across offsets."""
    num_anchors, max_offset = sim_matrix.shape
    offsets = np.arange(1, max_offset + 1)
    slopes = np.zeros(num_anchors, dtype=np.float32)

    for i in range(num_anchors):
        coeffs = np.polyfit(offsets, sim_matrix[i], deg=1)
        slopes[i] = coeffs[0]

    return slopes


def plot_similarity_heatmap(sim_matrix, anchors, output_dir):
    """Heatmap of offset similarity matrix."""
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(sim_matrix, aspect="auto", cmap="viridis",
                   interpolation="nearest")
    ax.set_xlabel("Offset (frames ahead)")
    ax.set_ylabel("Anchor frame index")

    # Label y-axis with actual frame indices (subsample for readability)
    num_ticks = min(15, len(anchors))
    tick_positions = np.linspace(0, len(anchors) - 1, num_ticks, dtype=int)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([anchors[i] for i in tick_positions])

    # Label x-axis with offset numbers
    x_ticks = np.arange(0, sim_matrix.shape[1], max(1, sim_matrix.shape[1] // 10))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks + 1)

    ax.set_title("Cosine Similarity: Anchor vs Subsequent Frames")
    fig.colorbar(im, ax=ax, label="Cosine Similarity")
    plt.tight_layout()

    path = os.path.join(output_dir, "similarity_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_similarity_curves(sim_matrix, anchors, max_offset, output_dir):
    """5 example anchor curves evenly spaced through sequence."""
    fig, ax = plt.subplots(figsize=(10, 6))
    offsets = np.arange(1, max_offset + 1)

    num_examples = min(5, len(anchors))
    example_indices = np.linspace(0, len(anchors) - 1, num_examples, dtype=int)

    for idx in example_indices:
        ax.plot(offsets, sim_matrix[idx], marker="o", markersize=3,
                label=f"Anchor frame {anchors[idx]}")

    ax.set_xlabel("Offset (frames ahead)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Similarity Decay Curves for Selected Anchors")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "similarity_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_drop_rate(slopes, anchors, output_dir):
    """Top: raw slope vs anchor index; Bottom: rolling mean (window=5)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(anchors, slopes, color="steelblue", linewidth=0.8)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Slope (drop rate)")
    ax1.set_title("Similarity Drop Rate per Anchor")
    ax1.grid(True, alpha=0.3)

    # Rolling mean with window=5
    window = 5
    if len(slopes) >= window:
        kernel = np.ones(window) / window
        rolling_mean = np.convolve(slopes, kernel, mode="valid")
        rolling_anchors = anchors[window // 2 : window // 2 + len(rolling_mean)]
        ax2.plot(rolling_anchors, rolling_mean, color="darkorange", linewidth=1.5)
    else:
        ax2.plot(anchors, slopes, color="darkorange", linewidth=1.5)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Anchor Frame Index")
    ax2.set_ylabel("Smoothed Slope")
    ax2.set_title(f"Rolling Mean Drop Rate (window={window})")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "drop_rate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def print_summary(num_images, anchors, sim_matrix, max_offset, output_files):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Images loaded:    {num_images}")
    print(f"  Anchor count:     {len(anchors)}")
    print(f"  Max offset:       {max_offset}")
    print()
    sim_offset1 = sim_matrix[:, 0]
    sim_offsetN = sim_matrix[:, -1]
    print(f"  Similarity at offset 1:  "
          f"mean={sim_offset1.mean():.4f}  "
          f"min={sim_offset1.min():.4f}  "
          f"max={sim_offset1.max():.4f}")
    print(f"  Similarity at offset {max_offset}: "
          f"mean={sim_offsetN.mean():.4f}  "
          f"min={sim_offsetN.min():.4f}  "
          f"max={sim_offsetN.max():.4f}")
    print()
    print("  Output files:")
    for f in output_files:
        print(f"    {f}")
    print("=" * 60)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load images
    imgs = load_images(args.image_dir, args.max_images)
    if len(imgs) <= args.max_offset:
        raise ValueError(
            f"Not enough images ({len(imgs)}) for max_offset={args.max_offset}. "
            f"Need at least {args.max_offset + 1} images."
        )

    # 2. Extract features (with caching)
    features = extract_features(imgs, args.output_dir, args.image_dir,
                                args.batch_size)

    # 3. Normalize & compute offset similarity matrix
    features = normalize_l2(features)
    sim_matrix, anchors = compute_offset_similarity(
        features, args.step, args.max_offset
    )

    # 4. Compute drop rate
    slopes = compute_drop_rate(sim_matrix)
    anchors_arr = np.array(anchors)

    # 5. Generate PNGs
    output_files = []
    output_files.append(plot_similarity_heatmap(sim_matrix, anchors, args.output_dir))
    output_files.append(plot_similarity_curves(
        sim_matrix, anchors, args.max_offset, args.output_dir
    ))
    output_files.append(plot_drop_rate(slopes, anchors_arr, args.output_dir))

    # 6. Print summary
    print_summary(len(imgs), anchors, sim_matrix, args.max_offset, output_files)


if __name__ == "__main__":
    main()
