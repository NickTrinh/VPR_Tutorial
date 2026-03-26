"""
Extract DINOv2 SALAD descriptors for Nordland-500.

Must be run on a GPU node (ciscluster).
Produces cached descriptors compatible with the validation script.

Usage:
    python -m experiments.extract_dinov2_salad

Requires: torch.hub access to serizba/salad
"""

import os
import sys
import pickle
import time

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


N = 500
IMAGE_DIR_WINTER = "images/Nordland_Mini/winter"
IMAGE_DIR_SUMMER = "images/Nordland_Mini/summer"
CACHE_DIR_WINTER = "cache/Nordland_salad/winter/dinov2_salad"
CACHE_DIR_SUMMER = "cache/Nordland_salad/summer/dinov2_salad"
BATCH_SIZE = 32


def get_image_paths(image_dir, n):
    """Get first n image paths sorted by place number."""
    all_files = sorted(os.listdir(image_dir))
    all_files = [f for f in all_files if f.endswith('.png') or f.endswith('.jpg')]
    paths = [os.path.join(image_dir, f) for f in all_files[:n]]
    return paths


def extract_features(image_paths, cache_dir, model, preprocess, device):
    """Extract and cache DINOv2 SALAD features."""
    os.makedirs(cache_dir, exist_ok=True)

    # Check which are already cached
    to_extract = []
    for i, path in enumerate(image_paths):
        cache_path = os.path.join(cache_dir, f"img_{i}_descriptor.pkl")
        if not os.path.exists(cache_path):
            to_extract.append((i, path))

    if not to_extract:
        print(f"  All {len(image_paths)} descriptors already cached in {cache_dir}")
        return

    print(f"  Extracting {len(to_extract)} descriptors (batch_size={BATCH_SIZE})...")
    model.eval()

    for batch_start in tqdm(range(0, len(to_extract), BATCH_SIZE)):
        batch = to_extract[batch_start:batch_start + BATCH_SIZE]
        imgs = []
        for _, path in batch:
            img = Image.open(path).convert("RGB")
            imgs.append(preprocess(img))

        batch_tensor = torch.stack(imgs).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            features = model(batch_tensor)  # (B, 8448)

        features_np = features.cpu().numpy().astype(np.float32)

        for j, (idx, _) in enumerate(batch):
            cache_path = os.path.join(cache_dir, f"img_{idx}_descriptor.pkl")
            with open(cache_path, "wb") as f:
                pickle.dump({"descriptor": features_np[j]}, f)

    print(f"  Done. Cached to {cache_dir}")


def main():
    print("=" * 60)
    print("DINOv2 SALAD Feature Extraction for Nordland-500")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run this on a GPU node.")
        print("  ssh node001")
        print("  conda activate vprtutorial")
        print("  python -m experiments.extract_dinov2_salad")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {device} ({torch.cuda.get_device_name()})")

    # Load model
    print("\nLoading DINOv2 SALAD model...")
    t0 = time.time()
    model = torch.hub.load("serizba/salad", "dinov2_salad")
    model = model.to(device)
    model.eval()
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Test output dimension
    dummy = torch.randn(1, 3, 322, 322).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"  Output dimension: {out.shape[1]}")

    # Preprocessing (must be divisible by 14, paper uses 322)
    preprocess = transforms.Compose([
        transforms.Resize((322, 322),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Get image paths
    winter_paths = get_image_paths(IMAGE_DIR_WINTER, N)
    summer_paths = get_image_paths(IMAGE_DIR_SUMMER, N)
    print(f"\nWinter images: {len(winter_paths)}")
    print(f"Summer images: {len(summer_paths)}")

    # Extract winter (reference)
    print(f"\n--- Winter (reference) ---")
    extract_features(winter_paths, CACHE_DIR_WINTER, model, preprocess, device)

    # Extract summer (query)
    print(f"\n--- Summer (query) ---")
    extract_features(summer_paths, CACHE_DIR_SUMMER, model, preprocess, device)

    print(f"\n{'='*60}")
    print("DONE. Now run validation:")
    print("  python -m experiments.nordland_500_vysotska_validation --descriptor dinov2_salad")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
