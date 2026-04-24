"""
Extract DINOv2 SALAD descriptors for ESSEX3IN1 dataset.

Must be run on a GPU node (ciscluster).

ESSEX3IN1: 210 reference + 210 query images, 1-to-1 matching.
Images named 0.jpg .. 209.jpg in each folder.
Images 0-132 are "confusing", 133-209 are "good".

Usage (on GPU node):
    conda activate vprtutorial
    python -m experiments.extract_essex3in1
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

IMAGE_DIR_REF = "images/ESSEX3IN1/reference_combined"
IMAGE_DIR_QUERY = "images/ESSEX3IN1/query_combined"
CACHE_DIR_REF = "cache/ESSEX3IN1/reference/dinov2-salad"
CACHE_DIR_QUERY = "cache/ESSEX3IN1/query/dinov2-salad"
BATCH_SIZE = 32


def get_image_paths(image_dir):
    """Get image paths sorted by numeric index (0.jpg, 1.jpg, ..., 209.jpg)."""
    all_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    all_files.sort(key=lambda f: int(os.path.splitext(f)[0]))
    return [os.path.join(image_dir, f) for f in all_files]


def extract_features(image_paths, cache_dir, model, preprocess, device):
    """Extract and cache DINOv2 SALAD features."""
    os.makedirs(cache_dir, exist_ok=True)

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
            features = model(batch_tensor)

        features_np = features.cpu().numpy().astype(np.float32)

        for j, (idx, _) in enumerate(batch):
            cache_path = os.path.join(cache_dir, f"img_{idx}_descriptor.pkl")
            with open(cache_path, "wb") as f:
                pickle.dump({"descriptor": features_np[j]}, f)

    print(f"  Done. Cached to {cache_dir}")


def main():
    print("=" * 60)
    print("DINOv2 SALAD Feature Extraction for ESSEX3IN1")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run this on a GPU node.")
        print("  ssh ciscluster -> ssh node002")
        print("  conda activate vprtutorial")
        print("  python -m experiments.extract_essex3in1")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {device} ({torch.cuda.get_device_name()})")

    print("\nLoading DINOv2 SALAD model...")
    t0 = time.time()
    model = torch.hub.load("serizba/salad", "dinov2_salad")
    model = model.to(device)
    model.eval()
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    preprocess = transforms.Compose([
        transforms.Resize((322, 322),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ref_paths = get_image_paths(IMAGE_DIR_REF)
    query_paths = get_image_paths(IMAGE_DIR_QUERY)
    print(f"\nReference images: {len(ref_paths)}")
    print(f"Query images: {len(query_paths)}")

    print(f"\n--- Reference ---")
    extract_features(ref_paths, CACHE_DIR_REF, model, preprocess, device)

    print(f"\n--- Query ---")
    extract_features(query_paths, CACHE_DIR_QUERY, model, preprocess, device)

    print(f"\n{'='*60}")
    print("DONE.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
