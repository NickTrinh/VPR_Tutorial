"""
Extract DINOv2 SALAD descriptors for all datasets.

Must be run on a GPU node.

Usage:
    python -m experiments.extract_dinov2_salad_all
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

BATCH_SIZE = 32

DATASETS = {
    "nordland_winter": {
        "image_dir": "images/Nordland_Mini/winter",
        "cache_dir": "cache/Nordland_salad/winter/dinov2_salad",
        "n": 500,
    },
    "nordland_summer": {
        "image_dir": "images/Nordland_Mini/summer",
        "cache_dir": "cache/Nordland_salad/summer/dinov2_salad",
        "n": 500,
    },
    "gardenspoint_day_left": {
        "image_dir": "images/GardensPoint/day_left",
        "cache_dir": "cache/GardensPoint/day_left/dinov2-salad",
        "n": 200,
    },
    "gardenspoint_day_right": {
        "image_dir": "images/GardensPoint/day_right",
        "cache_dir": "cache/GardensPoint/day_right/dinov2-salad",
        "n": 200,
    },
    "sfu_dry": {
        "image_dir": "images/SFU/dry",
        "cache_dir": "cache/SFU/dry/dinov2-salad",
        "n": 385,
    },
    "sfu_jan": {
        "image_dir": "images/SFU/jan",
        "cache_dir": "cache/SFU/jan/dinov2-salad",
        "n": 385,
    },
}


def get_image_paths(image_dir, n):
    all_files = sorted(os.listdir(image_dir))
    all_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    paths = [os.path.join(image_dir, f) for f in all_files[:n]]
    return paths


def extract_features(image_paths, cache_dir, model, preprocess, device):
    os.makedirs(cache_dir, exist_ok=True)

    to_extract = []
    for i, path in enumerate(image_paths):
        cache_path = os.path.join(cache_dir, f"img_{i}_descriptor.pkl")
        if not os.path.exists(cache_path):
            to_extract.append((i, path))

    if not to_extract:
        print(f"  All {len(image_paths)} already cached in {cache_dir}")
        return

    print(f"  Extracting {len(to_extract)}/{len(image_paths)} descriptors...")
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
    print("DINOv2 SALAD Feature Extraction — All Datasets")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run on GPU node.")
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

    for name, cfg in DATASETS.items():
        print(f"\n--- {name} ---")
        if not os.path.isdir(cfg["image_dir"]):
            print(f"  SKIP: {cfg['image_dir']} not found")
            continue
        paths = get_image_paths(cfg["image_dir"], cfg["n"])
        print(f"  Found {len(paths)} images")
        extract_features(paths, cfg["cache_dir"], model, preprocess, device)

    print(f"\n{'='*60}")
    print("DONE.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
