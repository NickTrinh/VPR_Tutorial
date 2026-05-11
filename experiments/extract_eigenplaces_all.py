"""
Extract EigenPlaces descriptors for all six datasets, freshly.

Same image sources and target cache paths as `extract_dinov2_salad_all.py`,
except outputs go to `<cache>/eigenplaces/` and image counts are aligned
with the SALAD caches so the comparison is apples-to-apples.

Existing eigenplaces caches under any of the target paths are deleted
before extraction so stale or partial caches cannot mix with fresh ones.

Usage:
    python -u -m experiments.extract_eigenplaces_all
"""

import os
import sys
import pickle
import shutil
import time

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

BATCH_SIZE = 16

DATASETS = {
    "nordland_winter": {
        "image_dir": "images/Nordland-500/winter",
        "cache_dir": "cache/Nordland-500/winter/eigenplaces",
        "n": 500,
    },
    "nordland_summer": {
        "image_dir": "images/Nordland-500/summer",
        "cache_dir": "cache/Nordland-500/summer/eigenplaces",
        "n": 500,
    },
    "gardenspoint_day_left": {
        "image_dir": "images/GardensPoint/day_left",
        "cache_dir": "cache/GardensPoint/day_left/eigenplaces",
        "n": 200,
    },
    "gardenspoint_day_right": {
        "image_dir": "images/GardensPoint/day_right",
        "cache_dir": "cache/GardensPoint/day_right/eigenplaces",
        "n": 200,
    },
    "sfu_dry": {
        "image_dir": "images/SFU/dry",
        "cache_dir": "cache/SFU/dry/eigenplaces",
        "n": 385,
    },
    "sfu_jan": {
        "image_dir": "images/SFU/jan",
        "cache_dir": "cache/SFU/jan/eigenplaces",
        "n": 385,
    },
    "bonn_reference": {
        "image_dir": "images/bonn_example/reference/images",
        "cache_dir": "cache/Bonn/reference/eigenplaces",
        "n": None,
    },
    "bonn_query": {
        "image_dir": "images/bonn_example/query/images",
        "cache_dir": "cache/Bonn/query/eigenplaces",
        "n": None,
    },
    "freiburg_reference": {
        "image_dir": "images/freiburg_example/reference/images",
        "cache_dir": "cache/Freiburg/reference/eigenplaces",
        "n": None,
    },
    "freiburg_query": {
        "image_dir": "images/freiburg_example/query/images",
        "cache_dir": "cache/Freiburg/query/eigenplaces",
        "n": None,
    },
    "essex3in1_reference": {
        "image_dir": "images/ESSEX3IN1/reference_combined",
        "cache_dir": "cache/ESSEX3IN1/reference/eigenplaces",
        "n": 210,
    },
    "essex3in1_query": {
        "image_dir": "images/ESSEX3IN1/query_combined",
        "cache_dir": "cache/ESSEX3IN1/query/eigenplaces",
        "n": 210,
    },
}


def get_image_paths(image_dir, n):
    all_files = sorted(os.listdir(image_dir))
    all_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if n is not None:
        all_files = all_files[:n]
    return [os.path.join(image_dir, f) for f in all_files]


def extract_features(image_paths, cache_dir, model, preprocess, device):
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"  Extracting {len(image_paths)} descriptors...")
    model.eval()
    t0 = time.time()

    for batch_start in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        batch = image_paths[batch_start:batch_start + BATCH_SIZE]
        imgs = []
        for path in batch:
            img = Image.open(path).convert("RGB")
            imgs.append(preprocess(img))
        batch_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    features = model(batch_tensor)
            else:
                features = model(batch_tensor)

        features_np = features.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(features_np, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        features_np = features_np / norms

        for j, _ in enumerate(batch):
            idx = batch_start + j
            cache_path = os.path.join(cache_dir, f"img_{idx}_descriptor.pkl")
            with open(cache_path, "wb") as f:
                pickle.dump({"descriptor": features_np[j]}, f)

    dt = time.time() - t0
    print(f"  Done. {len(image_paths)} desc in {dt:.1f}s "
          f"({dt/max(len(image_paths),1):.2f}s/img)  →  {cache_dir}")


def main():
    print("=" * 60)
    print("EigenPlaces Feature Extraction — All Datasets (fresh)")
    print("=" * 60)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {device} ({torch.cuda.get_device_name()})")
    else:
        device = torch.device("cpu")
        print("Device: cpu (extraction will be slower than on GPU)")

    print("\nLoading EigenPlaces model (gmberton/eigenplaces, ResNet50, 2048d)...")
    t0 = time.time()
    model = torch.hub.load(
        "gmberton/eigenplaces", "get_trained_model",
        backbone="ResNet50", fc_output_dim=2048,
    )
    model = model.to(device)
    model.eval()
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    preprocess = transforms.Compose([
        transforms.Resize((480, 480),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    only = set(os.environ.get("ONLY", "").split(",")) - {""}
    for name, cfg in DATASETS.items():
        if only and name not in only:
            continue
        print(f"\n--- {name} ---")
        if not os.path.isdir(cfg["image_dir"]):
            print(f"  SKIP: {cfg['image_dir']} not found")
            continue
        paths = get_image_paths(cfg["image_dir"], cfg["n"])
        print(f"  Found {len(paths)} images in {cfg['image_dir']}")
        extract_features(paths, cfg["cache_dir"], model, preprocess, device)

    print(f"\n{'='*60}")
    print("DONE.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
