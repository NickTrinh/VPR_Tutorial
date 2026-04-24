"""Extract EigenPlaces and DINOv2 SALAD descriptors for Bonn and Freiburg datasets."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.vysotska_bonn_freiburg_validation import extract_or_load_descriptors, load_images_sorted

datasets = [
    ("Freiburg", "images/freiburg_example", "*.jpg"),
    ("Bonn", "images/bonn_example", "*.png"),
]

descriptors = ["eigenplaces", "dinov2-salad"]

for name, base_path, ext in datasets:
    for desc in descriptors:
        print(f"\n{'='*50}")
        print(f"{name} / {desc}")
        print(f"{'='*50}")
        for split in ["reference", "query"]:
            paths = load_images_sorted(f"{base_path}/{split}/images/", ext)
            print(f"  {split}: {len(paths)} images")
            extract_or_load_descriptors(paths, f"cache/{name}/{split}/{desc}", desc)
        print(f"Done: {name} / {desc}")

print("\nAll extractions complete.")
