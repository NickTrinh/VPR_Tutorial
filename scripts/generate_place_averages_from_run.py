import os
import csv
import argparse
from glob import glob
from collections import defaultdict
from typing import Dict, List
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate results/<Dataset>/place_averages.csv from one or more test_results_run_*.csv files."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to results/<Dataset> directory, e.g., results/Nordland_Mini",
    )
    parser.add_argument(
        "--run-files",
        nargs="*",
        default=None,
        help="Specific run files to use (default: glob all test_results_run_*.csv in dataset-dir)",
    )
    parser.add_argument(
        "--method",
        choices=["original", "legacy_mean_bad"],
        default="legacy_mean_bad",
        help="Threshold method to use when aggregating per-image stats (default: legacy_mean_bad)",
    )
    return parser.parse_args()


def load_runs(run_files: List[str]) -> List[Dict[str, Dict[str, float]]]:
    all_runs: List[Dict[str, Dict[str, float]]] = []
    for rf in sorted(run_files):
        run_data: Dict[str, Dict[str, float]] = {}
        with open(rf, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_key = row["Image"].strip()
                try:
                    mean_bad = float(row["Mean Bad Scores"])  # per-image
                    std_bad = float(row["Std Deviation Bad Scores"])  # per-image
                    filter_n = float(row["Filter N"])  # per-image
                except Exception:
                    # Skip malformed rows
                    continue
                run_data[img_key] = {
                    "mean_bad_scores": mean_bad,
                    "std_dev_bad_scores": std_bad,
                    "filter_n": filter_n,
                }
        all_runs.append(run_data)
    return all_runs


def aggregate_by_image(
    all_runs: List[Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, List[float]]]:
    # {image: {mean_bad_scores: [...], std_dev_bad_scores: [...], filter_n: [...]}}
    image_data: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"mean_bad_scores": [], "std_dev_bad_scores": [], "filter_n": []}
    )

    for run in all_runs:
        for img_key, vals in run.items():
            image_data[img_key]["mean_bad_scores"].append(vals["mean_bad_scores"])
            image_data[img_key]["std_dev_bad_scores"].append(vals["std_dev_bad_scores"])
            image_data[img_key]["filter_n"].append(vals["filter_n"])

    return image_data


def compute_image_averages(
    image_data: Dict[str, Dict[str, List[float]]]
) -> Dict[str, Dict[str, float]]:
    image_averages: Dict[str, Dict[str, float]] = {}

    for img_key, data in image_data.items():
        mean_bads = np.array(data["mean_bad_scores"], dtype=float)
        std_devs = np.array(data["std_dev_bad_scores"], dtype=float)
        filter_ns = np.array(data["filter_n"], dtype=float)

        image_averages[img_key] = {
            "mean_bad_scores": float(np.mean(mean_bads)) if mean_bads.size else 0.0,
            "std_dev_bad_scores": float(np.mean(std_devs)) if std_devs.size else 0.0,
            "filter_n": float(np.mean(filter_ns)) if filter_ns.size else 0.0,
        }

    return image_averages


def write_image_averages(dataset_dir: str, image_averages: Dict[str, Dict[str, float]]):
    out_file = os.path.join(dataset_dir, "image_averages.csv")
    os.makedirs(dataset_dir, exist_ok=True)
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Image",
            "Mean Bad Scores",
            "Std Deviation Bad Scores",
            "Filter N",
        ])
        def key_fn(x: str):
            # Expect format pX/iY; fallback to raw string ordering
            try:
                p, i = x.split("/")
                return (int(p[1:]), int(i[1:]))
            except Exception:
                return (x, 0)
        for img_key in sorted(image_averages.keys(), key=key_fn):
            r = image_averages[img_key]
            writer.writerow([
                img_key,
                r["mean_bad_scores"],
                r["std_dev_bad_scores"],
                r["filter_n"],
            ])
    print(f"Wrote {out_file}")


def aggregate_place_from_images(
    image_averages: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, List[float]]]:
    # Convert aggregated per-image stats into place buckets
    place_data: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"mean_bad_scores": [], "std_dev_bad_scores": [], "filter_n": []}
    )
    for img_key, vals in image_averages.items():
        try:
            place = img_key.split("/")[0]
        except Exception:
            continue
        place_data[place]["mean_bad_scores"].append(vals["mean_bad_scores"])
        place_data[place]["std_dev_bad_scores"].append(vals["std_dev_bad_scores"])
        place_data[place]["filter_n"].append(vals["filter_n"])
    return place_data


def aggregate_by_place(
    all_runs: List[Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, List[float]]]:
    # {place: {mean_bad_scores: [...], std_dev_bad_scores: [...], filter_n: [...]}}
    place_data: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"mean_bad_scores": [], "std_dev_bad_scores": [], "filter_n": []}
    )

    for run in all_runs:
        for img_key, vals in run.items():
            # img_key format: pX/iY
            try:
                place = img_key.split("/")[0]
            except Exception:
                continue
            place_data[place]["mean_bad_scores"].append(vals["mean_bad_scores"])
            place_data[place]["std_dev_bad_scores"].append(vals["std_dev_bad_scores"])
            place_data[place]["filter_n"].append(vals["filter_n"])

    return place_data


def compute_place_averages(
    place_data: Dict[str, Dict[str, List[float]]], method: str
) -> Dict[str, Dict[str, float]]:
    place_averages: Dict[str, Dict[str, float]] = {}

    for place, data in place_data.items():
        mean_bads = np.array(data["mean_bad_scores"], dtype=float)
        std_devs = np.array(data["std_dev_bad_scores"], dtype=float)
        filter_ns = np.array(data["filter_n"], dtype=float)

        if method == "legacy_mean_bad":
            thresholds = mean_bads
        else:
            thresholds = mean_bads + filter_ns * std_devs

        simple_avg_threshold = float(np.mean(thresholds)) if thresholds.size else 0.0
        # Avoid div by 0: add small epsilon
        weights = 1.0 / (np.power(std_devs, 2) + 1e-9)
        if np.sum(weights) > 0:
            weighted_avg_threshold = float(np.sum(weights * thresholds) / np.sum(weights))
        else:
            weighted_avg_threshold = simple_avg_threshold

        place_averages[place] = {
            "simple_avg_threshold": simple_avg_threshold,
            "weighted_avg_threshold": weighted_avg_threshold,
            "std_dev_of_thresholds": float(np.std(thresholds)) if thresholds.size else 0.0,
            "avg_filter_n": float(np.mean(filter_ns)) if filter_ns.size else 0.0,
        }

    return place_averages


def write_place_averages(dataset_dir: str, place_averages: Dict[str, Dict[str, float]]):
    out_file = os.path.join(dataset_dir, "place_averages.csv")
    os.makedirs(dataset_dir, exist_ok=True)
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "place",
                "simple_avg_threshold",
                "weighted_avg_threshold",
                "std_dev_of_thresholds",
                "avg_filter_n",
            ]
        )
        for place in sorted(place_averages.keys(), key=lambda x: int(x[1:]) if x.startswith("p") else x):
            r = place_averages[place]
            writer.writerow(
                [
                    place,
                    r["simple_avg_threshold"],
                    r["weighted_avg_threshold"],
                    r["std_dev_of_thresholds"],
                    r["avg_filter_n"],
                ]
            )
    print(f"Wrote {out_file}")


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir

    if args.run_files:
        run_files = args.run_files
    else:
        run_files = glob(os.path.join(dataset_dir, "test_results_run_*.csv"))

    if not run_files:
        raise FileNotFoundError(
            f"No run files found. Provide --run-files or place test_results_run_*.csv in {dataset_dir}"
        )

    all_runs = load_runs(run_files)
    # Phase 1: image averages across runs
    image_data = aggregate_by_image(all_runs)
    image_averages = compute_image_averages(image_data)
    write_image_averages(dataset_dir, image_averages)

    # Phase 2: place averages derived from the aggregated image averages
    place_data = aggregate_place_from_images(image_averages)
    place_averages = compute_place_averages(place_data, method=args.method)
    write_place_averages(dataset_dir, place_averages)


if __name__ == "__main__":
    main()


