"""
Visualize Recall@K results as grouped bar charts.

Reads recall_at_k.csv and produces:
  1. recall_at_1_comparison.png — Recall@1 across all datasets/descriptors
  2. recall_all_k.png — Panel of Recall@1,3,5,10 per dataset
"""

import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Display names
DATASET_LABELS = {
    "gardenspoint_mini": "GardensPoint\n(20 places)",
    "sfu_mini": "SFU\n(192 places)",
    "nordland_mini_g3s3": "Nordland g3s3\n(9,197 places)",
    "nordland_mini_g2s2": "Nordland g2s2\n(13,796 places)",
}

DESCRIPTOR_LABELS = {
    "eigenplaces": "EigenPlaces",
    "cosplace": "CosPlace",
    "alexnet": "AlexNet",
}

METHOD_LABELS = {
    "baseline": "Baseline",
    "simple_avg_thresholds": "Simple Avg",
    "weighted_avg_thresholds": "Weighted Avg",
}

METHOD_COLORS = {
    "baseline": "#7f8c8d",
    "simple_avg_thresholds": "#2980b9",
    "weighted_avg_thresholds": "#e67e22",
}

# Canonical order
DATASET_ORDER = ["gardenspoint_mini", "sfu_mini", "nordland_mini_g3s3", "nordland_mini_g2s2"]
DESCRIPTOR_ORDER = ["eigenplaces", "cosplace", "alexnet"]
METHOD_ORDER = ["baseline", "simple_avg_thresholds", "weighted_avg_thresholds"]
K_VALUES = ["Recall@1", "Recall@3", "Recall@5", "Recall@10"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Recall@K results as grouped bar charts"
    )
    parser.add_argument("--csv_path", type=str,
                        default="results/comparison/recall_at_k.csv",
                        help="Path to recall_at_k.csv")
    parser.add_argument("--output_dir", type=str,
                        default="results/visualizations",
                        help="Output directory for PNGs")
    return parser.parse_args()


def load_recall_data(csv_path):
    """Load and deduplicate recall data. Keeps latest entry per (dataset, descriptor, method).

    Filters out obviously bad entries (Recall@1 < 20 for simple/weighted methods
    where baseline is much higher).
    """
    raw = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw.append(row)

    # Deduplicate: keep latest timestamp per (dataset, descriptor, method)
    seen = {}
    for row in raw:
        key = (row["Dataset"], row["Descriptor"], row["Method"])
        ts = row["Timestamp"]
        if key not in seen or ts > seen[key]["Timestamp"]:
            seen[key] = row

    def parse_float(val):
        """Extract leading float from a string, tolerating CSV corruption."""
        m = re.match(r"[-+]?\d*\.?\d+", str(val).strip())
        return float(m.group()) if m else None

    # Build structured dict: data[(dataset, descriptor, method)] = {k: value}
    data = {}
    for key, row in seen.items():
        parsed = {k: parse_float(row[k]) for k in K_VALUES}
        if any(v is None for v in parsed.values()):
            continue
        data[key] = parsed

    return data


def plot_recall_at_1(data, output_dir):
    """Main figure: Recall@1 grouped bar chart across all dataset/descriptor combos."""
    # Build groups: each group is a (dataset, descriptor) pair
    groups = []
    for ds in DATASET_ORDER:
        for desc in DESCRIPTOR_ORDER:
            if any((ds, desc, m) in data for m in METHOD_ORDER):
                groups.append((ds, desc))

    n_groups = len(groups)
    n_methods = len(METHOD_ORDER)
    bar_width = 0.25
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(14, n_groups * 1.2), 6))

    for i, method in enumerate(METHOD_ORDER):
        values = []
        for ds, desc in groups:
            key = (ds, desc, method)
            if key in data:
                values.append(data[key]["Recall@1"])
            else:
                values.append(0)

        bars = ax.bar(x + i * bar_width, values, bar_width,
                      label=METHOD_LABELS[method],
                      color=METHOD_COLORS[method],
                      edgecolor="white", linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7,
                        fontweight="bold")

    # X-axis labels
    group_labels = []
    for ds, desc in groups:
        group_labels.append(f"{DESCRIPTOR_LABELS.get(desc, desc)}")

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(group_labels, fontsize=9)

    # Add dataset separators and labels
    ds_boundaries = []
    prev_ds = None
    for i, (ds, desc) in enumerate(groups):
        if ds != prev_ds:
            ds_boundaries.append(i)
            prev_ds = ds

    # Draw separators
    for boundary in ds_boundaries[1:]:
        ax.axvline(x=boundary - 0.35, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    # Add dataset labels at top
    for j, boundary in enumerate(ds_boundaries):
        next_boundary = ds_boundaries[j + 1] if j + 1 < len(ds_boundaries) else n_groups
        center = (boundary + next_boundary - 1) / 2
        ds_name = groups[boundary][0]
        label = DATASET_LABELS.get(ds_name, ds_name)
        ax.text(center + bar_width, 1.02, label.replace("\n", " "),
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Recall@1 (%)", fontsize=12)
    ax.set_title("Recall@1 Comparison: Baseline vs Adaptive Thresholding", fontsize=13,
                 fontweight="bold", pad=30)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, "recall_at_1_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_recall_all_k(data, output_dir):
    """Panel figure: one subplot per dataset, bars for each (descriptor, method) at each K."""
    datasets_present = [ds for ds in DATASET_ORDER
                        if any((ds, desc, m) in data
                               for desc in DESCRIPTOR_ORDER
                               for m in METHOD_ORDER)]

    n_datasets = len(datasets_present)
    fig, axes = plt.subplots(1, n_datasets, figsize=(5.5 * n_datasets, 5), squeeze=False)

    for col, ds in enumerate(datasets_present):
        ax = axes[0, col]

        # For this dataset, build bars per K value
        # Group: each K value on x-axis
        # Within each K: one bar per (descriptor, method) — but that's 9 bars, too many
        # Better: one bar per method, averaged or shown per descriptor
        # Cleanest: for each descriptor, show baseline vs best threshold at each K

        descriptors = [d for d in DESCRIPTOR_ORDER
                       if any((ds, d, m) in data for m in METHOD_ORDER)]

        n_k = len(K_VALUES)
        n_desc = len(descriptors)
        group_width = 0.8
        bar_width = group_width / (n_desc * 2)  # 2 bars per descriptor (baseline + threshold)

        x = np.arange(n_k)

        for d_idx, desc in enumerate(descriptors):
            # Baseline
            baseline_key = (ds, desc, "baseline")
            baseline_vals = [data[baseline_key][k] if baseline_key in data else 0
                             for k in K_VALUES]

            # Best threshold (simple avg — same as weighted in most cases)
            thresh_key = (ds, desc, "simple_avg_thresholds")
            thresh_vals = [data[thresh_key][k] if thresh_key in data else 0
                           for k in K_VALUES]

            offset = (d_idx - n_desc / 2 + 0.5) * bar_width * 2
            color_base = plt.cm.Set2(d_idx)
            color_thresh = plt.cm.Dark2(d_idx)

            ax.bar(x + offset - bar_width / 2, baseline_vals, bar_width,
                   color=color_base, alpha=0.6, edgecolor="white",
                   label=f"{DESCRIPTOR_LABELS[desc]} baseline" if col == 0 else "")
            ax.bar(x + offset + bar_width / 2, thresh_vals, bar_width,
                   color=color_thresh, alpha=0.9, edgecolor="white",
                   label=f"{DESCRIPTOR_LABELS[desc]} threshold" if col == 0 else "")

        ax.set_xticks(x)
        ax.set_xticklabels(["@1", "@3", "@5", "@10"], fontsize=10)
        ax.set_xlabel("Recall@K")
        ax.set_ylim(0, 105)
        ax.set_title(DATASET_LABELS.get(ds, ds).replace("\n", " "), fontsize=11,
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.2)

        if col == 0:
            ax.set_ylabel("Recall (%)", fontsize=11)

    # Build legend from first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(6, len(handles)), fontsize=9,
                   bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Recall@K: Baseline vs Adaptive Thresholding by Dataset",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "recall_all_k.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_improvement_delta(data, output_dir):
    """Show Recall@1 improvement (delta) over baseline for each dataset/descriptor."""
    groups = []
    for ds in DATASET_ORDER:
        for desc in DESCRIPTOR_ORDER:
            baseline_key = (ds, desc, "baseline")
            thresh_key = (ds, desc, "simple_avg_thresholds")
            if baseline_key in data and thresh_key in data:
                groups.append((ds, desc))

    n_groups = len(groups)
    deltas = []
    for ds, desc in groups:
        baseline = data[(ds, desc, "baseline")]["Recall@1"]
        threshold = data[(ds, desc, "simple_avg_thresholds")]["Recall@1"]
        deltas.append(threshold - baseline)

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.0), 5))

    colors = ["#27ae60" if d > 0 else "#e74c3c" for d in deltas]
    x = np.arange(n_groups)
    bars = ax.bar(x, deltas, 0.6, color=colors, edgecolor="white")

    # Value labels
    for bar, val in zip(bars, deltas):
        y_pos = bar.get_height() + (0.3 if val >= 0 else -1.2)
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"+{val:.1f}" if val > 0 else f"{val:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Labels
    labels = [f"{DESCRIPTOR_LABELS.get(desc, desc)}" for ds, desc in groups]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)

    # Dataset separators
    prev_ds = None
    ds_boundaries = []
    for i, (ds, desc) in enumerate(groups):
        if ds != prev_ds:
            ds_boundaries.append(i)
            prev_ds = ds

    for boundary in ds_boundaries[1:]:
        ax.axvline(x=boundary - 0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    for j, boundary in enumerate(ds_boundaries):
        next_boundary = ds_boundaries[j + 1] if j + 1 < len(ds_boundaries) else n_groups
        center = (boundary + next_boundary - 1) / 2
        ds_name = groups[boundary][0]
        label = DATASET_LABELS.get(ds_name, ds_name).replace("\n", " ")
        ax.text(center, 1.02, label, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Recall@1 Improvement (pp)", fontsize=11)
    ax.set_title("Recall@1 Gain from Adaptive Thresholding over Baseline",
                 fontsize=13, fontweight="bold", pad=25)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, "recall_improvement_delta.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading recall data from: {args.csv_path}")
    data = load_recall_data(args.csv_path)
    print(f"  Loaded {len(data)} unique (dataset, descriptor, method) entries")

    output_files = []
    output_files.append(plot_recall_at_1(data, args.output_dir))
    output_files.append(plot_recall_all_k(data, args.output_dir))
    output_files.append(plot_improvement_delta(data, args.output_dir))

    print(f"\nDone. {len(output_files)} figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
