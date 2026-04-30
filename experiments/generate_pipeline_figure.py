"""
Regenerate IEEE_RAL_VPR/fig8_method_pipeline.png with larger text.

Two stacked horizontal pipelines comparing our per-place threshold method
against Vysotska et al.'s per-query similarity-matrix method.

Usage:
    python -m experiments.generate_pipeline_figure
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "IEEE_RAL_VPR",
    "fig8_method_pipeline.png",
)

GRAY   = "#D9D9D9"
BLUE   = "#9DC3E6"
PURPLE = "#B4A7D6"
PINK   = "#F4B6C2"
TEAL   = "#9FE2BF"
YELLOW = "#FFE699"
RED    = "#F4A6A6"

TITLE_FS  = 22
ROWLBL_FS = 14
BOX_FS    = 14
SUB_FS    = 12
NOTE_FS   = 13


def box(ax, x, y, w, h, text, color, fontsize=BOX_FS):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor=color, edgecolor="#444", linewidth=1.4,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            wrap=True, family="DejaVu Sans")


def arrow(ax, x1, y1, x2, y2):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=22,
        linewidth=2.0, color="#444",
    )
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(18, 10), dpi=200)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 18)
    ax.axis("off")

    # ============================================================
    # TOP: OUR METHOD
    # ============================================================
    ax.text(15, 17.2,
            "Our Method: Per-Place Thresholds from Reference Data",
            ha="center", va="center", fontsize=TITLE_FS,
            color="#C00000", fontweight="bold")

    # Row labels
    ax.text(0.6, 14.5, "Offline\n(reference)",
            ha="center", va="center", fontsize=ROWLBL_FS,
            color="#444", style="italic")
    ax.text(0.6, 11.0, "Online\n(query)",
            ha="center", va="center", fontsize=ROWLBL_FS,
            color="#444", style="italic")

    # OFFLINE row (top)
    bw, bh, by = 4.6, 2.0, 13.5
    xs = [1.8, 7.2, 12.6, 18.0]
    box(ax, xs[0], by, bw, bh, "Reference\nImage Stream", GRAY)
    box(ax, xs[1], by, bw, bh, "Online Place\nDiscovery", BLUE)
    box(ax, xs[2], by, bw, bh,
        "Compute Per-Place\nStats $(\\mu_{\\mathrm{dist}}, \\sigma_{\\mathrm{dist}})$", PURPLE)
    box(ax, xs[3], by, bw, bh,
        "Per-Place\nThresholds\n$\\theta_p = \\mu_{\\mathrm{dist}} + k\\,\\sigma_{\\mathrm{dist}}$",
        PINK, fontsize=BOX_FS - 1)
    for i in range(3):
        arrow(ax, xs[i] + bw, by + bh / 2, xs[i + 1], by + bh / 2)

    # ONLINE row (middle)
    by2 = 10.0
    xs2 = [1.8, 7.2, 12.6, 18.0, 23.4]
    box(ax, xs2[0], by2, bw, bh, "New Query\nImage", GRAY)
    box(ax, xs2[1], by2, bw, bh, "Compute Similarity\nto All Ref Images", BLUE)
    box(ax, xs2[2], by2, bw, bh, "Average\nScore per Place", PURPLE)
    box(ax, xs2[3], by2, bw, bh, "Compare\nScore vs $\\theta_p$", PINK)
    box(ax, xs2[4], by2, bw, bh, "Accept /\nReject", TEAL)
    for i in range(4):
        arrow(ax, xs2[i] + bw, by2 + bh / 2, xs2[i + 1], by2 + bh / 2)

    # vertical link from offline thresholds down to online compare
    arrow(ax, xs[3] + bw / 2, by, xs2[3] + bw / 2, by2 + bh)

    # ============================================================
    # BOTTOM: VYSOTSKA
    # ============================================================
    ax.text(15, 7.0,
            "Vysotska et al.: Per-Query Thresholds from Similarity Matrix",
            ha="center", va="center", fontsize=TITLE_FS,
            color="#C77A0A", fontweight="bold")

    by3 = 3.5
    bw3 = 4.0
    xs3 = [0.7, 5.4, 10.1, 14.8, 19.5, 24.2]
    box(ax, xs3[0], by3, bw3, bh, "Full Query +\nReference Set", GRAY)
    box(ax, xs3[1], by3, bw3, bh, "Compute Full\nSimilarity Matrix $S$", BLUE)
    box(ax, xs3[2], by3, bw3, bh, "Extract $20{\\times}20$\nPatch Around Match", YELLOW)
    box(ax, xs3[3], by3, bw3, bh, "KS Test\n(bimodal?)", PINK)
    box(ax, xs3[4], by3, bw3, bh, "GMM Fit +\nDecision Boundary", RED)
    box(ax, xs3[5], by3, bw3, bh, "Kalman\nSmooth $\\theta_q$", TEAL)
    for i in range(5):
        arrow(ax, xs3[i] + bw3, by3 + bh / 2, xs3[i + 1], by3 + bh / 2)

    ax.text(15, 1.5,
            "Requires all queries upfront to build $S$; "
            "threshold depends on each query's similarity pattern",
            ha="center", va="center", fontsize=NOTE_FS,
            color="#666", style="italic")

    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
