# Adaptive Per-Place Thresholding for Open-Set Visual Place Recognition

Code accompanying the IEEE RAL submission from FRCV Lab, Fordham University
(Nick Trinh, Damian Lyons). Paper source: [`Adaptive Thresholding for Visual Place Recognition using Negative Gaussian Mixture Statistics`](https://arxiv.org/abs/2512.09071).

---

## What this paper does

A robot driving a route shouldn't claim to recognise *every* image. Most
visual place recognition (VPR) systems return the argmax of a similarity
matrix — they always answer, even when the right answer is "I haven't seen
this place before". We give each *place* its own rejection threshold

```
θ_k = μ_k⁻ + k · σ_k⁻,  k = clip(sep_k / 2, 1, 2)
```

computed entirely from the reference-to-reference similarity statistics. No
ground-truth labels at threshold time. No query-distribution tuning. Queries
whose score to every discovered place falls below that place's threshold are
rejected as unknown (open-set).

The architectural punchline: Vysotska et al. (ICRA 2025), the closest prior
work, fits a per-query threshold to a patch of the full similarity matrix.
They need every query upfront and pay one threshold per query. Our
thresholds are *properties of places*, computed once, fixed at evaluation.

![Method pipeline](IEEE_RAL_VPR/fig8_method_pipeline.png)

---

## Headline results (DINOv2 SALAD, 6 datasets, identical protocol)

**Closed-set F1 (%) — every query has a match:**

| Dataset      | Baseline | **Ours** | Vysotska |
|--------------|---------:|---------:|---------:|
| Nordland-500 |     98.9 |     90.5 | **99.4** |
| Bonn         |     90.4 |     89.2 | **90.4** |
| Freiburg     |     91.7 |     85.8 | **91.8** |
| GardensPoint |     98.5 |     96.6 | **98.7** |
| SFU          |     97.1 |     88.7 | **99.3** |
| ESSEX3IN1    |     98.5 |     91.2 | **99.5** |

**Natural open-set (truncate reference to 70%, treat the cut-off 30% as
same-environment distractors). F1 + distractor-rejection rate (%):**

| Dataset      | Ours F1 | **Ours Rej.** | Vys. F1 | Vys. Rej. |
|--------------|--------:|--------------:|--------:|----------:|
| Nordland-500 |    87.1 |   **100.0**   |    95.1 |      77.3 |
| Bonn         |    92.1 |   **92.2**    |    93.1 |      89.3 |
| Freiburg     |    88.0 |   **99.6**    |    94.4 |      93.7 |
| GardensPoint |    95.9 |   **100.0**   |    98.2 |      96.7 |
| SFU          |    71.8 |   **79.3**    |    91.8 |      60.3 |
| ESSEX3IN1    |    80.1 |     60.3      |    92.4 |  **65.1** |

On four of six datasets (Nordland, Bonn, Freiburg, GardensPoint) we reject
**92.2 – 100 %** of same-environment distractors versus Vysotska's 77.3 –
96.7 %, while staying within 2 – 8 F1 points of their closed-set numbers. On
SFU (forest trail) and ESSEX3IN1 (near-identical corridors), the local-window
prior baked into Vysotska's sequence matcher carries the signal. These are
sequence-dominated environments where per-image thresholds can't
recover the placement. The two methods are complementary, not strictly
ordered.

Reimplementation validation against Vysotska et al.'s published numbers
(`tab:vys_validation` in the paper): within 0.04 F1 on all three reported
datasets (max gap 0.04 on Freiburg).

---

## Reproduce in three commands

```bash
git clone https://github.com/NickTrinh/VPR_Tutorial.git
cd VPR_Tutorial
conda create -n vprtutorial python=3.11 && conda activate vprtutorial

bash setup.sh                                    # 1. install deps + auto-download datasets
python -m experiments.extract_dinov2_salad_all   # 2. descriptor extraction (GPU recommended)
python -m experiments.final_all_datasets         # 3. closed + natural open-set, prints Tables 2 & 3
```

Step 3 writes `results/final_all_datasets_dinov2salad.json` and prints the
summary tables that populate the paper's Tables II and III. A few minutes on
CPU once descriptors are cached. `torch` and `tensorflow` versions in
`requirements.txt` are pinned to what was used for the paper.

Step 2 supports CPU fallback (auto-detected), but is **much** slower
without a GPU. Extraction is ~10 s/image on CPU vs sub-second on a V100.
Step 3 is CPU-only and reads the cached descriptors.

---

## Datasets

`setup.sh` auto-fetches everything except Nordland-500 (HuggingFace LFS,
one manual step). The script prints the URL and target path at the end.

| Dataset        | Size   | Auto-fetched? | Source |
|----------------|--------|---------------|--------|
| GardensPoint   | 32 MB  | yes (TU-Chemnitz mirror) | [QUT Wiki](https://wiki.qut.edu.au/display/cyphy/Day+and+Night+with+Lateral+Pose+Change+Datasets) |
| SFU Mountain   | 72 MB  | yes (TU-Chemnitz mirror) | <http://autonomy.cs.sfu.ca/sfu-mountain-dataset/> |
| Bonn           | 1.2 GB | yes | <http://www.ipb.uni-bonn.de/html/projects/visual_place_recognition/bonn_example.zip> |
| Freiburg       | 738 MB | yes | <http://www.ipb.uni-bonn.de/html/projects/visual_place_recognition/freiburg_example.zip> |
| ESSEX3IN1      | 1.5 GB | yes (git clone) | <https://github.com/MubarizZaffar/ESSEX3IN1-Dataset> |
| Nordland-500   | 500-img subset of ~36 GB | **manual** | <https://huggingface.co/datasets/Somayeh-h/Nordland> |

For Nordland, place the first 500 frames of the 1 fps winter and summer
traversals at `images/Nordland_Mini/{winter,summer}/*.jpg` to match
Vysotska et al.'s protocol.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `experiments/final_all_datasets.py` | Canonical paper reproducer — loads 6 datasets, runs discovery → threshold → filter-then-rank, compares to Vysotska sequence matcher. |
| `experiments/online_place_discovery.py` | `OnlinePlaceDiscovery` class — bootstrap (α=1.5) → online (m=2, h=2), Welford's algorithm for per-place statistics. |
| `experiments/vysotska_threshold.py` | Vysotska et al. adaptive threshold: KS test + 2-GMM + 1D Kalman filter on similarity-matrix patches. |
| `experiments/vysotska_sequence_matcher.py` | Vysotska et al. graph-based sequence matcher (shortest-path on DAG built from similarity matrix). |
| `experiments/extract_dinov2_salad_all.py` | Descriptor extraction for all 6 datasets via `torch.hub.load("serizba/salad", "dinov2_salad")`; CPU fallback. |
| `experiments/generate_pipeline_figure.py` | Regenerates `IEEE_RAL_VPR/fig8_method_pipeline.png`. |
| `IEEE_RAL_VPR/` | Paper source (`RCC2025.tex`), bibliography, figures, generated PDF. |
| `sequential_runner.py`, `demo.py`, `experiment_runner.py`, `test_runner.py`, `multi_dataset_runner.py` | RCC 2025 pipeline (EigenPlaces / CosPlace / PatchNetVLAD). Not used by the RAL paper. |

All numbers in the paper come out of `experiments/final_all_datasets.py`;
the JSON it writes is the source of truth for the result tables.

---

## What we found (and didn't)

- **Open-set rejection is the headline.** Per-place thresholds reject
  same-environment distractors significantly better than Vysotska's
  per-query thresholding on visually distinctive datasets (Nordland, Bonn,
  Freiburg, GardensPoint). The 70%-reference protocol is harder than the
  more common cross-environment-distractor protocol because rejection has
  to come from the descriptor's intrinsic separability, not "this image
  looks nothing like a campus".

- **Closed-set is a tradeoff, not a free lunch.** Our F1 is 2–10 points
  below Vysotska on closed-set across the board. The gap is real and we
  don't paper over it: the sequence matcher's local-window prior is a
  powerful tool when every query is genuine.

- **Sequence-dominated environments are not our wheelhouse.** SFU (forest)
  and ESSEX3IN1 (corridors) have visually near-degenerate images; sequential
  ordering carries the signal there, and our per-image thresholds can't
  reach it.

- **No per-dataset tuning.** Same descriptor, same tolerance (±2 frames),
  same reference fraction (70%) for open-set, same multi-match fanout (5),
  and the same online-discovery parameters across all six datasets. The
  adaptive multiplier `k = clip(sep/2, 1, 2)` is computed per place from
  reference-only statistics; no manual knob.

---

## Citation

```bibtex
@article{Trinh2026RAL,
  author  = {Trinh, Nick and Lyons, Damian},
  title   = {Adaptive Per-Place Thresholding for Open-Set Visual Place Recognition},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026},
  note    = {Submitted}
}
```

Earlier work: "Adaptive Thresholding for Visual Place Recognition using
Negative Gaussian Mixture Statistics" (IEEE RCC 2025) — closed-set only.
