# VPR Tutorial — Full Codebase Context

*Updated: May 2026 (IEEE RAL submission, paper-ready, with threshold-only ablation column).*
*Three-step reviewer flow:*

```bash
bash setup.sh                                   # install + auto-download datasets
python -m experiments.extract_dinov2_salad_all  # GPU descriptor extraction
python -m experiments.final_all_datasets        # produces results JSON + Tables 2/3
```

---

## Overview

This repository implements Visual Place Recognition (VPR) with adaptive
per-place thresholding. Originally forked from Stefan Schubert's VPR Tutorial,
extended by the FRCV Lab (Fordham University, Dr. Damian Lyons) with:

1. **Online place discovery** — segment a reference stream into places without ground truth
2. **Per-place adaptive thresholds** — reject unknown queries using statistics from reference data only
3. **Open-set evaluation** — natural same-environment distractors (70% reference map)
4. **Head-to-head Vysotska et al. (ICRA 2025) comparison** — reimplemented sequence matcher + adaptive threshold, evaluated *both* as full pipeline and threshold-only (apples-to-apples thresholding)

Earlier closed-set work: "Adaptive Thresholding for VPR using Negative Gaussian
Mixture Statistics" (IEEE RCC 2025). Current submission adds the open-set
extension and targets IEEE RAL.

---

## Technology Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.11 (env name `vprtutorial`) |
| Deep Learning | PyTorch (DINOv2 SALAD via `torch.hub`) |
| Numerical | NumPy, SciPy (KS test), scikit-learn (GMM) |
| Visualization | Matplotlib (only for ad-hoc figures, not the reproducer) |
| Data | pickle (descriptor caches), JSON (results) |

`torch` and `tensorflow` versions in `requirements.txt` are pinned to what was
used for the paper. A CUDA GPU is needed only for the descriptor extraction step.

---

## Repository Structure

### Root — entry points & runners

| File | Role | Used by RAL paper? |
|------|------|--------------------|
| `README.md` | Reviewer-facing reproduction guide | yes |
| `setup.sh` | One-shot install + dataset auto-download | yes (step 1) |
| `requirements.txt` | Pinned dependencies | yes |
| `setup.py` | Legacy setuptools spec (upstream tutorial) | no |
| `demo.py` | End-to-end demo (legacy upstream) | no |
| `experiment_runner.py`, `test_runner.py`, `multi_dataset_runner.py`, `sequential_runner.py` | RCC 2025 pipeline (EigenPlaces / CosPlace / PatchNetVLAD) | no |
| `config.py`, `data_utils.py`, `utils.py` | Misc helpers (only `utils.normalize_l2` is used by the paper code) | partial |
| `prepare_mini_dataset.py` | Builds mini subsets (legacy) | no |

### `experiments/` — paper code

| File | Purpose |
|------|---------|
| `final_all_datasets.py` | **Canonical paper reproducer (Tables II and III).** Loads all 6 datasets with DINOv2 SALAD descriptors, runs discovery → threshold → filter-then-rank, compares to Vysotska. Writes `results/final_all_datasets_dinov2salad.json`. |
| `final_all_datasets_eigenplaces.py` | Same pipeline as above but reads EigenPlaces caches; produces **Tables IV and V** (descriptor-agnostic validation in §IV-E). Writes `results/final_all_datasets_eigenplaces.json`. |
| `extract_dinov2_salad_all.py` | GPU descriptor extraction for all 6 datasets via `torch.hub.load("serizba/salad", "dinov2_salad")`. Re-runs skip already-cached images. |
| `extract_eigenplaces_all.py` | GPU EigenPlaces (ResNet50, 2048-d) extraction for the same 6 datasets. Wipes target caches before extracting to guarantee a fresh run. Uses `Resize((480, 480))` so mixed-aspect-ratio datasets (ESSEX3IN1) don't crash `torch.stack`. |
| `online_place_discovery.py` | `OnlinePlaceDiscovery` class — bootstrap (α=1.5) → online (m=2, h=2), Welford's incremental statistics. |
| `vysotska_sequence_matcher.py` | Vysotska et al. graph-based matcher: DAG over (q,r) cells with real edges (cost=1−S) and hidden edges (fixed non-match cost), shortest path via topological sort. Self-contained — does not depend on the original `image_sequence_matcher/` package. |
| `vysotska_threshold.py` | Vysotska et al. adaptive threshold: KS bimodality test + 2-component GMM + 1D Kalman filter on similarity-matrix patches. |
| `robustness_sweep.py` | Sweeps α, m, h, divisor across the natural open-set protocol on all 6 datasets. Confirms α (F1 spread 0.20) and m (F1 stable for m≥2, catastrophic on ESSEX for m>2) are robust; divisor shows a real F1-vs-rejection Pareto tradeoff. Not reported in the paper. |
| `bayes_k_comparison.py` | Per-place Bayes-derived divisor `k = sep / (1 + σ⁺/σ⁻)` vs the fixed `sep/2`. Reduces to `sep/2` when σ⁺=σ⁻. Tested across 6 datasets; Bayes wins F1 on 4/6 but uniformly loses rejection (1–23 pts), so the paper keeps `sep/2`. Not reported in the paper. |

### `feature_extraction/` — legacy descriptor extractors

Inherited from the upstream tutorial. **Not used by the RAL paper code** —
`extract_dinov2_salad_all.py` calls `torch.hub` directly.

| File | Descriptor | Notes |
|------|-----------|-------|
| `feature_extractor_holistic.py` | AlexNet, EigenPlaces, CosPlace | upstream |
| `feature_extractor_dinov2_salad.py` | DINOv2 SALAD wrapper | superseded by direct hub call in extract_dinov2_salad_all.py |
| `feature_extractor_patchnetvlad.py` | PatchNetVLAD | upstream |
| `feature_extractor_local.py`, `feature_extractor_cosplace.py`, `feature_extractor_eigenplaces.py`, `feature_extractor.py`, `common.py` | Various | upstream |

### `IEEE_RAL_VPR/` — paper source

| File | Purpose |
|------|---------|
| `RCC2025.tex` | LaTeX manuscript ("Adaptive Per-Place Thresholding for Open-Set Visual Place Recognition") |
| `RCC2025.pdf` | Compiled PDF (6 pages) |
| `referencesB.bib` | Bibliography |
| `IEEEtran.cls`, `IEEEtran_HOWTO.pdf` | IEEE class file |
| `fig8_method_pipeline.png` | Method pipeline diagram (only figure used by the paper) |

### Other directories

| Directory | Purpose |
|-----------|---------|
| `images/` | Datasets — not committed; populated by `setup.sh`. ~5 GB once filled. |
| `cache/` | Descriptor caches — `cache/<Dataset>/<condition>/dinov2-salad/img_<i>_descriptor.pkl` |
| `results/` | JSON outputs from experiments. The reproducer writes `results/final_all_datasets_dinov2salad.json`. |
| `datasets/`, `evaluation/`, `matching/`, `feature_aggregation/` | Upstream tutorial helpers; not used by the paper reproducer |
| `visualizations/` | Ad-hoc plot scripts (not used by the paper reproducer) |

---

## Dataset Layout

After `bash setup.sh`, `images/` contains:

```
images/
├── GardensPoint/{day_left, day_right}/*.jpg              (200 each)
├── SFU/{dry, jan}/*.jpg  +  SFU/GT.npz                   (385 each)
├── bonn_example/{reference, query}/images/*.jpg          (488 / 544) + gt_bonn_example.txt
├── freiburg_example/{reference, query}/images/*.jpg      (361 / 676) + gt_freiburg_example.txt
├── ESSEX3IN1/{reference_combined, query_combined}/*.jpg  (210 each)
└── Nordland-500/{winter, summer}/*.png                   (500 each, 1 fps, first 500 sorted alphabetically from the HF dump)
```

**Vysotska ground-truth format** (Bonn, Freiburg) — `gt_<name>.txt`:
```
queryId numMatches refId1 refId2 ...
```

Per-query tolerance for matching (used by `final_all_datasets.py`): ±2 frames
for image-level datasets; ±1 frame for Nordland-500 (place-level).

---

## Descriptor Cache Layout

```
cache/<Dataset>/<condition_or_split>/dinov2-salad/img_<i>_descriptor.pkl
```

Examples:
- `cache/Nordland-500/winter/dinov2-salad/img_0_descriptor.pkl`
- `cache/Bonn/reference/dinov2-salad/img_0_descriptor.pkl`
- `cache/GardensPoint/day_left/dinov2-salad/img_0_descriptor.pkl`

Each `.pkl` contains `{"descriptor": np.float32 array (8448-dim)}`.

**IMPORTANT — numeric sort:** `sorted(glob(...))` orders lexicographically
(`img_100` before `img_10`), which would scramble Bonn/Freiburg results.
`final_all_datasets.py` uses a numeric regex sort key:

```python
def numeric_key(path):
    m = re.search(r'img_(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else 0
files = sorted(glob(os.path.join(cache_dir, "*.pkl")), key=numeric_key)
```

---

## Core Algorithms

### 1. Online Place Discovery (`OnlinePlaceDiscovery`)

Segments a sequential reference stream into places without ground truth.

**Two phases:**
1. **Bootstrap** — Accumulate frames, compute running mean/std of consecutive similarities. When similarity drops below `μ − α·σ`, the first place boundary is found.
2. **Online** — For each new frame, compare to current place's mean descriptor. If similarity drops below threshold for `h` consecutive frames (hysteresis), start a new place. Update statistics incrementally via Welford's algorithm.

**Parameters (paper defaults, uniform across all 6 datasets):**
- `bootstrap_std_factor α = 1.5`
- `min_place_size m = 2`
- `hysteresis h = 2`

### 2. Per-Place Adaptive Threshold

Computed entirely from reference data. No query-time computation.

For each place `p`:
1. Positive similarities = within-place pairs
2. Negative similarities = `p` vs all other places
3. Statistics: `μ_pos`, `μ_neg`, `σ_neg`
4. Separability: `sep = (μ_pos − μ_neg) / σ_neg`
5. **Continuous adaptive k**: `k = clip(sep / 2, 1, 2)`
6. Threshold: `θ_p = μ_neg + k · σ_neg`

`/2` accounts for the appearance gap between reference and query conditions.
The `[1, 2]` clip caps both ends — k≈1 for repetitive scenes (Nordland railway,
ESSEX warehouse), k≈2 for distinctive places (Freiburg intersections).

### 3. Filter-Then-Rank Query Evaluation

For each query:
1. Score against every discovered place (mean similarity to place members)
2. **Filter**: discard places where score < `θ_p`
3. **Rank**: among survivors, pick the highest-scoring place
4. If no places survive → reject as unknown (open-set)

### 4. Vysotska Pipeline (Reimplemented)

**Sequence matcher** (`vysotska_sequence_matcher.py`): Build DAG from
similarity matrix `S[q, r]`. Real edges have cost `1 − S[q, r]`, hidden edges
have fixed `non_matching_cost`. Shortest path via topological sort gives the
matching trajectory; queries on hidden segments are rejected.

**Adaptive threshold** (`vysotska_threshold.py`): For each query, extract a
20×20 patch from the similarity matrix. KS test for bimodality → if bimodal,
fit 2-component GMM → threshold = decision boundary. 1D Kalman filter smooths
thresholds across queries. Per-query threshold (uses similarity matrix at
query time).

### 5. DINOv2 SALAD Descriptor

- Model: `torch.hub.load("serizba/salad", "dinov2_salad")`
- Backbone: ViT-B/14
- Input: 322×322, bilinear interpolation, ImageNet normalization
- Output: 8448-dimensional descriptor (L2-normalized)
- Extracted on GPU cluster (erdos → ciscluster → node002, V100s)

---

## Evaluation Protocols (used by `final_all_datasets.py`)

### Closed-Set
All queries have a ground-truth match. Evaluate F1 (precision/recall).

### Natural Open-Set (same-environment distractors)
Use only the first 70% of reference images as the "known map." Queries that
map to the excluded 30% become natural distractors — visually similar (same
environment) but from unmapped areas. Simulates a robot driving through
familiar territory then entering new, unseen areas.

Both metrics reported: F1 on retained queries + Rejection Rate on distractors.

---

## Datasets (6 used by the RAL paper)

| Dataset | Ref / Query | Source | Size |
|---------|-------------|--------|------|
| Nordland-500 | 500 winter / 500 summer | Sünderhauf 2013 (HF mirror) | ~36 GB full / 500-img subset used |
| Bonn | 488 / 544 | Vysotska et al. (Prof. Stachniss / IPB Bonn) | 1.2 GB |
| Freiburg | 361 / 676 | Vysotska et al. (Prof. Stachniss / IPB Bonn) | 738 MB |
| GardensPoint | 200 day_left / 200 day_right | QUT (Schubert 2023) | 32 MB |
| SFU Mountain | 385 dry / 385 jan | Bruce et al. | 72 MB |
| ESSEX3IN1 | 210 / 210 | Zaffar 2020 | 1.5 GB |

All under one protocol: DINOv2 SALAD descriptors, tolerance ±2 frames (±1 for
Nordland), 70% reference fraction for natural open-set, Vysotska fanout=5,
discovery `m=2 / h=2 / α=1.5`, continuous adaptive k.

---

## Results (reproducible via `python -m experiments.final_all_datasets`)

### Closed-Set F1 (%)

| Dataset | Ours | Vys. (thresh.) | Vys. (full) |
|---------|------|----------------|-------------|
| Nordland-500 | 90.5 | 98.7 | 99.4 |
| Bonn | 89.2 | 89.5 | 90.4 |
| Freiburg | 85.8 | 91.7 | 91.8 |
| GardensPoint | 96.6 | 98.2 | 98.7 |
| SFU | 88.7 | 97.1 | 99.3 |
| ESSEX3IN1 | 91.2 | 98.5 | 99.5 |

Note: Vys. (thresh.) ≈ Baseline on closed-set — their per-query threshold
accepts almost everything when there are no distractors, the correct
behaviour.

### Natural Open-Set (70% reference map)

| Dataset | Ours F1 | Ours Rej | Vys-thr F1 | Vys-thr Rej | Vys F1 | Vys Rej |
|---------|---------|----------|------------|-------------|--------|---------|
| Nordland-500 | 87.1 | 100.0% | 81.4 | 0.0% | 95.1 | 77.3% |
| Bonn | 92.1 | 92.2% | 86.2 | 33.0% | 93.1 | 89.3% |
| Freiburg | 88.0 | 99.6% | 76.7 | 11.3% | 94.4 | 93.7% |
| GardensPoint | 95.9 | 100.0% | 83.6 | 20.0% | 98.2 | 96.7% |
| SFU | 71.8 | 79.3% | 81.7 | 17.2% | 91.8 | 60.3% |
| ESSEX3IN1 | 80.1 | 60.3% | 81.0 | 4.8% | 92.4 | 65.1% |

**Key findings:**

1. **Apples-to-apples thresholding (Ours vs Vys-thr).** Our per-place
   thresholds reject same-environment distractors better than per-query
   thresholding on **every dataset** (Vys-thr: 0–33% rejection; Ours:
   60–100%). On open-set F1 we win on the four visually structured
   datasets and lose marginally on SFU / ESSEX3IN1, where neither
   standalone threshold is doing meaningful work.
2. **Pipeline vs pipeline (Ours vs Vys full).** On the four structured
   datasets we reject 92–100% of distractors vs Vysotska's full pipeline
   77–97%, while staying within 2–8 F1 points. The full pipeline's high
   rejection rate is carried by the sequence matcher's graph traversal,
   not the adaptive threshold.
3. **Sequence-dominated environments.** On SFU (forest trail) and
   ESSEX3IN1 (corridors), images are visually near-degenerate; sequential
   ordering carries the signal. The full pipeline retains its F1 edge
   there because of the local-window prior, while neither standalone
   threshold (ours or theirs) is reliable.

### Descriptor-Agnostic Validation (EigenPlaces, Tables IV–V)

Reproducible via `python -m experiments.final_all_datasets_eigenplaces`
after running `python -m experiments.extract_eigenplaces_all` (GPU
recommended).

**Closed-set F1 (%):**

| Dataset | Ours | Vys. (thresh.) | Vys. (full) |
|---------|------|----------------|-------------|
| Nordland-500 | 82.1 | 95.5 | 98.2 |
| Bonn | 73.7 | 89.2 | 92.1 |
| Freiburg | 22.6 | 60.3 | 61.9 |
| GardensPoint | 96.1 | 98.2 | 99.0 |
| SFU | 67.2 | 90.0 | 98.7 |
| ESSEX3IN1 | 85.9 | 96.3 | 99.0 |

**Natural open-set F1 / rejection rate (%):**

| Dataset | Ours F1 | Ours Rej | Vys-thr F1 | Vys-thr Rej | Vys F1 | Vys Rej |
|---------|---------|----------|------------|-------------|--------|---------|
| Nordland-500 | 76.3 | 98.7% | 79.1 | 1.3% | 89.2 | 52.0% |
| Bonn | 73.0 | 87.4% | 82.9 | 10.7% | 93.9 | 86.4% |
| Freiburg | 21.8 | 98.7% | 45.9 | 10.1% | 46.8 | 11.3% |
| GardensPoint | 94.8 | 95.0% | 87.8 | 41.7% | 98.6 | 95.0% |
| SFU | 52.8 | 62.9% | 73.1 | 16.4% | 86.4 | 33.6% |
| ESSEX3IN1 | 77.0 | 31.8% | 80.2 | 1.6% | 87.9 | 39.7% |

The relative ordering between thresholding mechanisms is preserved:
per-place thresholds beat per-query thresholds on rejection on **all six
datasets** by factors of ~1.5× to >75×. Absolute F1 drops because
EigenPlaces is a weaker descriptor (baseline drops by 5 pts on Nordland
and 30+ pts on Freiburg). Cited in §III-A and §IV-E of the paper.

---

## Reviewer Reproduction Steps

```bash
git clone https://github.com/NickTrinh/VPR_Tutorial.git
cd VPR_Tutorial

conda create -n vprtutorial python=3.11
conda activate vprtutorial

bash setup.sh                                    # ~10–20 min on a fast link
python -m experiments.extract_dinov2_salad_all   # GPU; ~10 min when cached, longer cold
python -m experiments.final_all_datasets         # CPU; a few minutes
```

Output: `results/final_all_datasets_dinov2salad.json` plus a printed summary
matching paper Tables 2 and 3.

`setup.sh` auto-downloads Bonn, Freiburg, GardensPoint, SFU, and ESSEX3IN1
from public mirrors. Nordland-500 requires a manual step (HuggingFace gated
LFS) — clear instructions are printed at the end of the script.

---

## Known Issues & Fixes

1. **Numeric vs. lexicographic sort.** `sorted(glob(...))` puts `img_100`
   before `img_10`. Use `numeric_key` regex (see Cache Layout above). GP / NL
   / SFU were unaffected because their loaders use a different path.

2. **Fixed k=2 was too aggressive on Nordland.** Railway scenes are repetitive
   (low separability); k=2 rejected too many genuine queries (F1=75.5%).
   Replaced with continuous `k = clip(sep/2, 1, 2)`.

3. **Removed `image_sequence_matcher/` external dependency.** Earlier versions
   imported Vysotska's published package; the matcher is now self-contained
   inside `experiments/vysotska_sequence_matcher.py` to keep the repo
   single-source.
