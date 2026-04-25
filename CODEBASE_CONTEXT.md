# VPR Tutorial — Full Codebase Context

*Updated: April 2026 (IEEE RAL submission, paper-ready, post-cleanup).*
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
4. **Head-to-head Vysotska et al. (ICRA 2025) comparison** — reimplemented sequence matcher + adaptive threshold

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

### `experiments/` — paper code (5 files only)

| File | Purpose |
|------|---------|
| `final_all_datasets.py` | **Canonical paper reproducer.** Loads all 6 datasets, runs discovery → threshold → filter-then-rank, compares to Vysotska. Writes `results/final_all_datasets_dinov2salad.json` and prints the summary tables that populate paper Tables 2 and 3. |
| `extract_dinov2_salad_all.py` | GPU descriptor extraction for all 6 datasets via `torch.hub.load("serizba/salad", "dinov2_salad")`. Re-runs skip already-cached images. |
| `online_place_discovery.py` | `OnlinePlaceDiscovery` class — bootstrap (α=1.5) → online (m=2, h=2), Welford's incremental statistics. |
| `vysotska_sequence_matcher.py` | Vysotska et al. graph-based matcher: DAG over (q,r) cells with real edges (cost=1−S) and hidden edges (fixed non-match cost), shortest path via topological sort. Self-contained — does not depend on the original `image_sequence_matcher/` package. |
| `vysotska_threshold.py` | Vysotska et al. adaptive threshold: KS bimodality test + 2-component GMM + 1D Kalman filter on similarity-matrix patches. |

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
└── Nordland_Mini/{winter, summer}/*.jpg                  (first 500 each, 1 fps)
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
- `cache/Nordland_salad/winter/dinov2_salad/img_0_descriptor.pkl`
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

| Dataset | Ours | Vysotska |
|---------|------|----------|
| Nordland-500 | 90.5 | 99.4 |
| Bonn | 89.2 | 90.4 |
| Freiburg | 85.8 | 91.8 |
| GardensPoint | 96.6 | 98.7 |
| SFU | 88.7 | 99.3 |
| ESSEX3IN1 | 91.2 | 99.5 |

### Natural Open-Set (70% reference map)

| Dataset | Ours F1 | Ours Rej | Vysotska F1 | Vysotska Rej |
|---------|---------|----------|-------------|--------------|
| Nordland-500 | 87.1 | 100.0% | 95.1 | 77.3% |
| Bonn | 92.1 | 92.2% | 93.1 | 89.3% |
| Freiburg | 88.0 | 99.6% | 94.4 | 93.7% |
| GardensPoint | 95.9 | 100.0% | 98.2 | 96.7% |
| SFU | 71.8 | 79.3% | 91.6 | 59.5% |
| ESSEX3IN1 | 80.1 | 60.3% | 92.7 | 66.7% |

**Key finding:** On the four structured datasets (Nordland, Bonn, Freiburg,
GardensPoint), our method rejects 92–100% of same-environment distractors vs
Vysotska's 77–97%, while staying within 2–8 F1 points. On SFU and ESSEX3IN1,
where scenes are visually near-degenerate, Vysotska's sequence-matcher prior
dominates F1 but rejection drops to 60–67%.

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
