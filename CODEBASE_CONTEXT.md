# VPR Tutorial — Full Codebase Context

*Updated: April 2026 (post-RCC, IEEE RAL submission in progress)*

## Overview

This repository implements Visual Place Recognition (VPR) with adaptive per-place thresholding. Originally forked from Stefan Schubert's VPR Tutorial, it has been extended by the FRCV Lab (Fordham University, Dr. Damian Lyons) with:

1. **Online place discovery** — segment a reference stream into places without ground truth
2. **Per-place adaptive thresholds** — reject unknown queries using statistics computed solely from reference data
3. **Open-set evaluation** — test distractor rejection (cross-environment and natural same-environment)
4. **Head-to-head comparison with Vysotska et al. (ICRA 2025)** — reimplemented their full pipeline (sequence matcher + adaptive threshold)

Published work: "Adaptive Thresholding for Visual Place Recognition using Negative Gaussian Mixture Statistics" (IEEE RCC 2025, closed-set only). Current submission targets IEEE RAL with open-set extensions.

---

## Technology Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.8+ |
| Environment | Conda (`vprtutorial`) |
| Deep Learning | PyTorch (descriptors, torch.hub models), TensorFlow (DELF only) |
| Numerical | NumPy, SciPy (GMM, KS test, Kalman filter) |
| Visualization | Matplotlib |
| Config | PyYAML (Vysotska dataset configs) |
| Data | pickle (descriptor caches), JSON (results) |

---

## Repository Structure

### Root — Entry Points & Runners

| File | Purpose |
|------|---------|
| `demo.py` | End-to-end demo: dataset + descriptor + matching + Recall@K |
| `experiment_runner.py` | Run N experiments with cross-validation, generate per-place thresholds |
| `test_runner.py` | Evaluate with place/image-level thresholds (TP/FP/TN/FN metrics) |
| `multi_dataset_runner.py` | Batch pipeline across multiple datasets and descriptors |
| `sequential_runner.py` | **[NEW]** Sequential VPR pipeline: discover places online → compute thresholds → evaluate closed-set and open-set. Presets for gardens_point, nordland_mini, sfu |
| `config.py` | Dataset and experiment configuration (DatasetConfig, ExperimentConfig dataclasses) |
| `data_utils.py` | Dataset loading utilities |
| `utils.py` | General utilities (normalize_l2, etc.) |
| `prepare_mini_dataset.py` | Creates mini dataset subsets for quick testing |
| `setup.py` | Package installation |

### `feature_extraction/` — Descriptor Extractors

| File | Descriptor | Dimensions | Notes |
|------|-----------|------------|-------|
| `feature_extractor_holistic.py` | AlexNet (conv3), EigenPlaces, CosPlace | 64896 / 2048 / 512 | ResNet50 via torch.hub |
| `feature_extractor_dinov2_salad.py` | **[NEW]** DINOv2 SALAD | 8448 | ViT-B/14, 322x322 input, `torch.hub serizba/salad` |
| `feature_extractor_patchnetvlad.py` | PatchNetVLAD | 4096 | Local + global |
| `feature_extractor_local.py` | Local features | varies | |
| `feature_extractor_cosplace.py` | CosPlace (stub) | 512 | |
| `feature_extractor_eigenplaces.py` | EigenPlaces (stub) | 2048 | |
| `feature_extractor.py` | Main interface | — | Routes to specific extractors |
| `common.py` | Shared utilities | — | Device management, preprocessing |

### `experiments/` — Experiment Scripts

#### Core Algorithms
| File | Purpose |
|------|---------|
| `online_place_discovery.py` | **[NEW]** OnlinePlaceDiscovery class — truly online, processes one frame at a time with Welford's incremental statistics |
| `sequential_place_discovery.py` | Offline sequential variant (requires precomputed similarity matrix) |
| `vysotska_sequence_matcher.py` | **[NEW]** Wrapper around Vysotska's graph-based sequence matching (DAG shortest path with non-matching cost) |
| `vysotska_threshold.py` | **[NEW]** Vysotska adaptive threshold: p×p patch extraction → KS bimodality test → 2-component GMM → 1D Kalman filter smoothing |
| `experiment_utils.py` | Shared experiment utilities |

#### Paper Experiments (RAL)
| File | Purpose |
|------|---------|
| `final_all_datasets.py` | **[NEW]** Consolidated evaluation on all 3 Vysotska datasets (Nordland-500, Bonn, Freiburg). Runs closed-set, cross-env open-set, natural open-set. Outputs JSON results |
| `bonn_freiburg_openset.py` | **[NEW]** Cross-environment open-set: inject GardensPoint distractors into Bonn/Freiburg query stream |
| `bonn_freiburg_natural_openset.py` | **[NEW]** Natural open-set: truncate reference to 70%, treat excluded-region queries as same-environment distractors |
| `nordland_500_paper_experiments.py` | **[NEW]** Nordland-500 specific experiments for paper |
| `nordland_500_vysotska_validation.py` | **[NEW]** Validate Vysotska reimplementation against their reported Nordland numbers |
| `vysotska_bonn_freiburg_validation.py` | **[NEW]** Validate reimplementation on Bonn/Freiburg against Vysotska's reported results |

#### Feature Extraction (GPU)
| File | Purpose |
|------|---------|
| `extract_dinov2_salad.py` | **[NEW]** Extract DINOv2 SALAD descriptors for a single dataset |
| `extract_dinov2_salad_all.py` | **[NEW]** Batch extraction across all datasets |
| `extract_bonn_freiburg.py` | **[NEW]** Extract descriptors for Bonn/Freiburg datasets |

#### Analysis & Visualization
| File | Purpose |
|------|---------|
| `generate_paper_figures.py` | **[NEW]** Generate all paper figures (similarity matrices, threshold plots, F1 comparisons) |
| `compare_thresholds.py` | Compare threshold methods (mean_bad, weighted_mean_bad, filter_n, etc.) |
| `per_place_threshold_viz.py` | Visualize per-place threshold distributions |
| `filter_n_sensitivity.py` | Sensitivity analysis of filter_n parameter |
| `distractor_ratio_sweep.py` | Sweep distractor ratios for robustness analysis |
| `scaling_analysis.py` | Computational scaling analysis |
| `dynamic_place_recognition.py` | Dynamic place recognition with online updates |
| `discovery_recognition_mini.py` | Combined discovery + recognition on mini datasets |
| `discovery_then_recognition.py` | Two-stage pipeline: discover then recognize |
| `compare_with_distractors.py` | Comparison with distractor variations |

### `image_sequence_matcher/` — Vysotska's Sequence Matcher

Adapted from [ovysotska/image_sequence_matcher](https://github.com/ovysotska/image_sequence_matcher).

| File | Purpose |
|------|---------|
| `src/graph.py` | DAG construction for sequence matching. Nodes = (query, ref) cells; edges = real matches (dissimilarity cost) or hidden transitions (fixed non-matching cost) |
| `src/path_tools.py` | Path extraction utilities |
| `src/topological_sorting.py` | Topological sort for efficient DAG traversal |
| `img_seq_matcher.py` | Main interface |

### `IEEE_RAL_VPR/` — Paper Source

| File | Purpose |
|------|---------|
| `RCC2025.tex` | Main LaTeX manuscript (title: "Adaptive Per-Place Thresholding for Open-Set Visual Place Recognition") |
| `referencesB.bib` | Bibliography |
| `fig8_method_pipeline.png` | Method pipeline diagram |
| `fig14_filter_n_sensitivity.png` | Filter sensitivity ablation figure |
| `fig16_distractor_ratio_sweep.png` | Distractor robustness figure |

### Other Directories

| Directory | Purpose |
|-----------|---------|
| `datasets/` | Dataset manifests and metadata |
| `evaluation/` | Evaluation metrics and scoring functions |
| `matching/` | Matching algorithms (cosine similarity, etc.) |
| `feature_aggregation/` | Feature pooling/aggregation methods |
| `visualizations/` | Generated visualization outputs |
| `results/` | Experiment results (JSON files, organized by dataset/descriptor) |

---

## Dataset Formats

### Format 1: `landmark` (original)
```
images/<DatasetName>/Place###/Image###.jpg
```
Each place is a directory containing condition variants.

### Format 2: `sequential` (Nordland, GardensPoint, SFU)
```
images/<DatasetName>/<condition>/img_<index>.png
```
Plus `GT.npz` with ground-truth matching matrix. Conditions are environment variants (e.g., `summer`, `winter`, `day_left`, `night_right`).

### Format 3: `landmark_grouped` (mini datasets)
```
images/<DatasetName>/<condition>/Place####_Cond##_G##_*.jpg
```
Groups of images per place, with condition and group identifiers.

### Format 4: `vysotska` (Bonn, Freiburg) **[NEW]**
```
images/<dataset_name>/reference/images/img_<index>_<timestamp>.png
images/<dataset_name>/query/images/img_<index>_<timestamp>.png
images/<dataset_name>/gt_<name>.txt
images/<dataset_name>/config.yaml
```

**Ground truth format** (`gt_<name>.txt`):
```
queryId numMatches refId1 refId2 ...
```
Each line: query index, number of matching references, then the matching reference indices.

**Config format** (`config.yaml`):
```yaml
fanOut: 5
nonMatchCost: 0.25
imgExt: .png
querySize: 544
```

---

## Descriptor Cache Layout

```
cache/<DatasetName>/<condition_or_split>/<descriptor_name>/img_<index>_descriptor.pkl
```

Examples:
- `cache/Nordland_salad/winter/dinov2-salad/img_0_descriptor.pkl`
- `cache/Bonn/reference/dinov2-salad/img_0_descriptor.pkl`
- `cache/GardensPoint/day_left/eigenplaces/img_0_descriptor.pkl`

Each `.pkl` contains either a raw numpy array or a dict `{"descriptor": array}`.

**IMPORTANT**: Cache files must be loaded with numeric sorting (not lexicographic). Lexicographic sort puts `img_100` before `img_10`, scrambling descriptor order. Use:
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

1. **Bootstrap** — Accumulate frames, computing running mean and std of consecutive similarities. When similarity drops below `mu - 1.5*sigma`, the first place boundary is found.

2. **Online** — For each new frame:
   - Compare to current place's mean descriptor
   - If similarity drops below threshold for `h=2` consecutive frames (hysteresis), start a new place
   - Update statistics incrementally via Welford's algorithm (no need to store all past similarities)

**Parameters:**
- `bootstrap_std_factor=1.5` — How many stds below mean triggers first boundary
- `min_place_size=3` — Minimum frames per place
- `hysteresis=2` — Consecutive low-similarity frames needed to confirm boundary

**Output:** List of places, each a list of frame indices.

### 2. Per-Place Adaptive Threshold

Computed entirely from reference data during place discovery. No query-time computation.

**For each place `p`:**
1. Compute positive similarities: similarities between frames within place `p` (intra-place)
2. Compute negative similarities: similarities between frames in place `p` and frames in all other places (inter-place)
3. Fit statistics: `mu_pos`, `mu_neg`, `sigma_neg`
4. Compute separability: `sep = (mu_pos - mu_neg) / sigma_neg`
5. Set adaptive k: `k = max(1, min(floor(sep / 2), 2))`
6. Threshold: `theta_p = mu_neg + k * sigma_neg`

**Intuition:**
- `sep` measures how many standard deviations apart the "matching" and "non-matching" distributions are
- `floor(sep/2)` places the threshold halfway between the noise floor and where matches land
- The `/2` accounts for the appearance gap between reference and query conditions (different season, time of day)
- Capped at [1, 2]: k=1 for repetitive scenes (e.g., Nordland railway), k=2 for distinctive places (e.g., Freiburg intersections)

### 3. Filter-Then-Rank Query Evaluation

For each query:
1. Compute similarity to all reference places (mean similarity to place members)
2. **Filter**: discard places where score < `theta_p` (per-place threshold)
3. **Rank**: among surviving places, pick the best match
4. If no places survive → reject query as unknown (open-set)

### 4. Vysotska Pipeline (Reimplemented)

**Sequence Matcher** (`vysotska_sequence_matcher.py`):
- Build DAG from similarity matrix S[q, r]
- Nodes: each (query, ref) cell
- Edges: "real" edges with cost = `1 - S[q, r]`, "hidden" edges with fixed `non_matching_cost`
- Shortest path via topological sort determines matches
- Queries on hidden segments are rejected

**Adaptive Threshold** (`vysotska_threshold.py`):
- For each query, extract a `p×p` patch (default 20×20) from the similarity matrix
- KS test for bimodality — if bimodal:
  - Fit 2-component GMM
  - Threshold = decision boundary between components
- 1D Kalman filter smooths thresholds over time
- Per-query threshold (requires similarity matrix at query time)

### 5. DINOv2 SALAD Descriptor

- Model: `torch.hub.load("serizba/salad", "dinov2_salad")`
- Backbone: ViT-B/14
- Input: 322×322, bilinear interpolation, ImageNet normalization
- Output: 8448-dimensional descriptor
- Extracted on GPU cluster (erdos → ciscluster → node002, V100s)

---

## Evaluation Protocols

### Closed-Set
All queries have a ground-truth match in the reference set. Evaluate F1 (precision/recall) with tolerance (±1 for Nordland place-level, ±2 for Bonn/Freiburg image-level).

### Open-Set: Cross-Environment Distractors
Inject images from a completely different dataset (e.g., 200 GardensPoint campus images) into the query stream. These are visually dissimilar — tests basic rejection capability. Both our method and Vysotska achieve ~100% rejection.

### Open-Set: Natural Same-Environment Distractors
Use only the first 70% of reference images as the "known map." Queries that map to the excluded 30% are natural distractors — visually similar (same city, same conditions) but from unmapped areas. This simulates a robot driving through familiar territory then entering new, unseen areas. Much harder than cross-environment.

---

## Configured Datasets

### In `config.py` (DATASETS dict)
| Key | Images | Places | Type | Conditions |
|-----|--------|--------|------|------------|
| `gardens_point_mini` | 20 | 20 | sequential | day_left / night_right |
| `sfu_mini` | 192 | 192 | sequential | dry / jan |
| `nordland_mini` | ~130 | 4–14 | landmark_grouped | summer / winter (various group/step) |
| `nordland_mini_2` | ~130 | varies | landmark_grouped | Additional Nordland variant |
| `nordland_mini_3` | ~130 | varies | landmark_grouped | Additional Nordland variant |
| `fordham_places` | 33 | 11 | landmark | 3 images per place |

### Vysotska-format datasets (loaded directly by experiment scripts)
| Dataset | Ref Images | Query Images | Source |
|---------|-----------|-------------|--------|
| Nordland-500 | 500 | 500 | Nordland winter/summer, filtered to 500 |
| Bonn | 488 | 544 | Vysotska et al., urban driving |
| Freiburg | 361 | 676 | Vysotska et al., urban driving |

---

## Results (DINOv2 SALAD, Adaptive k)

### Closed-Set (F1%)

| Dataset | Ours | Vysotska |
|---------|------|----------|
| Nordland-500 | 92.9 | 99.4 |
| Bonn | 89.8 | 90.0 |
| Freiburg | 93.1 | 91.8 |

### Open-Set: Cross-Environment Distractors

| Dataset | Ours F1 | Ours Rej | Vysotska F1 | Vysotska Rej |
|---------|---------|----------|-------------|--------------|
| Nordland-500 | 92.9 | 100% | 99.5 | 100% |
| Bonn | 89.8 | 100% | 90.0 | 100% |
| Freiburg | 93.1 | 100% | 91.8 | 100% |

### Open-Set: Natural Same-Environment (70% reference map)

| Dataset | Ours F1 | Ours Rej | Vysotska F1 | Vysotska Rej |
|---------|---------|----------|-------------|--------------|
| Nordland-500 | 90.1 | 88.7% | 95.1 | 77.3% |
| Bonn | 93.1 | 92.2% | 93.1 | 89.3% |
| Freiburg | 91.7 | 99.6% | 94.4 | 93.7% |

**Key finding:** Our method consistently rejects more same-environment distractors (88.7–99.6% vs 77.3–93.7%) while maintaining competitive F1. Vysotska's sequence matcher excels at closed-set sequential alignment but lacks robust rejection for open-set scenarios.

---

## Experiment Pipeline (sequential_runner.py)

```
1. Load descriptors (reference + query conditions)
2. Online place discovery on reference stream
   → List of places (frame index groups)
3. Compute per-place thresholds from reference-only statistics
   → theta_p = mu_neg + k * sigma_neg (adaptive k)
4. For each query:
   a. Score against all places
   b. Filter by per-place threshold
   c. Rank surviving places → match or reject
5. Evaluate: F1, precision, recall, rejection rate
6. Compare against: baseline (no threshold), Vysotska full pipeline
7. Save results to JSON
```

---

## Key File Paths

| What | Where |
|------|-------|
| Dataset images | `images/<DatasetName>/` |
| Descriptor cache | `cache/<DatasetName>/<condition>/<descriptor>/` |
| Experiment results | `results/<experiment_name>.json` |
| Paper source | `IEEE_RAL_VPR/RCC2025.tex` |
| Paper figures | `IEEE_RAL_VPR/fig*.png` |
| Ground truth (sequential) | `images/<DatasetName>/GT.npz` |
| Ground truth (Vysotska) | `images/<dataset>/gt_<name>.txt` |
| Dataset configs (Vysotska) | `images/<dataset>/config.yaml` |

---

## Known Issues & Fixes

1. **Descriptor cache sort bug** — `sorted(glob(...))` sorts lexicographically (`img_100` before `img_10`), scrambling Bonn/Freiburg results. Fixed with numeric regex sort key. GP/NL/SFU unaffected (different loading path).

2. **Fixed k=2 too aggressive on Nordland** — Railway scenes are repetitive (low separability). k=2 rejected too many genuine queries (F1=75.5%). Replaced with adaptive k based on separability score.

3. **Nordland-500 paper table error** — Previously showed baseline F1=94.7% as "Ours" because the old `mean_bad` threshold happened to reject nothing, equaling baseline. Corrected with adaptive k results.
