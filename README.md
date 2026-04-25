# Adaptive Per-Place Thresholding for Open-Set Visual Place Recognition

Code accompanying the IEEE RAL submission from FRCV Lab, Fordham University
(Nick Trinh, Damian Lyons). Paper source: [`IEEE_RAL_VPR/RCC2025.tex`](IEEE_RAL_VPR/RCC2025.tex).

The method discovers places online from a sequential reference stream, then
assigns each place its own rejection threshold `θ_k = μ_k⁻ + k · σ_k⁻`, where
`k = clip(sep/2, 1, 2)` and `sep` is computed from reference-to-reference
statistics only — no query-time tuning, no ground-truth labels. Queries whose
score to every discovered place falls below its threshold are rejected as
unknown (open-set).

---

## Install

```bash
git clone https://github.com/NickTrinh/VPR_Tutorial.git
cd VPR_Tutorial

conda create -n vprtutorial python=3.11
conda activate vprtutorial
pip install -r requirements.txt
```

`torch` and `tensorflow` versions in `requirements.txt` are pinned to what was
used for the paper; a GPU with CUDA is needed only for descriptor extraction.

---

## Datasets

The repository does **not** ship the datasets — combined they exceed 40 GB.
Download each from its canonical source and unpack into `images/` with the
directory layout shown below.

| Dataset        | Size   | Download |
|----------------|--------|----------|
| GardensPoint   | 32 MB  | [QUT Wiki](https://wiki.qut.edu.au/display/cyphy/Day+and+Night+with+Lateral+Pose+Change+Datasets) — also auto-fetched by [stschubert/VPR_Tutorial](https://github.com/stschubert/VPR_Tutorial) |
| SFU Mountain   | 72 MB  | <http://autonomy.cs.sfu.ca/sfu-mountain-dataset/> |
| Bonn           | 1.2 GB | <http://www.ipb.uni-bonn.de/html/projects/visual_place_recognition/bonn_example.zip> |
| Freiburg       | 738 MB | <http://www.ipb.uni-bonn.de/html/projects/visual_place_recognition/freiburg_example.zip> |
| ESSEX3IN1      | 1.5 GB | <https://github.com/MubarizZaffar/ESSEX3IN1-Dataset> |
| Nordland       | ~36 GB full, 500-image subset used | <https://huggingface.co/datasets/Somayeh-h/Nordland> |

Bonn and Freiburg are direct zips and can be fetched non-interactively:

```bash
mkdir -p images && cd images
wget http://www.ipb.uni-bonn.de/html/projects/visual_place_recognition/bonn_example.zip
wget http://www.ipb.uni-bonn.de/html/projects/visual_place_recognition/freiburg_example.zip
unzip bonn_example.zip && unzip freiburg_example.zip && cd ..
```

Expected layout after download:

```
images/
├── GardensPoint/{day_left, day_right}/*.jpg         (200 each)
├── SFU/{dry, jan}/*.jpg  +  SFU/GT.npz              (385 each)
├── bonn_example/{reference, query}/images/*.jpg     (488 / 544)  + gt_bonn_example.txt
├── freiburg_example/{reference, query}/images/*.jpg (361 / 676)  + gt_freiburg_example.txt
├── ESSEX3IN1/{reference_combined, query_combined}/*.jpg (210 each)
└── Nordland_Mini/{winter, summer}/*.jpg             (first 500 each, 1 fps)
```

For Nordland we take the first 500 frames of the 1 fps winter and summer
traversals to match Vysotska et al.'s protocol.

---

## Reproduce the paper

### 1. Extract DINOv2 SALAD descriptors (GPU required)

```bash
python -m experiments.extract_dinov2_salad_all
```

Output is cached to `cache/<dataset>/<condition>/dinov2-salad/*.pkl`.
Re-runs skip already-cached images.

### 2. Run all 6 datasets, closed-set + natural open-set

```bash
python -m experiments.final_all_datasets
```

Writes `results/final_all_datasets_dinov2salad.json` and prints the summary
tables that populate the paper's Tables 2 and 3. Runs in a few minutes on CPU
once descriptors are cached.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `experiments/final_all_datasets.py` | Canonical paper reproducer — loads 6 datasets, runs discovery → threshold → filter-then-rank, compares to Vysotska sequence matcher. |
| `experiments/online_place_discovery.py` | `OnlinePlaceDiscovery` class — bootstrap (α=1.5) → online (m=2, h=2), Welford's algorithm for per-place statistics. |
| `experiments/vysotska_threshold.py` | Vysotska et al. adaptive threshold: KS test + 2-GMM + 1D Kalman filter on similarity-matrix patches. |
| `experiments/vysotska_sequence_matcher.py` | Vysotska et al. graph-based sequence matcher (shortest-path on DAG built from similarity matrix). |
| `experiments/extract_dinov2_salad_all.py` | GPU descriptor extraction for all 6 datasets via `torch.hub.load("serizba/salad", "dinov2_salad")`. |
| `IEEE_RAL_VPR/` | Paper source (`RCC2025.tex`), bibliography, and generated PDF. |
| `sequential_runner.py`, `demo.py`, `experiment_runner.py`, `test_runner.py`, `multi_dataset_runner.py` | RCC 2025 pipeline (EigenPlaces / CosPlace / PatchNetVLAD). Not used by the RAL paper. |

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
