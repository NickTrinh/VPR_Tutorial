# Threshold Analysis Tools

This document describes the analysis tools available to investigate why the two threshold calculation methods (original vs legacy) may not diverge significantly on large datasets.

## Background

Two threshold calculation methods:
- **Legacy**: `threshold = mean_bad` (mean of bad scores only)
- **Original**: `threshold = mean_bad + filter_n × std_dev` (adds margin based on score distribution)

## Analysis Tools

### Analysis #1: Threshold Component Distribution

**Purpose**: Understand the distribution of `filter_n` and `std_dev` values to see if the additional term (`filter_n × std_dev`) is negligible.

**How to run**: 
```bash
# Automatically runs when you execute experiments
python multi_dataset_runner.py --dataset gardenspoint_mini --experiment-only --descriptor eigenplaces --threshold-method original --num-runs 50
```

**What it shows**:
- Statistics (mean, std, min, max, median) for:
  - `filter_n` values
  - `std_dev` (of bad scores)
  - `mean_bad` values
  - Additional term (`filter_n × std_dev`)
- Relative impact: what percentage the additional term adds to mean_bad
- Interpretation of current threshold method

**Key insights**:
- If `filter_n` is consistently small (< 1), the additional term has minimal impact
- If `std_dev` is very small, even large `filter_n` won't change thresholds much
- If relative impact is < 5%, the two methods will produce similar results

---

### Analysis #2: Side-by-Side Threshold Comparison

**Purpose**: Compare actual threshold values from both methods to quantify differences.

**How to run**:
```bash
# Step 1: Run experiments with both methods
python multi_dataset_runner.py --dataset gardenspoint_mini --experiment-only --threshold-method legacy_mean_bad --descriptor eigenplaces --num-runs 100
python multi_dataset_runner.py --dataset gardenspoint_mini --experiment-only --threshold-method original --descriptor eigenplaces --num-runs 100

# Step 2: Compare results
python run_threshold_comparison.py --dataset gardenspoint_mini
```

**What it shows**:
- Scatter plots comparing legacy vs original thresholds (both simple and weighted)
- Histograms showing distribution of percentage differences
- Statistics: mean difference, max difference, mean % difference
- Saved outputs:
  - `threshold_comparison_<dataset>.png` - visualization
  - `threshold_comparison_<dataset>.csv` - detailed comparison data

**Key insights**:
- If scatter plots cluster near the y=x line, methods are nearly identical
- If mean % difference is < 5%, differences are negligible
- Large differences indicate the additional term is significant

---

### Analysis #3: Detailed Per-Run Statistics

**Purpose**: Inspect individual run details to see variability across runs.

**How to use**:
Call `save_detailed_run_stats()` from `threshold_analysis.py` within your experiment loop to save per-run data for each place.

**Output**: `results/<Dataset>/run_details_place_<place_id>.csv`

**Columns**:
- `run`: Run number
- `num_good`, `num_bad`: Count of positive/negative matches
- `mean_good`, `mean_bad`: Mean similarity scores
- `std_good`, `std_bad`: Standard deviations
- `std_dev`: Standard deviation used in threshold calculation
- `filter_n`: Calculated filter_n value
- `threshold_legacy`: Threshold using legacy method
- `threshold_original`: Threshold using original method

**Key insights**:
- See how thresholds vary across runs for the same place
- Identify places where methods diverge significantly
- Understand stability of threshold estimation

---

### Analysis #4: Score Distribution Visualization

**Purpose**: Visualize the separation between good and bad similarity scores to understand descriptor quality.

**How to run**:
```bash
python run_score_analysis.py --dataset gardenspoint_mini --descriptor eigenplaces --num-queries 16
```

**What it shows**:
- Histograms of good vs bad scores for multiple queries
- Visual representation of score overlap
- Statistics on separation (in standard deviations)
- Saved output: `score_distributions_<dataset>_<descriptor>.png`

**Key insights**:
- **HIGH separation (>5σ)**: Good and bad scores are well-separated
  - Descriptor is high-quality
  - Additional margin (`filter_n × std_dev`) has minimal impact
  - **This likely explains why methods don't diverge!**
- **MODERATE separation (2-5σ)**: Some overlap exists
  - Additional margin may help reduce false positives
- **LOW separation (<2σ)**: Significant overlap
  - Challenging task, both methods may struggle

---

## Hypothesis: Why Methods Don't Diverge

Based on these analyses, the most likely explanation is:

**Modern descriptors (EigenPlaces, CosPlace) produce such well-separated good/bad score distributions that:**
1. Standard deviation of bad scores is very small (tight cluster of negatives)
2. `filter_n` is small because good scores are far from bad scores
3. Therefore `filter_n × std_dev ≈ 0`
4. Result: `mean_bad` and `mean_bad + filter_n × std_dev` are nearly identical

### Testing This Hypothesis

Compare descriptor quality:
```bash
# High-quality descriptor (expected: low divergence)
python run_score_analysis.py --dataset gardenspoint_mini --descriptor eigenplaces

# Medium-quality descriptor (expected: moderate divergence)
python run_score_analysis.py --dataset gardenspoint_mini --descriptor alexnet

# Low-quality descriptor (expected: higher divergence)
python run_score_analysis.py --dataset gardenspoint_mini --descriptor sad
```

If separation decreases with weaker descriptors, and threshold divergence increases, the hypothesis is confirmed.

---

## Dataset Size Impact

Test with different numbers of runs:
```bash
# Few runs (high variance)
python multi_dataset_runner.py --dataset gardenspoint_mini --num-runs 10 --threshold-method original

# Many runs (low variance, better estimation)
python multi_dataset_runner.py --dataset gardenspoint_mini --num-runs 1000 --threshold-method original
```

**Expected**: Larger datasets (more runs) → better threshold estimation → more stable results, but **not** necessarily more divergence between methods.

---

## Quick Workflow

To run a complete analysis:

```bash
# Option 1: Run all analyses automatically
python run_all_threshold_analyses.py --dataset gardenspoint_mini --descriptor eigenplaces --num-runs 100

# Option 2: Run step-by-step manually
# 1. Run experiments with both methods
python multi_dataset_runner.py --dataset gardenspoint_mini --experiment-only --threshold-method legacy_mean_bad --descriptor eigenplaces --num-runs 100
python multi_dataset_runner.py --dataset gardenspoint_mini --experiment-only --threshold-method original --descriptor eigenplaces --num-runs 100

# 2. Compare thresholds (Analysis #2)
python run_threshold_comparison.py --dataset gardenspoint_mini

# 3. Visualize score distributions (Analysis #4)
python run_score_analysis.py --dataset gardenspoint_mini --descriptor eigenplaces

# 4. Review Analysis #1 output from step 1 (in terminal)
```

This will give you a comprehensive understanding of:
- Whether the additional term is negligible (Analysis #1)
- How much thresholds actually differ (Analysis #2)
- Why they differ or don't differ (Analysis #4)

---

## File Reference

- `threshold_analysis.py` - Core analysis functions
- `run_threshold_comparison.py` - Standalone script for Analysis #2
- `run_score_analysis.py` - Standalone script for Analysis #4
- `run_all_threshold_analyses.py` - Automated runner for all analyses
- `experiment_runner.py` - Automatically runs Analysis #1 during experiments
- `demo.py` - Can be modified to save S and GThard for offline analysis

---

## Adding More Analyses

To add a new analysis function:

1. Add the function to `threshold_analysis.py`
2. Import it where needed
3. Create a standalone script if appropriate
4. Document it in this README

Example structure:
```python
def my_new_analysis(data, **kwargs):
    """
    Analysis #N: Brief description
    
    Args:
        data: Input data structure
        **kwargs: Optional parameters
    
    Returns:
        Analysis results or None
    """
    print("\n" + "="*70)
    print("ANALYSIS #N: My New Analysis")
    print("="*70)
    
    # Your analysis code here
    
    print("="*70 + "\n")
```

