# Refactored VPR Tutorial - Multi-Dataset Support

This refactored version provides a modular, configurable system for running Visual Place Recognition (VPR) experiments across multiple datasets.

## New Features

- **Multi-dataset Support**: Easily run experiments on different datasets
- **Configurable Parameters**: Centralized configuration management
- **Modular Architecture**: Clean separation of concerns
- **Automated Comparison**: Generate comparison reports across datasets
- **Flexible Testing**: Support for both place-level and image-level thresholds

## File Structure

```
├── config.py                 # Dataset and experiment configurations
├── data_utils.py             # Data loading and processing utilities
├── experiment_runner.py      # Main experiment execution
├── test_runner.py            # Testing with pre-calculated thresholds
├── multi_dataset_runner.py   # Multi-dataset experiment runner
├── places_data.py            # Legacy threshold data (for FordhamPlaces)
└── results/                  # Output directory (auto-created)
    ├── FordhamPlaces/        # Results for each dataset
    ├── StLuciaSmall/
    └── comparison/           # Cross-dataset comparison reports
```

## Quick Start

### 1. List Available Datasets
```bash
python multi_dataset_runner.py --list
```

### 2. Run Experiment on Single Dataset
```bash
# Run full pipeline (experiments + testing)
python multi_dataset_runner.py --datasets fordham_places

# Run just experiments (generate thresholds)
python multi_dataset_runner.py --datasets fordham_places --experiment-only

# Run just testing (requires existing thresholds)
python multi_dataset_runner.py --datasets fordham_places --test-only
```

### 3. Run Experiments on Multiple Datasets
```bash
python multi_dataset_runner.py --datasets fordham_places st_lucia gardens_point
```

### 4. Custom Experiment Parameters
```bash
python multi_dataset_runner.py --datasets fordham_places --num-runs 10 --random-state 123
```

## Individual Module Usage

### Running Experiments
```python
from experiment_runner import run_experiment_on_dataset
from config import ExperimentConfig

# Custom experiment configuration
config = ExperimentConfig(num_runs=10, random_seed=42)

# Run experiment
image_averages, place_averages = run_experiment_on_dataset("fordham_places", config)
```

### Running Tests
```python
from test_runner import test_dataset

# Test dataset (requires existing threshold files)
place_results, image_results = test_dataset("fordham_places")
```

### Working with Configurations
```python
from config import get_dataset_config, auto_detect_dataset_structure

# Get dataset configuration
dataset_config = get_dataset_config("fordham_places")

# Auto-detect dataset structure
dataset_config = auto_detect_dataset_structure(dataset_config)
```

## Adding New Datasets

### 1. Add Dataset Configuration
Edit `config.py` and add your dataset to the `DATASETS` dictionary:

```python
DATASETS = {
    # ... existing datasets ...
    "my_dataset": DatasetConfig(
        name="MyDataset",
        path="images/MyDataset/",
        num_places=0,  # 0 = auto-detect
        images_per_place=0,  # 0 = auto-detect
        description="My custom dataset"
    )
}
```

### 2. Expected Dataset Structure
Your dataset should follow this structure:
```
images/MyDataset/
├── p0/                 # Place 0
│   ├── i0/            # Image set 0
│   │   └── *.jpg      # Images
│   ├── i1/            # Image set 1
│   │   └── *.jpg
│   └── ...
├── p1/                 # Place 1
│   ├── i0/
│   └── ...
└── ...
```

### 3. Run Experiments
```bash
python multi_dataset_runner.py --datasets my_dataset
```

## Output Files

For each dataset, the following files are generated:

### Experiment Results (`results/{dataset_name}/`)
- `test_results_run_X.csv` - Individual run results
- `image_averages.csv` - Image-level averaged thresholds
- `place_averages.csv` - Place-level averaged thresholds

### Test Results
- `final_test_results.csv` - TP, FP, TN, FN and metrics

### Comparison Reports (`results/comparison/`)
- `dataset_comparison.csv` - Side-by-side comparison of all tested datasets

## Configuration Options

### Dataset Configuration
```python
@dataclass
class DatasetConfig:
    name: str                    # Display name
    path: str                    # Path to dataset
    num_places: int              # Number of places (0 = auto-detect)
    images_per_place: int        # Images per place (0 = auto-detect)
    image_extension: str         # File pattern (default: "*.jpg")
    description: str             # Description
```

### Experiment Configuration
```python
@dataclass
class ExperimentConfig:
    num_runs: int = 30           # Number of experiment runs
    random_seed: int = 42        # Random seed for reproducibility
    train_test_split: Tuple = (2, 1)  # (train, test) images per place
    output_dir: str = "results"  # Output directory
```

## Legacy Compatibility

The refactored code maintains backward compatibility:

- `places_data.py` - Still contains FordhamPlaces threshold data
- Original scripts still work but are deprecated
- Results format is unchanged

## Example Workflows

### Compare Methods Across Datasets
```bash
# Run experiments on multiple datasets
python multi_dataset_runner.py --datasets fordham_places st_lucia gardens_point

# Results will be in results/comparison/dataset_comparison.csv
```

### Development/Testing with Fewer Runs
```bash
# Quick test with 5 runs instead of 30
python multi_dataset_runner.py --datasets fordham_places --num-runs 5
```

### Reproduce Specific Results
```bash
# Use specific random seed for reproducibility
python multi_dataset_runner.py --datasets fordham_places --random-state 12345
```

## Migration from Legacy Code

### Old Way
```python
# Old: Hardcoded for FordhamPlaces
from get_n_scores import run_multiple_tests
run_multiple_tests()
```

### New Way
```python
# New: Configurable for any dataset
from experiment_runner import run_experiment_on_dataset
run_experiment_on_dataset("fordham_places")
```

## Error Handling

The refactored code includes comprehensive error handling:

- Dataset structure validation
- Missing file detection
- Graceful failure with error messages
- Automatic directory creation

## Performance Notes

- Feature extraction is cached per experiment run
- Results are saved incrementally
- Memory usage is optimized for large datasets
- Progress reporting for long-running experiments

## Future Extensions

The modular design makes it easy to add:

- New feature extraction methods
- Different similarity metrics
- Alternative threshold calculation methods
- Custom evaluation metrics
- Integration with machine learning frameworks (for triplet loss experiments) 