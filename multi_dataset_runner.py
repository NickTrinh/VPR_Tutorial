import os
import csv
from typing import Dict, List
import argparse
from datetime import datetime
import time
import numpy as np
import pandas as pd

from config import DATASETS, ExperimentConfig, DEFAULT_EXPERIMENT
from experiment_runner import run_experiment_on_dataset
from test_runner import test_dataset, TestResults
from data_utils import validate_dataset_structure
from config import get_dataset_config, auto_detect_dataset_structure

class MultiDatasetRunner:
    """Run experiments and tests across multiple datasets"""
    
    def __init__(self, experiment_config: ExperimentConfig = None):
        if experiment_config is None:
            experiment_config = DEFAULT_EXPERIMENT
        
        self.experiment_config = experiment_config
        self.results_summary = []
    
    def list_available_datasets(self):
        """List all available datasets with their status"""
        print("Available datasets:")
        print("-" * 50)
        
        for name, config in DATASETS.items():
            try:
                config = auto_detect_dataset_structure(config)
                if validate_dataset_structure(config):
                    status = "[OK] Ready"
                    info = f"({config.num_places} places, {config.images_per_place} images/place)"
                else:
                    status = "[X] Invalid structure"
                    info = ""
            except Exception as e:
                status = f"[X] Error: {str(e)}"
                info = ""
            
            print(f"{name:15} | {status:15} | {config.description} {info}")
    
    def run_experiments_on_datasets(self, dataset_names: List[str], use_cache: bool = True):
        """Run experiments on multiple datasets"""
        print(f"Running experiments on {len(dataset_names)} datasets")
        print("=" * 60)
        
        for dataset_name in dataset_names:
            try:
                print(f"\n{'='*20} Starting {dataset_name} {'='*20}")
                
                # Run experiment
                image_averages, place_averages = run_experiment_on_dataset(
                    dataset_name, self.experiment_config, use_cache
                )
                
                print(f"[✓] Experiment completed for {dataset_name}")
                
            except Exception as e:
                print(f"[X] Error running experiment on {dataset_name}: {str(e)}")
                continue
    
    def test_datasets(self, dataset_names: List[str], random_state: int = None, use_cache: bool = True):
        """Test multiple datasets and collect results"""
        print(f"Testing {len(dataset_names)} datasets")
        print("=" * 60)
        
        for dataset_name in dataset_names:
            try:
                print(f"\n{'='*20} Testing {dataset_name} {'='*20}")
                
                # Run test
                place_results, image_results = test_dataset(dataset_name, random_state, use_cache)
                
                # Store results for summary
                dataset_config = get_dataset_config(dataset_name)
                self.results_summary.append({
                    'dataset': dataset_name,
                    'dataset_name': dataset_config.name,
                    'place_results': place_results,
                    'image_results': image_results,
                    'random_state': random_state,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                print(f"[✓] Test completed for {dataset_name}")
                
            except Exception as e:
                print(f"[X] Error testing {dataset_name}: {str(e)}")
                continue
    
    def run_full_pipeline_on_datasets(self, dataset_names: List[str], random_state: int = None, use_cache: bool = True):
        """Run both experiments and tests on multiple datasets"""
        print(f"Running full pipeline on {len(dataset_names)} datasets")
        print("=" * 60)
        
        # First run experiments to generate thresholds
        print("\nPhase 1: Running experiments to generate thresholds")
        self.run_experiments_on_datasets(dataset_names, use_cache)
        
        # Then run tests using the generated thresholds
        print("\nPhase 2: Running tests using generated thresholds")
        self.test_datasets(dataset_names, random_state, use_cache)
        
        # Generate comparison report
        self.save_comparison_report()
    
    def save_comparison_report(self):
        """Save a comparison report of all tested datasets"""
        if not self.results_summary:
            print("No results to summarize")
            return
        
        # Create comparison directory
        comparison_dir = os.path.join(self.experiment_config.output_dir, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Save detailed comparison
        filename = os.path.join(comparison_dir, "dataset_comparison.csv")
        
        # Check if file exists to determine if we should write header
        file_exists = os.path.exists(filename)
        
        with open(filename, 'a', newline='') as file:  # 'a' for append mode
            writer = csv.writer(file)
            
            # Write header only if file doesn't exist
            if not file_exists:
                writer.writerow([
                    'Timestamp', 'Dataset', 'Test Type', 'Random State', 'TP', 'FP', 'TN', 'FN', 
                    'Precision', 'Recall', 'Accuracy', 'F1 Score'
                ])
            
            # Write results for each dataset
            for result in self.results_summary:
                dataset = result['dataset_name']
                timestamp = result['timestamp']
                random_state = result['random_state'] if result['random_state'] is not None else "random"
                
                # Place-level results
                pr = result['place_results']
                writer.writerow([
                    timestamp, dataset, 'Place-level', random_state, pr.TP, pr.FP, pr.TN, pr.FN,
                    f"{pr.precision:.4f}", f"{pr.recall:.4f}", 
                    f"{pr.accuracy:.4f}", f"{pr.f1_score:.4f}"
                ])
                
                # Image-level results
                ir = result['image_results']
                writer.writerow([
                    timestamp, dataset, 'Image-level', random_state, ir.TP, ir.FP, ir.TN, ir.FN,
                    f"{ir.precision:.4f}", f"{ir.recall:.4f}", 
                    f"{ir.accuracy:.4f}", f"{ir.f1_score:.4f}"
                ])
        
        print(f"\nComparison report appended to: {filename}")
        
        # Print summary to console
        self.print_comparison_summary()
    
    def print_comparison_summary(self):
        """Print a summary comparison of all datasets"""
        print("\n" + "="*80)
        print("DATASET COMPARISON SUMMARY")
        print("="*80)
        
        print(f"{'Dataset':<15} {'Type':<12} {'Precision':<10} {'Recall':<10} {'Accuracy':<10} {'F1 Score':<10}")
        print("-" * 80)
        
        for result in self.results_summary:
            dataset = result['dataset_name']
            
            # Place-level results
            pr = result['place_results']
            print(f"{dataset:<15} {'Place':<12} {pr.precision:<10.4f} {pr.recall:<10.4f} {pr.accuracy:<10.4f} {pr.f1_score:<10.4f}")
            
            # Image-level results
            ir = result['image_results']
            print(f"{'':15} {'Image':<12} {ir.precision:<10.4f} {ir.recall:<10.4f} {ir.accuracy:<10.4f} {ir.f1_score:<10.4f}")
            print()

def main():
    parser = argparse.ArgumentParser(description='Run VPR experiments on multiple datasets')
    parser.add_argument('--dataset', nargs='+', 
                       help='Dataset names to process (e.g., fordham_places st_lucia)')
    parser.add_argument('--list', action='store_true', 
                       help='List available datasets')
    parser.add_argument('--experiment-only', action='store_true',
                       help='Run experiments only (generate thresholds)')
    parser.add_argument('--test-only', action='store_true',
                       help='Run tests only (requires existing thresholds)')
    parser.add_argument('--num-runs', type=int, default=30,
                       help='Number of experiment runs (default: 30)')
    parser.add_argument('--random-state', type=int, default=None,
                       help='Random state for reproducibility (default: None for random)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable descriptor caching (slower but saves disk space)')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear cached descriptors before running')
    parser.add_argument('--threshold-multiplier', type=float, default=1.0,
                       help='Multiplier for threshold adjustment (< 1.0 = more lenient, default: 1.0)')
    parser.add_argument('--threshold-method', type=str, default="original",
                               choices=["original"],
                       help='Threshold calculation method (default: original)')
    
    args = parser.parse_args()
    
    # Determine cache usage
    use_cache = not args.no_cache
    
    # Create experiment configuration
    experiment_config = ExperimentConfig(
        num_runs=args.num_runs,
        random_seed=args.random_state if args.random_state is not None else 42,
        threshold_multiplier=args.threshold_multiplier,
        threshold_method=args.threshold_method
    )
    
    runner = MultiDatasetRunner(experiment_config)
    
    if args.list:
        runner.list_available_datasets()
        return
    
    if not args.dataset:
        print("No datasets specified. Use --list to see available datasets.")
        return
    
    # Clear cache if requested
    if args.clear_cache:
        from data_utils import DatasetLoader
        for dataset_name in args.dataset:
            try:
                dataset_config = get_dataset_config(dataset_name)
                loader = DatasetLoader(dataset_config, use_cache=True)
                loader.clear_cache()
            except Exception as e:
                print(f"Error clearing cache for {dataset_name}: {e}")
    
    # Validate dataset names
    valid_datasets = []
    for dataset_name in args.dataset:
        if dataset_name in DATASETS:
            valid_datasets.append(dataset_name)
        else:
            print(f"Warning: Unknown dataset '{dataset_name}'. Use --list to see available datasets.")
    
    if not valid_datasets:
        print("No valid datasets specified.")
        return
    
    print(f"Processing datasets: {valid_datasets}")
    print(f"Caching enabled: {use_cache}")
    print(f"Random state: {args.random_state if args.random_state is not None else 'random'}")
    
    if args.experiment_only:
        runner.run_experiments_on_datasets(valid_datasets, use_cache)
    elif args.test_only:
        runner.test_datasets(valid_datasets, args.random_state, use_cache)
        runner.save_comparison_report()
    else:
        # Run full pipeline
        runner.run_full_pipeline_on_datasets(valid_datasets, args.random_state, use_cache)

if __name__ == "__main__":
    main() 