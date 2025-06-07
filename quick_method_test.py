#!/usr/bin/env python3
"""
Quick test script for comparing specific threshold method combinations
"""

import subprocess
import csv
import os
import argparse
from typing import List, Dict

def run_method_test(dataset: str, method: str, num_runs: int = 5) -> Dict:
    """Run a quick test with specific method"""
    print(f"Testing {method}...")
    
    # Run experiment
    cmd_exp = [
        "python", "multi_dataset_runner.py",
        "--datasets", dataset,
        "--num-runs", str(num_runs),
        "--experiment-only",
        "--threshold-method", method
    ]
    
    result = subprocess.run(cmd_exp, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[X] Experiment failed for {method}")
        return None
    
    # Run test
    cmd_test = [
        "python", "multi_dataset_runner.py",
        "--datasets", dataset,
        "--test-only"
    ]
    
    result = subprocess.run(cmd_test, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[X] Test failed for {method}")
        return None
    
    # Read results
    results_file = f"results/{dataset.title().replace('_', '')}/final_test_results.csv"
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            reader = csv.DictReader(f)
            results = list(reader)
            
            return {
                'method': method,
                'place_results': {k: float(v) for k, v in results[0].items() if k != 'Test Type'},
                'image_results': {k: float(v) for k, v in results[1].items() if k != 'Test Type'}
            }
    
    return None

def compare_method_subset(dataset: str, methods: List[str], num_runs: int = 5):
    """Compare a specific subset of methods"""
    print(f"Quick comparison of {len(methods)} methods on {dataset}")
    print(f"Using {num_runs} runs per method for speed")
    print("=" * 60)
    
    results = []
    
    for method in methods:
        result = run_method_test(dataset, method, num_runs)
        if result:
            results.append(result)
            
            # Show immediate results
            pr = result['place_results']
            ir = result['image_results']
            print(f"[OK] {method:15} | Place: P={pr['Precision']:.3f} R={pr['Recall']:.3f} F1={pr['F1 Score']:.3f} | Image: P={ir['Precision']:.3f} R={ir['Recall']:.3f} F1={ir['F1 Score']:.3f}")
        else:
            print(f"[X] {method:15} | FAILED")
    
    print("\n" + "=" * 80)
    print("QUICK COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Method':<20} {'Level':<8} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'TP':<5} {'FP':<5}")
    print("-" * 80)
    
    # Sort by F1 score
    results.sort(key=lambda x: x['image_results']['F1 Score'], reverse=True)
    
    for result in results:
        method = result['method']
        
        # Place-level
        pr = result['place_results']
        print(f"{method:<20} {'Place':<8} {pr['Precision']:<10.3f} {pr['Recall']:<10.3f} {pr['F1 Score']:<10.3f} {int(pr['TP']):<5} {int(pr['FP']):<5}")
        
        # Image-level
        ir = result['image_results']
        print(f"{'':<20} {'Image':<8} {ir['Precision']:<10.3f} {ir['Recall']:<10.3f} {ir['F1 Score']:<10.3f} {int(ir['TP']):<5} {int(ir['FP']):<5}")
        print()
    
    if results:
        best = results[0]
        print(f"[WINNER] {best['method']} - Image F1: {best['image_results']['F1 Score']:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Quick test of specific threshold methods')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset to test')
    parser.add_argument('--methods', nargs='+', 
                       help='Specific methods to test')
    parser.add_argument('--preset', type=str, 
                       choices=['precision_focused', 'balanced', 'recall_focused', 'conservative', 'all_new'],
                       help='Use a preset combination of methods')
    parser.add_argument('--num-runs', type=int, default=5,
                       help='Number of runs per method (default: 5 for speed)')
    
    args = parser.parse_args()
    
    # Define preset method combinations
    presets = {
        'precision_focused': [
            'precision_80',
            'precision_90', 
            'precision_95',
            'conservative_f1',
            'max_fpr_5'
        ],
        'conservative': [
            'mean_plus_2std',
            'mean_plus_3std',
            'precision_95',
            'max_fpr_5'
        ],
        'balanced': [
            'optimal_f1',
            'youden_j',
            'conservative_f1',
            'precision_80'
        ],
        'recall_focused': [
            'quantile_90',
            'quantile_80',
            'cost_sensitive',
            'gaussian_intersection'
        ],
        'all_new': [
            'precision_80',
            'precision_90', 
            'precision_95',
            'conservative_f1',
            'max_fpr_5',
            'mean_plus_2std',
            'mean_plus_3std'
        ]
    }
    
    if args.preset:
        methods = presets[args.preset]
        print(f"Using preset: {args.preset}")
    elif args.methods:
        methods = args.methods
    else:
        print("Error: Must specify either --methods or --preset")
        print("\nAvailable presets:")
        for preset, method_list in presets.items():
            print(f"  {preset}: {', '.join(method_list)}")
        return
    
    compare_method_subset(args.dataset, methods, args.num_runs)

if __name__ == "__main__":
    main() 