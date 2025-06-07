#!/usr/bin/env python3
"""
Script to compare different threshold calculation methods
"""

import subprocess
import csv
import os
from typing import Dict, List
import argparse

def run_experiment_with_method(dataset: str, method: str, num_runs: int = 10) -> Dict:
    """Run experiment with specific threshold method"""
    print(f"Testing method: {method}")
    
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
        print(f"Error running experiment with {method}: {result.stderr}")
        return None
    
    # Run test
    cmd_test = [
        "python", "multi_dataset_runner.py",
        "--datasets", dataset,
        "--test-only"
    ]
    
    result = subprocess.run(cmd_test, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running test with {method}: {result.stderr}")
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

def compare_threshold_methods(dataset: str, num_runs: int = 10):
    """Compare all threshold methods on a dataset"""
    methods = [
        "original",
        "optimal_f1", 
        "youden_j",
        "cost_sensitive",
        "gaussian_intersection",
        "quantile_90",
        "quantile_80",
        "otsu_adapted",
        "ensemble",
        "precision_80",
        "precision_90", 
        "precision_95",
        "conservative_f1",
        "max_fpr_5",
        "mean_plus_2std",
        "mean_plus_3std"
    ]
    
    results = []
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Testing threshold method: {method.upper()}")
        print(f"{'='*50}")
        
        result = run_experiment_with_method(dataset, method, num_runs)
        if result:
            results.append(result)
            
            # Print immediate results
            place_f1 = result['place_results']['F1 Score']
            image_f1 = result['image_results']['F1 Score']
            print(f"[OK] {method}: Place F1={place_f1:.4f}, Image F1={image_f1:.4f}")
        else:
            print(f"[X] {method}: Failed")
    
    # Save comparison results
    comparison_file = f"results/{dataset.title().replace('_', '')}/threshold_method_comparison.csv"
    os.makedirs(os.path.dirname(comparison_file), exist_ok=True)
    
    with open(comparison_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Method', 'Level', 'TP', 'FP', 'TN', 'FN', 
            'Precision', 'Recall', 'Accuracy', 'F1 Score'
        ])
        
        for result in results:
            method = result['method']
            
            # Place-level
            pr = result['place_results']
            writer.writerow([
                method, 'Place', int(pr['TP']), int(pr['FP']), int(pr['TN']), int(pr['FN']),
                f"{pr['Precision']:.4f}", f"{pr['Recall']:.4f}", 
                f"{pr['Accuracy']:.4f}", f"{pr['F1 Score']:.4f}"
            ])
            
            # Image-level  
            ir = result['image_results']
            writer.writerow([
                method, 'Image', int(ir['TP']), int(ir['FP']), int(ir['TN']), int(ir['FN']),
                f"{ir['Precision']:.4f}", f"{ir['Recall']:.4f}", 
                f"{ir['Accuracy']:.4f}", f"{ir['F1 Score']:.4f}"
            ])
    
    print(f"\n{'='*60}")
    print("THRESHOLD METHOD COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'Level':<8} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 60)
    
    # Sort by F1 score
    results.sort(key=lambda x: x['image_results']['F1 Score'], reverse=True)
    
    for result in results:
        method = result['method']
        
        # Place-level
        pr = result['place_results'] 
        print(f"{method:<20} {'Place':<8} {pr['Precision']:<10.4f} {pr['Recall']:<10.4f} {pr['F1 Score']:<10.4f}")
        
        # Image-level
        ir = result['image_results']
        print(f"{'':<20} {'Image':<8} {ir['Precision']:<10.4f} {ir['Recall']:<10.4f} {ir['F1 Score']:<10.4f}")
        print()
    
    print(f"Detailed results saved to: {comparison_file}")
    
    # Find best method
    best_method = results[0]['method']
    best_f1 = results[0]['image_results']['F1 Score']
    print(f"\n[WINNER] Best method: {best_method.upper()} (Image F1: {best_f1:.4f})")

def main():
    parser = argparse.ArgumentParser(description='Compare threshold calculation methods')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset to test (e.g., matching_triplets, fordham_places)')
    parser.add_argument('--num-runs', type=int, default=10,
                       help='Number of experiment runs per method (default: 10)')
    
    args = parser.parse_args()
    
    print(f"Comparing threshold methods on {args.dataset}")
    print(f"Using {args.num_runs} runs per method")
    
    compare_threshold_methods(args.dataset, args.num_runs)

if __name__ == "__main__":
    main() 