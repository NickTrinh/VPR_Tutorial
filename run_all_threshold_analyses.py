"""
Comprehensive Threshold Analysis Runner

This script runs all threshold analyses in sequence to investigate why
the two threshold calculation methods may not diverge significantly.

Usage:
    python run_all_threshold_analyses.py --dataset gardenspoint_mini --descriptor eigenplaces --num-runs 100
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime


def run_command(cmd, description):
    """Run a command and print status"""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print(f"Command: {cmd}")
    print("="*80)
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with return code {result.returncode}")
        return False
    else:
        print(f"\n[SUCCESS] {description} completed")
        return True


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive threshold analysis')
    parser.add_argument('--dataset', type=str, default='gardenspoint_mini',
                       choices=['gardenspoint_mini', 'sfu_mini', 'nordland_mini', 
                               'nordland_mini_2', 'nordland_mini_3'],
                       help='Dataset to analyze (default: gardenspoint_mini)')
    parser.add_argument('--descriptor', type=str, default='eigenplaces',
                       choices=['eigenplaces', 'cosplace', 'alexnet', 'sad', 'hdc-delf'],
                       help='Descriptor to use (default: eigenplaces)')
    parser.add_argument('--num-runs', type=int, default=100,
                       help='Number of experiment runs (default: 100)')
    parser.add_argument('--skip-experiments', action='store_true',
                       help='Skip running experiments (use existing results)')
    parser.add_argument('--skip-score-analysis', action='store_true',
                       help='Skip score distribution analysis (can be slow)')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE THRESHOLD ANALYSIS")
    print("="*80)
    print(f"Dataset:     {args.dataset}")
    print(f"Descriptor:  {args.descriptor}")
    print(f"Num runs:    {args.num_runs}")
    print(f"Start time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    success_count = 0
    total_steps = 4
    
    # Step 1: Run experiments with LEGACY method
    if not args.skip_experiments:
        # Convert gardenspoint_mini to gardens_point_mini for multi_dataset_runner
        dataset_key = args.dataset.replace('gardenspoint', 'gardens_point')
        cmd = (f"python multi_dataset_runner.py "
               f"--dataset {dataset_key} "
               f"--experiment-only "
               f"--descriptor {args.descriptor} "
               f"--threshold-method legacy_mean_bad "
               f"--num-runs {args.num_runs}")
        
        if run_command(cmd, "Step 1: Experiment with LEGACY threshold method (includes Analysis #1)"):
            success_count += 1
        else:
            print("\n[ABORT] Failed at Step 1. Stopping analysis.")
            sys.exit(1)
    else:
        print("\n[SKIPPED] Step 1: Experiments (using existing results)")
        success_count += 1
    
    # Step 2: Run experiments with ORIGINAL method
    if not args.skip_experiments:
        # Convert gardenspoint_mini to gardens_point_mini for multi_dataset_runner
        dataset_key = args.dataset.replace('gardenspoint', 'gardens_point')
        cmd = (f"python multi_dataset_runner.py "
               f"--dataset {dataset_key} "
               f"--experiment-only "
               f"--descriptor {args.descriptor} "
               f"--threshold-method original "
               f"--num-runs {args.num_runs}")
        
        if run_command(cmd, "Step 2: Experiment with ORIGINAL threshold method (includes Analysis #1)"):
            success_count += 1
        else:
            print("\n[ABORT] Failed at Step 2. Stopping analysis.")
            sys.exit(1)
    else:
        print("\n[SKIPPED] Step 2: Experiments (using existing results)")
        success_count += 1
    
    # Step 3: Compare thresholds (Analysis #2) using descriptor-specific paths
    dataset_map_cmp = {
        'gardenspoint_mini': 'GardensPoint_Mini',
        'sfu_mini': 'SFU_Mini',
        'nordland_mini': 'Nordland_Mini',
        'nordland_mini_2': 'Nordland_Mini_2',
        'nordland_mini_3': 'Nordland_Mini_3'
    }
    dataset_dir_cmp = dataset_map_cmp.get(args.dataset, args.dataset)
    legacy_dir = os.path.join('results', f'{dataset_dir_cmp}_legacy', args.descriptor)
    original_dir = os.path.join('results', dataset_dir_cmp, args.descriptor)
    output_plot = f'threshold_comparison_{args.dataset}_{args.descriptor}.png'
    cmd = (
        f"python run_threshold_comparison.py "
        f"--dataset {args.dataset} "
        f"--legacy-dir {legacy_dir} "
        f"--original-dir {original_dir} "
        f"--output-plot {output_plot}"
    )

    if run_command(cmd, "Step 3: Analysis #2 - Side-by-Side Threshold Comparison"):
        success_count += 1
    else:
        print("\n[WARNING] Step 3 failed, but continuing...")
    
    # Step 4: Analyze score distributions (Analysis #4)
    if not args.skip_score_analysis:
        cmd = (f"python run_score_analysis.py "
               f"--dataset {args.dataset} "
               f"--descriptor {args.descriptor} "
               f"--num-queries 16")
        
        if run_command(cmd, "Step 4: Analysis #4 - Score Distribution Visualization"):
            success_count += 1
        else:
            print("\n[WARNING] Step 4 failed, but continuing...")
    else:
        print("\n[SKIPPED] Step 4: Score distribution analysis")
        success_count += 1
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Successful steps: {success_count}/{total_steps}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List generated files
    print("\n" + "="*80)
    print("GENERATED FILES:")
    print("="*80)
    
    # Map dataset key to result directory name
    dataset_map = {
        'gardenspoint_mini': 'GardensPoint_Mini',
        'sfu_mini': 'SFU_Mini',
        'nordland_mini': 'Nordland_Mini',
        'nordland_mini_2': 'Nordland_Mini_2',
        'nordland_mini_3': 'Nordland_Mini_3'
    }
    dataset_dir = dataset_map.get(args.dataset, args.dataset)
    
    files_to_check = [
        (f"results/{dataset_dir}_legacy/{args.descriptor}/place_averages.csv", "Legacy thresholds"),
        (f"results/{dataset_dir}/{args.descriptor}/place_averages.csv", "Original thresholds"),
        (f"threshold_comparison_{args.dataset}_{args.descriptor}.png", "Threshold comparison plot"),
        (f"threshold_comparison_{args.dataset}.csv", "Threshold comparison data"),
        (f"score_distributions_{args.dataset}_{args.descriptor}.png", "Score distribution plot"),
    ]
    
    for filepath, description in files_to_check:
        if os.path.exists(filepath):
            print(f"  ✓ {filepath:60} ({description})")
        else:
            print(f"  ✗ {filepath:60} (NOT FOUND)")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review Analysis #1 output in the terminal logs above")
    print(f"2. Open threshold_comparison_{args.dataset}.png to see threshold differences")
    print(f"3. Open score_distributions_{args.dataset}_{args.descriptor}.png to see separation")
    print(f"4. Review threshold_comparison_{args.dataset}.csv for detailed comparison")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

