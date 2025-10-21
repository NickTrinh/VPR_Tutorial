"""
Threshold Analysis Tools for VPR Experiments

This module contains various analysis functions to investigate why the two threshold
calculation methods (original vs legacy) may not diverge significantly on large datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import os


def print_threshold_component_analysis(place_data: Dict, threshold_method: str = "original"):
    """
    Analysis #1: Print distribution statistics for filter_n and std_dev components
    
    Args:
        place_data: Dictionary mapping place IDs to their threshold component data
        threshold_method: Current threshold calculation method being used
    """
    print("\n" + "="*70)
    print("ANALYSIS #1: Threshold Component Distribution Analysis")
    print("="*70)
    
    all_filter_ns = []
    all_std_devs = []
    all_mean_bads = []
    
    for place, data in place_data.items():
        all_filter_ns.extend(data['filter_n'])
        all_std_devs.extend(data['std_dev_bad_scores'])
        all_mean_bads.extend(data['mean_bad_scores'])
    
    filter_ns_array = np.array(all_filter_ns)
    std_devs_array = np.array(all_std_devs)
    mean_bads_array = np.array(all_mean_bads)
    additional_term = filter_ns_array * std_devs_array
    
    print(f"\nTotal samples analyzed: {len(filter_ns_array)}")
    print(f"\nfilter_n statistics:")
    print(f"  Mean:   {filter_ns_array.mean():.4f}")
    print(f"  Std:    {filter_ns_array.std():.4f}")
    print(f"  Min:    {filter_ns_array.min():.4f}")
    print(f"  Max:    {filter_ns_array.max():.4f}")
    print(f"  Median: {np.median(filter_ns_array):.4f}")
    
    print(f"\nstd_dev (of bad scores) statistics:")
    print(f"  Mean:   {std_devs_array.mean():.6f}")
    print(f"  Std:    {std_devs_array.std():.6f}")
    print(f"  Min:    {std_devs_array.min():.6f}")
    print(f"  Max:    {std_devs_array.max():.6f}")
    print(f"  Median: {np.median(std_devs_array):.6f}")
    
    print(f"\nmean_bad statistics:")
    print(f"  Mean:   {mean_bads_array.mean():.6f}")
    print(f"  Std:    {mean_bads_array.std():.6f}")
    print(f"  Min:    {mean_bads_array.min():.6f}")
    print(f"  Max:    {mean_bads_array.max():.6f}")
    print(f"  Median: {np.median(mean_bads_array):.6f}")
    
    print(f"\nAdditional term (filter_n × std_dev) statistics:")
    print(f"  Mean:   {additional_term.mean():.6f}")
    print(f"  Std:    {additional_term.std():.6f}")
    print(f"  Min:    {additional_term.min():.6f}")
    print(f"  Max:    {additional_term.max():.6f}")
    print(f"  Median: {np.median(additional_term):.6f}")
    
    print(f"\nRelative impact of additional term:")
    relative_impact = (additional_term / mean_bads_array) * 100
    print(f"  Mean % of mean_bad:   {relative_impact.mean():.2f}%")
    print(f"  Median % of mean_bad: {np.median(relative_impact):.2f}%")
    print(f"  Max % of mean_bad:    {relative_impact.max():.2f}%")
    
    print(f"\nCurrent threshold method: {threshold_method}")
    if threshold_method == 'legacy_mean_bad':
        print("  → Using: threshold = mean_bad")
        print("  → Additional term (filter_n × std_dev) is IGNORED")
    else:
        print("  → Using: threshold = mean_bad + filter_n × std_dev")
        print(f"  → Additional term adds ~{relative_impact.mean():.2f}% on average to thresholds")
    
    print("="*70 + "\n")


def compare_threshold_files(legacy_path: str, original_path: str, output_plot: str = None):
    """
    Analysis #2: Compare threshold values from two different methods side-by-side
    
    Args:
        legacy_path: Path to place_averages.csv from legacy method
        original_path: Path to place_averages.csv from original method
        output_plot: Optional path to save comparison plot
    
    Returns:
        DataFrame with merged comparison data
    """
    print("\n" + "="*70)
    print("ANALYSIS #2: Side-by-Side Threshold Comparison")
    print("="*70)
    
    if not os.path.exists(legacy_path):
        print(f"ERROR: Legacy threshold file not found at {legacy_path}")
        return None
    
    if not os.path.exists(original_path):
        print(f"ERROR: Original threshold file not found at {original_path}")
        return None
    
    legacy = pd.read_csv(legacy_path)
    original = pd.read_csv(original_path)
    
    # Normalize column names
    legacy.columns = [c.lower() for c in legacy.columns]
    original.columns = [c.lower() for c in original.columns]
    
    # Merge on place
    merged = legacy.merge(original, on='place', suffixes=('_legacy', '_original'))
    
    # Calculate differences
    merged['simple_threshold_diff'] = merged['simple_avg_threshold_original'] - merged['simple_avg_threshold_legacy']
    merged['simple_percent_diff'] = 100 * merged['simple_threshold_diff'] / merged['simple_avg_threshold_legacy']
    
    merged['weighted_threshold_diff'] = merged['weighted_avg_threshold_original'] - merged['weighted_avg_threshold_legacy']
    merged['weighted_percent_diff'] = 100 * merged['weighted_threshold_diff'] / merged['weighted_avg_threshold_legacy']
    
    print(f"\nComparing {len(merged)} places")
    print(f"\nSimple Average Threshold Differences:")
    print(f"  Mean difference:        {merged['simple_threshold_diff'].mean():.6f}")
    print(f"  Std of differences:     {merged['simple_threshold_diff'].std():.6f}")
    print(f"  Max difference:         {merged['simple_threshold_diff'].max():.6f}")
    print(f"  Min difference:         {merged['simple_threshold_diff'].min():.6f}")
    print(f"  Mean % difference:      {merged['simple_percent_diff'].mean():.2f}%")
    print(f"  Max % difference:       {merged['simple_percent_diff'].max():.2f}%")
    
    print(f"\nWeighted Average Threshold Differences:")
    print(f"  Mean difference:        {merged['weighted_threshold_diff'].mean():.6f}")
    print(f"  Std of differences:     {merged['weighted_threshold_diff'].std():.6f}")
    print(f"  Max difference:         {merged['weighted_threshold_diff'].max():.6f}")
    print(f"  Min difference:         {merged['weighted_threshold_diff'].min():.6f}")
    print(f"  Mean % difference:      {merged['weighted_percent_diff'].mean():.2f}%")
    print(f"  Max % difference:       {merged['weighted_percent_diff'].max():.2f}%")
    
    # Create visualization
    if output_plot or True:  # Always create plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Scatter comparison - Simple
        ax = axes[0, 0]
        ax.scatter(merged['simple_avg_threshold_legacy'], 
                  merged['simple_avg_threshold_original'], 
                  alpha=0.6, s=50)
        lims = [
            min(merged['simple_avg_threshold_legacy'].min(), merged['simple_avg_threshold_original'].min()),
            max(merged['simple_avg_threshold_legacy'].max(), merged['simple_avg_threshold_original'].max())
        ]
        ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=2, label='y=x (no difference)')
        ax.set_xlabel('Legacy (mean_bad only)', fontsize=11)
        ax.set_ylabel('Original (mean_bad + filter_n × std)', fontsize=11)
        ax.set_title('Simple Average Threshold Comparison', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Histogram of % differences - Simple
        ax = axes[0, 1]
        ax.hist(merged['simple_percent_diff'], bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(merged['simple_percent_diff'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {merged['simple_percent_diff'].mean():.2f}%")
        ax.set_xlabel('% Difference', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Distribution of % Differences (Simple)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Scatter comparison - Weighted
        ax = axes[1, 0]
        ax.scatter(merged['weighted_avg_threshold_legacy'], 
                  merged['weighted_avg_threshold_original'], 
                  alpha=0.6, s=50, color='green')
        lims = [
            min(merged['weighted_avg_threshold_legacy'].min(), merged['weighted_avg_threshold_original'].min()),
            max(merged['weighted_avg_threshold_legacy'].max(), merged['weighted_avg_threshold_original'].max())
        ]
        ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=2, label='y=x (no difference)')
        ax.set_xlabel('Legacy (mean_bad only)', fontsize=11)
        ax.set_ylabel('Original (mean_bad + filter_n × std)', fontsize=11)
        ax.set_title('Weighted Average Threshold Comparison', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Histogram of % differences - Weighted
        ax = axes[1, 1]
        ax.hist(merged['weighted_percent_diff'], bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.axvline(merged['weighted_percent_diff'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {merged['weighted_percent_diff'].mean():.2f}%")
        ax.set_xlabel('% Difference', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Distribution of % Differences (Weighted)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_plot:
            plt.savefig(output_plot, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {output_plot}")
        else:
            default_path = 'threshold_comparison.png'
            plt.savefig(default_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {default_path}")
        
        plt.close()
    
    print("="*70 + "\n")
    return merged


def analyze_score_distributions(S: np.ndarray, GThard: np.ndarray, 
                                output_plot: str = 'score_distributions.png',
                                num_queries_to_plot: int = 16):
    """
    Analysis #4: Visualize good vs bad score distributions to understand separation
    
    Args:
        S: Similarity matrix (database × query)
        GThard: Ground truth hard matches
        output_plot: Path to save the plot
        num_queries_to_plot: Number of query distributions to visualize
    """
    print("\n" + "="*70)
    print("ANALYSIS #4: Score Distribution Visualization")
    print("="*70)
    
    num_queries = min(num_queries_to_plot, S.shape[1])
    grid_size = int(np.ceil(np.sqrt(num_queries)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
    axes = axes.flatten() if num_queries > 1 else [axes]
    
    for q_idx in range(num_queries):
        ax = axes[q_idx]
        
        good_scores = S[GThard[:, q_idx], q_idx]
        bad_scores = S[~GThard[:, q_idx], q_idx]
        
        if len(bad_scores) > 0:
            ax.hist(bad_scores, bins=50, alpha=0.5, label=f'Bad (n={len(bad_scores)})', color='red')
        if len(good_scores) > 0:
            ax.hist(good_scores, bins=20, alpha=0.7, label=f'Good (n={len(good_scores)})', color='green')
        
        if len(bad_scores) > 0:
            mean_bad = np.mean(bad_scores)
            std_bad = np.std(bad_scores)
            
            ax.axvline(mean_bad, color='red', linestyle='--', linewidth=1.5, label='mean_bad')
            ax.axvline(mean_bad + 2*std_bad, color='orange', linestyle='--', 
                      linewidth=1.5, label='mean_bad + 2σ')
        
        ax.set_title(f'Query {q_idx}', fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(num_queries, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"\nScore distribution plot saved to: {output_plot}")
    plt.close()
    
    # Print separation statistics
    print(f"\nScore Separation Analysis (across {S.shape[1]} queries):")
    all_good_scores = []
    all_bad_scores = []
    separations = []
    
    for q_idx in range(S.shape[1]):
        good_scores = S[GThard[:, q_idx], q_idx]
        bad_scores = S[~GThard[:, q_idx], q_idx]
        
        if len(good_scores) > 0 and len(bad_scores) > 0:
            all_good_scores.extend(good_scores)
            all_bad_scores.extend(bad_scores)
            
            mean_good = np.mean(good_scores)
            mean_bad = np.mean(bad_scores)
            std_bad = np.std(bad_scores)
            
            # Separation in standard deviations
            separation = (mean_good - mean_bad) / (std_bad + 1e-9)
            separations.append(separation)
    
    all_good_scores = np.array(all_good_scores)
    all_bad_scores = np.array(all_bad_scores)
    separations = np.array(separations)
    
    print(f"\nGood scores: mean={all_good_scores.mean():.4f}, std={all_good_scores.std():.4f}")
    print(f"Bad scores:  mean={all_bad_scores.mean():.4f}, std={all_bad_scores.std():.4f}")
    print(f"\nSeparation (in std devs of bad scores):")
    print(f"  Mean:   {separations.mean():.2f}σ")
    print(f"  Median: {np.median(separations):.2f}σ")
    print(f"  Min:    {separations.min():.2f}σ")
    print(f"  Max:    {separations.max():.2f}σ")
    
    if separations.mean() > 5:
        print("\n  → HIGH separation: descriptor produces well-separated distributions")
        print("     This may explain why filter_n × std_dev has minimal impact!")
    elif separations.mean() > 2:
        print("\n  → MODERATE separation: some overlap between good and bad scores")
    else:
        print("\n  → LOW separation: significant overlap, challenging retrieval task")
    
    print("="*70 + "\n")


def save_detailed_run_stats(run_idx: int, place_id: str, good_scores: np.ndarray, 
                           bad_scores: np.ndarray, threshold: float, filter_n: float,
                           output_dir: str):
    """
    Analysis #3: Save detailed per-run statistics for later inspection
    
    Args:
        run_idx: Run number
        place_id: Place identifier
        good_scores: Array of similarity scores for positive matches
        bad_scores: Array of similarity scores for negative matches
        threshold: Calculated threshold for this run
        filter_n: Filter_n value for this run
        output_dir: Directory to save results
    """
    out_file = os.path.join(output_dir, f'run_details_place_{place_id}.csv')
    file_exists = os.path.exists(out_file)
    
    with open(out_file, 'a', newline='') as f:
        import csv
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['run', 'num_good', 'num_bad', 'mean_good', 'mean_bad', 
                           'std_good', 'std_bad', 'std_dev', 'filter_n', 
                           'threshold_legacy', 'threshold_original'])
        
        mean_good = np.mean(good_scores) if len(good_scores) > 0 else 0
        mean_bad = np.mean(bad_scores) if len(bad_scores) > 0 else 0
        std_good = np.std(good_scores) if len(good_scores) > 0 else 0
        std_bad = np.std(bad_scores) if len(bad_scores) > 0 else 0
        std_dev = np.std(bad_scores) if len(bad_scores) > 1 else 0
        
        threshold_legacy = mean_bad
        threshold_original = mean_bad + filter_n * std_dev
        
        writer.writerow([run_idx, len(good_scores), len(bad_scores), mean_good, mean_bad,
                        std_good, std_bad, std_dev, filter_n, threshold_legacy, threshold_original])

