"""
Script to run Analysis #2: Compare threshold values from legacy vs original methods

Usage:
    python run_threshold_comparison.py --dataset gardenspoint_mini
    python run_threshold_comparison.py --dataset nordland_mini --legacy-dir results/Nordland_Mini_legacy
"""

import argparse
import os
from threshold_analysis import compare_threshold_files


def main():
    parser = argparse.ArgumentParser(description='Compare threshold calculation methods')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., gardenspoint_mini, nordland_mini)')
    parser.add_argument('--legacy-dir', type=str, default=None,
                       help='Directory containing legacy results (default: results/<Dataset>_legacy)')
    parser.add_argument('--original-dir', type=str, default=None,
                       help='Directory containing original results (default: results/<Dataset>)')
    parser.add_argument('--output-plot', type=str, default=None,
                       help='Path to save comparison plot (default: threshold_comparison_<dataset>.png)')
    args = parser.parse_args()
    
    # Map dataset key to result directory name
    dataset_map = {
        'gardenspoint_mini': 'GardensPoint_Mini',
        'sfu_mini': 'SFU_Mini',
        'nordland_mini': 'Nordland_Mini',
        'nordland_mini_2': 'Nordland_Mini_2',
        'nordland_mini_3': 'Nordland_Mini_3'
    }
    
    dataset_dir = dataset_map.get(args.dataset, args.dataset)
    
    # Determine paths
    if args.legacy_dir:
        legacy_dir = args.legacy_dir
    else:
        legacy_dir = os.path.join('results', f'{dataset_dir}_legacy')
    
    if args.original_dir:
        original_dir = args.original_dir
    else:
        original_dir = os.path.join('results', dataset_dir)
    
    legacy_path = os.path.join(legacy_dir, 'place_averages.csv')
    original_path = os.path.join(original_dir, 'place_averages.csv')
    
    if args.output_plot:
        output_plot = args.output_plot
    else:
        output_plot = f'threshold_comparison_{args.dataset}.png'
    
    print(f"\nComparing threshold files:")
    print(f"  Legacy:   {legacy_path}")
    print(f"  Original: {original_path}")
    print(f"  Output:   {output_plot}\n")
    
    # Run comparison
    merged_df = compare_threshold_files(legacy_path, original_path, output_plot)
    
    if merged_df is not None:
        # Save merged comparison to CSV
        output_csv = f'threshold_comparison_{args.dataset}.csv'
        merged_df.to_csv(output_csv, index=False)
        print(f"Detailed comparison saved to: {output_csv}")


if __name__ == '__main__':
    main()

