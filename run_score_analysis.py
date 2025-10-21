"""
Script to run Analysis #4: Visualize good vs bad score distributions

This script runs demo.py functionality but focuses on analyzing the similarity matrix
to understand why the two threshold methods may not diverge significantly.

Usage:
    python run_score_analysis.py --dataset gardenspoint_mini --descriptor eigenplaces
"""

import argparse
import numpy as np
from datasets.load_dataset import PlaceConditionsDataset
from threshold_analysis import analyze_score_distributions


def main():
    parser = argparse.ArgumentParser(description='Analyze score distributions for VPR')
    parser.add_argument('--dataset', type=lambda s: s.lower(), default='gardenspoint_mini',
                       choices=['gardenspoint_mini', 'sfu_mini', 'nordland_mini', 
                               'nordland_mini_2', 'nordland_mini_3'],
                       help='Dataset to analyze')
    parser.add_argument('--descriptor', type=lambda s: s.lower(), default='eigenplaces',
                       choices=['eigenplaces', 'cosplace', 'alexnet', 'sad'],
                       help='Descriptor to use')
    parser.add_argument('--output', type=str, default=None,
                       help='Output plot filename (default: score_distributions_<dataset>_<descriptor>.png)')
    parser.add_argument('--num-queries', type=int, default=16,
                       help='Number of query distributions to plot (default: 16)')
    args = parser.parse_args()
    
    print(f'Loading dataset: {args.dataset}')
    
    # Load dataset
    if args.dataset == 'gardenspoint_mini':
        from datasets.load_dataset import GardensPointDataset
        dataset = GardensPointDataset(destination='images/GardensPoint_Mini/')
    elif args.dataset == 'sfu_mini':
        dataset = PlaceConditionsDataset(destination='images/SFU_Mini/', 
                                        db_condition='dry', q_condition='jan')
    elif args.dataset == 'nordland_mini':
        dataset = PlaceConditionsDataset(destination='images/Nordland_Mini/', 
                                        db_condition='spring', q_condition='winter')
    elif args.dataset == 'nordland_mini_2':
        dataset = PlaceConditionsDataset(destination='images/Nordland_mini_2/', 
                                        db_condition='spring', q_condition='winter')
    elif args.dataset == 'nordland_mini_3':
        dataset = PlaceConditionsDataset(destination='images/Nordland_Mini_3/', 
                                        db_condition='spring', q_condition='winter')
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    
    imgs_db, imgs_q, GThard, GTsoft = dataset.load()
    print(f'Loaded {len(imgs_db)} database images and {len(imgs_q)} query images')
    
    # Load feature extractor
    print(f'Loading feature extractor: {args.descriptor}')
    if args.descriptor == 'eigenplaces':
        from feature_extraction.feature_extractor_eigenplaces import EigenPlacesFeatureExtractor
        feature_extractor = EigenPlacesFeatureExtractor()
    elif args.descriptor == 'cosplace':
        from feature_extraction.feature_extractor_cosplace import CosPlaceFeatureExtractor
        feature_extractor = CosPlaceFeatureExtractor()
    elif args.descriptor == 'alexnet':
        from feature_extraction.feature_extractor_holistic import AlexNetConv3Extractor
        feature_extractor = AlexNetConv3Extractor()
    elif args.descriptor == 'sad':
        from feature_extraction.feature_extractor_holistic import SAD
        feature_extractor = SAD()
    else:
        raise ValueError(f'Unknown descriptor: {args.descriptor}')
    
    # Extract features
    print('Extracting database features...')
    db_features = feature_extractor.compute_features(imgs_db)
    print('Extracting query features...')
    q_features = feature_extractor.compute_features(imgs_q)
    
    # Compute similarity matrix
    print('Computing similarity matrix...')
    if args.descriptor == 'sad':
        S = np.empty([len(imgs_db), len(imgs_q)], 'float32')
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                diff = db_features[i] - q_features[j]
                dim = len(db_features[0]) - np.sum(np.isnan(diff))
                diff[np.isnan(diff)] = 0
                S[i, j] = -np.sum(np.abs(diff)) / dim
    else:
        db_features = db_features / np.linalg.norm(db_features, axis=1, keepdims=True)
        q_features = q_features / np.linalg.norm(q_features, axis=1, keepdims=True)
        S = np.matmul(db_features, q_features.transpose())
    
    # Determine output filename
    if args.output:
        output_plot = args.output
    else:
        output_plot = f'score_distributions_{args.dataset}_{args.descriptor}.png'
    
    # Run analysis
    analyze_score_distributions(S, GThard, output_plot, num_queries_to_plot=args.num_queries)
    
    print(f'\nAnalysis complete!')


if __name__ == '__main__':
    main()

