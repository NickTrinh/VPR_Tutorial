#   =====================================================================
#   Copyright (C) 2023  Stefan Schubert, stefan.schubert@etit.tu-chemnitz.de
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#   =====================================================================
#

import argparse
import configparser
import os

from evaluation.metrics import recallAtK, calculate_recall_at_k_with_filtering, print_recall_at_k_results, save_recall_results_csv
from evaluation import show_correct_and_wrong_matches
from matching.matching import best_match_per_query, thresholding
from datasets.load_dataset import GardensPointDataset, StLuciaDataset, SFUDataset, Tokyo247Dataset, PlaceConditionsDataset
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from utils import normalize_l2


def get_place_ids_from_paths(paths):
    """Extracts place ID from filenames. Supports ImageNNN.jpg and Place####_*.jpg patterns."""
    place_ids = []
    for path in paths:
        filename = os.path.basename(path)
        # Prefer unified Place#### naming if present
        if filename.lower().startswith('place'):
            token = filename.split('_', 1)[0]  # Place####
            try:
                place_num = int(token[5:])
                place_ids.append(f"p{place_num}")
                continue
            except Exception:
                pass
        # Fallback to ImageNNN grouping
        try:
            img_num = int(filename.replace('Image', '').replace('.jpg', ''))
            place_id_num = img_num // 10
            place_ids.append(f"p{place_id_num}")
        except Exception:
            place_ids.append("p-1")
    return place_ids

def compute_features_chunked(feature_extractor, imgs, env_var='VPR_DEMO_FEAT_BATCH', default_chunk=128):
    """Compute features in chunks to avoid GPU OOM (especially for AlexNet).

    Respects environment variable env_var for initial chunk size and
    halves the chunk size on CUDA OOM until successful (min 1).
    """
    import os as _os
    import numpy as _np
    chunk_size = int(_os.environ.get(env_var, default_chunk))
    n = len(imgs)
    if n == 0:
        return _np.empty((0,))
    results = []
    i = 0
    while i < n:
        end = min(i + chunk_size, n)
        batch = imgs[i:end]
        try:
            feats = feature_extractor.compute_features(batch)
            if isinstance(feats, list):
                feats = _np.array(feats)
            results.append(feats)
            i = end
        except Exception as e:
            msg = str(e).lower()
            is_oom = ('out of memory' in msg) or ('cuda' in msg and 'memory' in msg)
            if is_oom and chunk_size > 1:
                # Back off and retry this batch with smaller chunk
                chunk_size = max(1, chunk_size // 2)
                try:
                    import torch as _torch
                    _torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            raise
    return _np.vstack(results) if results else _np.empty((0,))

def main():
    parser = argparse.ArgumentParser(description='Visual Place Recognition: A Tutorial. Code repository supplementing our paper.')
    parser.add_argument('--descriptor', type=lambda s: s.lower(), default='eigenplaces', choices=['hdc-delf', 'alexnet', 'netvlad', 'patchnetvlad', 'cosplace', 'eigenplaces', 'sad'], help='Select descriptor (case-insensitive; default: hdc-delf)')
    parser.add_argument('--dataset', type=lambda s: s.lower(), default='gardenspoint', choices=['gardenspoint', 'gardenspoint_mini', 'stlucia', 'sfu', 'tokyo247', 'sfu_mini', 'nordland_mini', 'nordland_mini_2', 'nordland_mini_3', 'nordland_mini_g2s2', 'nordland_mini_g3s3', 'nordland_mini_g3s5', 'nordland_mini_g3s10', 'nordland_mini_g2s10'], help='Select dataset (case-insensitive; default: gardenspoint)')
    args = parser.parse_args()

    print('========== Start VPR with {} descriptor on dataset {}'.format(args.descriptor, args.dataset))

    # load dataset
    print('===== Load dataset')
    if args.dataset == 'gardenspoint':
        dataset = GardensPointDataset()
    elif args.dataset == 'gardenspoint_mini':
        dataset = PlaceConditionsDataset(destination='images/GardensPoint_Mini/', db_condition='day_right', q_condition='night_right')
    elif args.dataset == 'stlucia':
        dataset = StLuciaDataset()
    elif args.dataset == 'sfu':
        dataset = SFUDataset()
    elif args.dataset == 'tokyo247':
        dataset = Tokyo247Dataset()
    elif args.dataset == 'sfu_mini':
        dataset = PlaceConditionsDataset(destination='images/SFU_Mini/', db_condition='dry', q_condition='jan')
    elif args.dataset == 'nordland_mini':
        # Use PlaceConditionsDataset to compare two Nordland seasons (default: spring vs winter)
        dataset = PlaceConditionsDataset(destination='images/Nordland_Mini/', db_condition='spring', q_condition='winter')
    elif args.dataset == 'nordland_mini_2':
        dataset = PlaceConditionsDataset(destination='images/Nordland_mini_2/', db_condition='spring', q_condition='winter')
    elif args.dataset == 'nordland_mini_3':
        dataset = PlaceConditionsDataset(destination='images/Nordland_Mini_3/', db_condition='spring', q_condition='winter')
    elif args.dataset == 'nordland_mini_g2s2':
        dataset = PlaceConditionsDataset(destination='images/Nordland_Mini_g2s2/', db_condition='spring', q_condition='winter')
    elif args.dataset == 'nordland_mini_g3s3':
        dataset = PlaceConditionsDataset(destination='images/Nordland_Mini_g3s3/', db_condition='spring', q_condition='winter')
    elif args.dataset == 'nordland_mini_g3s5':
        dataset = PlaceConditionsDataset(destination='images/Nordland_Mini_g3_s5/', db_condition='spring', q_condition='winter')
    elif args.dataset == 'nordland_mini_g3s10':
        dataset = PlaceConditionsDataset(destination='images/Nordland_Mini_g3s10/', db_condition='spring', q_condition='winter')
    elif args.dataset == 'nordland_mini_g2s10':
        dataset = PlaceConditionsDataset(destination='images/Nordland_Mini_g2s10/', db_condition='spring', q_condition='winter')
    else:
        raise ValueError('Unknown dataset: ' + args.dataset)

    imgs_db, imgs_q, GThard, GTsoft = dataset.load()

    if args.descriptor == 'hdc-delf':
        from feature_extraction.feature_extractor_holistic import HDCDELF
        feature_extractor = HDCDELF()
    elif args.descriptor == 'alexnet':
        from feature_extraction.feature_extractor_holistic import AlexNetConv3Extractor
        feature_extractor = AlexNetConv3Extractor()
    elif args.descriptor == 'sad':
        from feature_extraction.feature_extractor_holistic import SAD
        feature_extractor = SAD()
    elif args.descriptor == 'netvlad' or args.descriptor == 'patchnetvlad':
        from feature_extraction.feature_extractor_patchnetvlad import PatchNetVLADFeatureExtractor
        from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR
        if args.descriptor == 'netvlad':
            configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/netvlad_extract.ini')
        else:
            configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/speed.ini')
        assert os.path.isfile(configfile)
        config = configparser.ConfigParser()
        config.read(configfile)
        feature_extractor = PatchNetVLADFeatureExtractor(config)
    elif args.descriptor == 'cosplace':
        from feature_extraction.feature_extractor_cosplace import CosPlaceFeatureExtractor
        feature_extractor = CosPlaceFeatureExtractor()
    elif args.descriptor == 'eigenplaces':
        from feature_extraction.feature_extractor_eigenplaces import EigenPlacesFeatureExtractor
        feature_extractor = EigenPlacesFeatureExtractor()
    else:
        raise ValueError('Unknown descriptor: ' + args.descriptor)

    if args.descriptor != 'patchnetvlad' and args.descriptor != 'sad':
        print('===== Compute reference set descriptors')
        db_D_holistic = compute_features_chunked(feature_extractor, imgs_db)
        print('===== Compute query set descriptors')
        q_D_holistic = compute_features_chunked(feature_extractor, imgs_q)

        # normalize descriptors and compute S-matrix
        print('===== Compute cosine similarities S')
        db_D_holistic = normalize_l2(db_D_holistic)
        q_D_holistic = normalize_l2(q_D_holistic)
        S = np.matmul(db_D_holistic , q_D_holistic.transpose())
    elif args.descriptor == 'sad':
        print('===== Compute reference set descriptors')
        db_D_holistic = feature_extractor.compute_features(imgs_db)
        print('===== Compute query set descriptors')
        q_D_holistic = feature_extractor.compute_features(imgs_q)

        # compute similarity matrix S with sum of absolute differences (SAD)
        print('===== Compute similarities S from sum of absolute differences (SAD)')
        S = np.empty([len(imgs_db), len(imgs_q)], 'float32')
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                diff = db_D_holistic[i]-q_D_holistic[j]
                dim = len(db_D_holistic[0]) - np.sum(np.isnan(diff))
                diff[np.isnan(diff)] = 0
                S[i,j] = -np.sum(np.abs(diff)) / dim
    else:
        print('=== WARNING: The PatchNetVLAD code in this repository is not optimised and will be slow and memory consuming.')
        print('===== Compute reference set descriptors')
        db_D_holistic, db_D_patches = feature_extractor.compute_features(imgs_db)
        print('===== Compute query set descriptors')
        q_D_holistic, q_D_patches = feature_extractor.compute_features(imgs_q)
        S = feature_extractor.local_matcher_from_numpy_single_scale(q_D_patches, db_D_patches)

    # show similarity matrix
    fig = plt.figure()
    plt.imshow(S)
    plt.axis('off')
    plt.title('Similarity matrix S')

    # matching decision making
    print('===== Match images')

    # best match per query -> Single-best-match VPR
    M1 = best_match_per_query(S)

    # thresholding -> Multi-match VPR
    M2 = thresholding(S, 'auto')
    TP = np.argwhere(M2 & GThard)  # true positives
    FP = np.argwhere(M2 & ~GTsoft)  # false positives

    # evaluation
    print('===== Evaluation')
    # show correct and wrong image matches
    show_correct_and_wrong_matches.show(
        imgs_db, imgs_q, TP, FP)  # show random matches

    # show M's
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(M1)
    ax1.axis('off')
    ax1.set_title('Best match per query')
    ax2 = fig.add_subplot(122)
    ax2.imshow(M2)
    ax2.axis('off')
    ax2.set_title('Thresholding S>=thresh')

    # Preload thresholds for Recall@K evaluation
    _DATASET_RESULTS_DIR = {
        'gardenspoint': 'GardensPoint', 'gardenspoint_mini': 'GardensPoint_Mini',
        'stlucia': 'StLuciaSmall', 'sfu': 'SFU', 'tokyo247': 'Tokyo247',
        'sfu_mini': 'SFU_Mini', 'nordland_mini': 'Nordland_Mini',
        'nordland_mini_2': 'Nordland_Mini_2', 'nordland_mini_3': 'Nordland_Mini_3',
        'nordland_mini_g2s2': 'Nordland_Mini_g2s2', 'nordland_mini_g3s3': 'Nordland_Mini_g3s3',
        'nordland_mini_g3s5': 'Nordland_Mini_g3s5', 'nordland_mini_g3s10': 'Nordland_Mini_g3s10',
        'nordland_mini_g2s10': 'Nordland_Mini_g2s10',
    }
    results_dir = _DATASET_RESULTS_DIR.get(args.dataset, 'GardensPoint')
    thresholds_path = f"results/{results_dir}/{args.descriptor}/place_averages.csv"
    if os.path.exists(thresholds_path):
        print(f"\nLoading thresholds from: {thresholds_path}")
        df_thresholds = pd.read_csv(thresholds_path)
        # Normalize column names to lowercase for consistency
        df_thresholds.columns = [c.lower() for c in df_thresholds.columns]
        db_place_ids = get_place_ids_from_paths(dataset.db_paths)
        q_place_ids = get_place_ids_from_paths(dataset.q_paths)
    else:
        df_thresholds = None
        db_place_ids, q_place_ids = None, None
        print(f"\nWarning: Threshold file not found at {thresholds_path}")

    # =================================================================================
    # ===== New Section: Recall@K Evaluation ========================================
    # =================================================================================
    print("\n\n===== Recall@K Evaluation =====")

    # Fixed K values for Recall@K
    ks = [1, 3, 5, 10]

    # --- Method 1: Baseline (Original, Threshold-less Ranking) ---
    print(f"\n--- Method 1: Baseline (Threshold-less Ranking) ---")
    R_at_K_baseline = {}
    for k in ks:
        R_at_K_baseline[k] = recallAtK(S, GThard, K=k) * 100 # Convert to percentage
    print_recall_at_k_results(R_at_K_baseline)
    save_recall_results_csv(args.dataset, args.descriptor, 'baseline', R_at_K_baseline)
    
    # --- Our Methods (Per-Place Thresholds with Filtering) ---
    if os.path.exists(thresholds_path):
        simple_avg_thresholds = df_thresholds.set_index('place')['simple_avg_threshold'].to_dict()
        weighted_avg_thresholds = df_thresholds.set_index('place')['weighted_avg_threshold'].to_dict()

        # --- Method 2: Simple Average Per-Place Thresholds ---
        print(f"\n--- Method 2: Simple Average (Filter-then-Rank) ---")
        R_at_K_simple = calculate_recall_at_k_with_filtering(S, GThard, db_place_ids, q_place_ids, per_place_thresholds=simple_avg_thresholds, ks=ks)
        print_recall_at_k_results(R_at_K_simple)
        save_recall_results_csv(args.dataset, args.descriptor, 'simple_avg_thresholds', R_at_K_simple)

        # --- Method 3: Weighted Average Per-Place Thresholds ---
        print(f"\n--- Method 3: Weighted Average (Filter-then-Rank) ---")
        R_at_K_weighted = calculate_recall_at_k_with_filtering(S, GThard, db_place_ids, q_place_ids, per_place_thresholds=weighted_avg_thresholds, ks=ks)
        print_recall_at_k_results(R_at_K_weighted)
        save_recall_results_csv(args.dataset, args.descriptor, 'weighted_avg_thresholds', R_at_K_weighted)
    else:
        print("\nSkipping per-place threshold Recall@K evaluation because thresholds file was not found.")

    plt.show()


if __name__ == '__main__':
    main()
