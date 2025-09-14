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
import csv
from datetime import datetime

from evaluation.metrics import recallAtK
from evaluation import show_correct_and_wrong_matches
from matching import matching
from datasets.load_dataset import GardensPointDataset, StLuciaDataset, SFUDataset, Tokyo247Dataset, PlaceConditionsDataset
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.stats import norm # Added for Recall@K


def apply_per_place_thresholds(S, db_place_ids, q_place_ids, place_thresholds):
    """
    Applies per-place thresholds to the similarity matrix S.
    A match is made if S[i, j] is greater than or equal to the threshold
    for the place corresponding to the database image i.
    """
    M = np.zeros_like(S, dtype=bool)
    num_db, num_q = S.shape
    
    for i in range(num_db):
        place_id = db_place_ids[i]
        # Use the threshold for the database image's place
        if place_id in place_thresholds:
            threshold = place_thresholds[place_id]
            for j in range(num_q):
                if S[i, j] >= threshold:
                    M[i, j] = True
        # If a place has no threshold (e.g., filtered out by the sweep),
        # it cannot make any matches.
            
    return M

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

def main():
    parser = argparse.ArgumentParser(description='Visual Place Recognition: A Tutorial. Code repository supplementing our paper.')
    parser.add_argument('--descriptor', type=lambda s: s.lower(), default='eigenplaces', choices=['hdc-delf', 'alexnet', 'netvlad', 'patchnetvlad', 'cosplace', 'eigenplaces', 'sad'], help='Select descriptor (case-insensitive; default: hdc-delf)')
    parser.add_argument('--dataset', type=lambda s: s.lower(), default='gardenspoint', choices=['gardenspoint', 'gardenspoint_mini', 'stlucia', 'sfu', 'tokyo247', 'stlucia_mini', 'sfu_mini', 'nordland_mini'], help='Select dataset (case-insensitive; default: gardenspoint)')
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
    elif args.dataset == 'stlucia_mini':
        dataset = PlaceConditionsDataset(destination='images/StLucia_Mini/', db_condition='100909_0845', q_condition='180809_1545')
    elif args.dataset == 'sfu_mini':
        dataset = PlaceConditionsDataset(destination='images/SFU_Mini/', db_condition='dry', q_condition='jan')
    elif args.dataset == 'nordland_mini':
        # Use PlaceConditionsDataset to compare two Nordland seasons (default: spring vs winter)
        dataset = PlaceConditionsDataset(destination='images/Nordland_Mini/', db_condition='spring', q_condition='winter')
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
        raise ValueError('Unknown dataset: ' + args.descriptor)

    if args.descriptor != 'patchnetvlad' and args.descriptor != 'sad':
        print('===== Compute reference set descriptors')
        db_D_holistic = feature_extractor.compute_features(imgs_db)
        print('===== Compute query set descriptors')
        q_D_holistic = feature_extractor.compute_features(imgs_q)

        # normalize descriptors and compute S-matrix
        print('===== Compute cosine similarities S')
        db_D_holistic = db_D_holistic / np.linalg.norm(db_D_holistic , axis=1, keepdims=True)
        q_D_holistic = q_D_holistic / np.linalg.norm(q_D_holistic , axis=1, keepdims=True)
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
        # S_hol = np.matmul(db_D_holistic , q_D_holistic.transpose())
        S = feature_extractor.local_matcher_from_numpy_single_scale(q_D_patches, db_D_patches)

    # show similarity matrix
    fig = plt.figure()
    plt.imshow(S)
    plt.axis('off')
    plt.title('Similarity matrix S')

    # matching decision making
    print('===== Match images')

    # best match per query -> Single-best-match VPR
    M1 = matching.best_match_per_query(S)

    # thresholding -> Multi-match VPR
    M2 = matching.thresholding(S, 'auto')
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
    if args.dataset == 'gardenspoint_mini':
        thresholds_path = "results/GardensPoint_Mini/place_averages.csv"
    elif args.dataset == 'stlucia_mini':
        thresholds_path = "results/StLucia_Mini/place_averages.csv"
    elif args.dataset == 'sfu_mini':
        thresholds_path = "results/SFU_Mini/place_averages.csv"
    elif args.dataset == 'nordland_mini':
        thresholds_path = "results/Nordland_Mini/place_averages.csv"
    else:
        thresholds_path = "results/GardensPoint/place_averages.csv"
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

    # --- Method 1: Baseline (Original, Threshold-less Ranking) ---
    print(f"\n--- Method 1: Baseline (Threshold-less Ranking) ---")
    R_at_K_baseline = {}
    for k in [1, 3, 5]:
        R_at_K_baseline[k] = recallAtK(S, GThard, K=k) * 100 # Convert to percentage
    print_recall_at_k_results(R_at_K_baseline)
    save_recall_results_csv(args.dataset, args.descriptor, 'baseline', R_at_K_baseline)
    
    # --- Our Methods (Per-Place Thresholds with Filtering) ---
    if os.path.exists(thresholds_path):
        # --- Method 2: Simple Average Per-Place Thresholds ---
        simple_avg_thresholds = df_thresholds.set_index('place')['simple_avg_threshold'].to_dict()
        print("\n--- Method 2: Simple Average (Filter-then-Rank) ---")
        R_at_K_simple = calculate_recall_at_k_with_filtering(S, GThard, db_place_ids, q_place_ids, per_place_thresholds=simple_avg_thresholds)
        print_recall_at_k_results(R_at_K_simple)
        save_recall_results_csv(args.dataset, args.descriptor, 'simple_avg_thresholds', R_at_K_simple)

        # --- Method 3: Weighted Average Per-Place Thresholds ---
        weighted_avg_thresholds = df_thresholds.set_index('place')['weighted_avg_threshold'].to_dict()
        print("\n--- Method 3: Weighted Average (Filter-then-Rank) ---")
        R_at_K_weighted = calculate_recall_at_k_with_filtering(S, GThard, db_place_ids, q_place_ids, per_place_thresholds=weighted_avg_thresholds)
        print_recall_at_k_results(R_at_K_weighted)
        save_recall_results_csv(args.dataset, args.descriptor, 'weighted_avg_thresholds', R_at_K_weighted)
    else:
        print("\nSkipping per-place threshold Recall@K evaluation because thresholds file was not found.")

    plt.show()


def calculate_recall_at_k_with_filtering(S, GThard, db_place_ids, q_place_ids, per_place_thresholds=None, ks=[1, 3, 5]):
    """
    Calculates Recall@K for our methods by first filtering with per-place thresholds, then ranking.
    """
    num_queries = S.shape[1]
    correct_at_k = {k: 0 for k in ks}
    valid_query_count = 0
    
    # Pre-compute mapping from DB index to place ID
    db_idx_to_place = {i: pid for i, pid in enumerate(db_place_ids)}

    for q_idx in range(num_queries):
        # Find the ground truth place(s) for this query
        gt_db_indices = np.where(GThard[:, q_idx])[0]
        if len(gt_db_indices) == 0:
            continue # Skip queries with no correct match in DB
        valid_query_count += 1
        
        correct_place_ids = set(db_idx_to_place[i] for i in gt_db_indices)

        # Step 1: Filtering - Find all candidate matches that pass the threshold
        candidate_db_indices = []
        scores = S[:, q_idx]

        if per_place_thresholds is not None:
            # Must check each DB image against its own place's threshold
            for db_idx, score in enumerate(scores):
                place_id = db_idx_to_place[db_idx]
                if place_id in per_place_thresholds and score >= per_place_thresholds[place_id]:
                    candidate_db_indices.append(db_idx)
        else:
            # This function is now only for per-place thresholding.
            # The baseline uses the original recallAtK.
            raise ValueError("This function requires per_place_thresholds.")

        if len(candidate_db_indices) == 0:
            continue

        # Step 2: Ranking - Rank the unique places of the candidates by their best score
        candidate_places = {} # {place_id: best_score}
        for db_idx in candidate_db_indices:
            place_id = db_idx_to_place[db_idx]
            score = scores[db_idx]
            if place_id not in candidate_places or score > candidate_places[place_id]:
                candidate_places[place_id] = score
        
        # Sort places by score in descending order
        ranked_places = sorted(candidate_places.keys(), key=lambda pid: candidate_places[pid], reverse=True)

        # Step 3: Evaluation - Check if a correct place is in the top K
        for k in ks:
            top_k_places = set(ranked_places[:k])
            if not top_k_places.isdisjoint(correct_place_ids):
                correct_at_k[k] += 1
    
    # Calculate final recall percentages using only queries with GT matches (aligns with baseline)
    denom = valid_query_count if valid_query_count > 0 else 1
    recall_results = {k: (count / denom) * 100 for k, count in correct_at_k.items()}
    return recall_results

def print_recall_at_k_results(results):
    """Prints a formatted string of Recall@K results."""
    for k, recall in results.items():
        print(f"Recall@{k}: {recall:.2f}%")


def save_recall_results_csv(dataset_name, descriptor_name, method_name, results_dict):
    """Append Recall@K results to results/comparison/recall_at_k.csv"""
    out_dir = os.path.join('results', 'comparison')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'recall_at_k.csv')

    file_exists = os.path.exists(out_file)
    with open(out_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Dataset', 'Descriptor', 'Method', 'Recall@1', 'Recall@3', 'Recall@5'])
        r1 = results_dict.get(1, '')
        r3 = results_dict.get(3, '')
        r5 = results_dict.get(5, '')
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            dataset_name,
            descriptor_name,
            method_name,
            f"{r1:.2f}" if r1 != '' else '',
            f"{r3:.2f}" if r3 != '' else '',
            f"{r5:.2f}" if r5 != '' else ''
        ])


if __name__ == '__main__':  
    main()
