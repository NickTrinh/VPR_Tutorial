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

from evaluation.metrics import createPR, recallAt100precision, recallAtK
from evaluation import show_correct_and_wrong_matches
from matching import matching
from datasets.load_dataset import GardensPointDataset, StLuciaDataset, SFUDataset, Tokyo247Dataset
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


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
    """Extracts place ID from a list of file paths as a string (e.g., 'p0')."""
    place_ids = []
    for path in paths:
        try:
            filename = os.path.basename(path)
            img_num = int(filename.replace('Image', '').replace('.jpg', ''))
            place_id_num = img_num // 10
            place_ids.append(f"p{place_id_num}")
        except (ValueError, IndexError):
            place_ids.append("p-1") # Invalid place ID
    return place_ids

def main():
    parser = argparse.ArgumentParser(description='Visual Place Recognition: A Tutorial. Code repository supplementing our paper.')
    parser.add_argument('--descriptor', type=str, default='HDC-DELF', choices=['HDC-DELF', 'AlexNet', 'NetVLAD', 'PatchNetVLAD', 'CosPlace', 'EigenPlaces', 'SAD'], help='Select descriptor (default: HDC-DELF)')
    parser.add_argument('--dataset', type=str, default='GardensPoint', choices=['GardensPoint', 'GardensPoint_Mini', 'StLucia', 'SFU', 'Tokyo247'], help='Select dataset (default: GardensPoint)')
    args = parser.parse_args()

    print('========== Start VPR with {} descriptor on dataset {}'.format(args.descriptor, args.dataset))

    # load dataset
    print('===== Load dataset')
    if args.dataset == 'GardensPoint':
        dataset = GardensPointDataset()
    elif args.dataset == 'GardensPoint_Mini':
        # We can reuse the original loader
        dataset = GardensPointDataset(destination='images/GardensPoint_Mini/')
    elif args.dataset == 'StLucia':
        dataset = StLuciaDataset()
    elif args.dataset == 'SFU':
        dataset = SFUDataset()
    elif args.dataset == 'Tokyo247':
        dataset = Tokyo247Dataset()
    else:
        raise ValueError('Unknown dataset: ' + args.dataset)

    imgs_db, imgs_q, GThard, GTsoft = dataset.load()

    if args.descriptor == 'HDC-DELF':
        from feature_extraction.feature_extractor_holistic import HDCDELF
        feature_extractor = HDCDELF()
    elif args.descriptor == 'AlexNet':
        from feature_extraction.feature_extractor_holistic import AlexNetConv3Extractor
        feature_extractor = AlexNetConv3Extractor()
    elif args.descriptor == 'SAD':
        from feature_extraction.feature_extractor_holistic import SAD
        feature_extractor = SAD()
    elif args.descriptor == 'NetVLAD' or args.descriptor == 'PatchNetVLAD':
        from feature_extraction.feature_extractor_patchnetvlad import PatchNetVLADFeatureExtractor
        from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR
        if args.descriptor == 'NetVLAD':
            configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/netvlad_extract.ini')
        else:
            configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/speed.ini')
        assert os.path.isfile(configfile)
        config = configparser.ConfigParser()
        config.read(configfile)
        feature_extractor = PatchNetVLADFeatureExtractor(config)
    elif args.descriptor == 'CosPlace':
        from feature_extraction.feature_extractor_cosplace import CosPlaceFeatureExtractor
        feature_extractor = CosPlaceFeatureExtractor()
    elif args.descriptor == 'EigenPlaces':
        from feature_extraction.feature_extractor_eigenplaces import EigenPlacesFeatureExtractor
        feature_extractor = EigenPlacesFeatureExtractor()
    else:
        raise ValueError('Unknown dataset: ' + args.descriptor)

    if args.descriptor != 'PatchNetVLAD' and args.descriptor != 'SAD':
        print('===== Compute reference set descriptors')
        db_D_holistic = feature_extractor.compute_features(imgs_db)
        print('===== Compute query set descriptors')
        q_D_holistic = feature_extractor.compute_features(imgs_q)

        # normalize descriptors and compute S-matrix
        print('===== Compute cosine similarities S')
        db_D_holistic = db_D_holistic / np.linalg.norm(db_D_holistic , axis=1, keepdims=True)
        q_D_holistic = q_D_holistic / np.linalg.norm(q_D_holistic , axis=1, keepdims=True)
        S = np.matmul(db_D_holistic , q_D_holistic.transpose())
    elif args.descriptor == 'SAD':
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

    # PR-curve
    print("\\n===== Generating Baseline PR Curve (Global Threshold) =====")
    P_baseline, R_baseline = createPR(S, GThard, GTsoft, matching='multi', n_thresh=100)
    AUC_baseline = np.trapz(P_baseline, R_baseline)
    print(f'Baseline AUC: {AUC_baseline:.3f}')

    # --- Your Method: Per-Place Thresholding ---
    print("\n===== Generating PR Curve for Our Method (Per-Place Thresholds) =====")
    
    # 1. Load the learned per-place thresholds
    thresholds_path = f"results/{dataset.destination.split('/')[-2]}/place_averages.csv"
    if not os.path.exists(thresholds_path):
        print(f"Error: Threshold file not found at {thresholds_path}")
        print("Please run multi_dataset_runner.py on gardens_point_mini first.")
        return
        
    df_thresholds = pd.read_csv(thresholds_path)
    # Create a dictionary mapping place_id to its learned threshold
    place_thresholds = df_thresholds.set_index('Place')['Mean Bad Scores'].to_dict()
    print(f"Loaded {len(place_thresholds)} per-place thresholds.")

    # 2. We need to know which image belongs to which place
    # We can infer this from the filenames based on our grouping rule
    db_place_ids = get_place_ids_from_paths(dataset.db_paths)
    q_place_ids = get_place_ids_from_paths(dataset.q_paths)

    # 3. Generate the PR curve for your method
    # We simulate a PR curve by incrementally applying the thresholds
    # Get the unique threshold values and sort them. This will be our sweep.
    threshold_values = sorted(list(set(place_thresholds.values())))
    
    P_ours, R_ours = [], []

    # Sweep from most lenient to most strict
    for thresh_sweep in threshold_values:
        # Only use per-place thresholds that are >= the current sweep value
        # This simulates making the matcher more strict
        active_thresholds = {pid: t for pid, t in place_thresholds.items() if t >= thresh_sweep}
        
        # Get the matching matrix M based on the active thresholds
        M_ours = apply_per_place_thresholds(S, db_place_ids, q_place_ids, active_thresholds)
        
        # Now, use the ground truth to score the matches in M
        TP = np.count_nonzero(M_ours & GThard)
        FP = np.count_nonzero(M_ours & ~GTsoft)
        GTP = np.count_nonzero(GThard)

        if (TP + FP) > 0:
            precision = TP / (TP + FP)
            recall = TP / GTP if GTP > 0 else 0
            P_ours.append(precision)
            R_ours.append(recall)

    # Ensure the curve starts at (0, 1) and is monotonic
    P_ours = [1.0] + P_ours
    R_ours = [0.0] + R_ours
    
    AUC_ours = np.trapz(P_ours, R_ours)
    print(f'Our Method AUC: {AUC_ours:.3f}')

    # --- Plotting Comparison ---
    dataset_name = dataset.destination.split('/')[-2]
    plt.figure(figsize=(10, 8))
    plt.plot(R_baseline, P_baseline, label=f'Baseline (Global Auto-Threshold) | AUC = {AUC_baseline:.3f}', marker='.')
    plt.plot(R_ours, P_ours, label=f'Our Method (Per-Place Thresholds) | AUC = {AUC_ours:.3f}', marker='x')
    
    plt.xlim(0, 1), plt.ylim(0, 1.01)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve Comparison on {dataset_name}')
    plt.grid('on')
    plt.legend()
    plt.savefig(f"output_images/pr_curve_comparison_{dataset_name}.jpg")
    print(f"\nSaved comparison PR curve to output_images/pr_curve_comparison_{dataset_name}.jpg")
    plt.show()


if __name__ == '__main__':
    main()
