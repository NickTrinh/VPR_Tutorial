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
import csv
import os
from datetime import datetime

import numpy as np


def recallAtK(S_in, GThard, GTsoft=None, K=1):
    """
    Calculates the recall@K for a given similarity matrix S_in and ground truth matrices 
    GThard and GTsoft.

    The matrices S_in, GThard and GTsoft are two-dimensional and should all have the
    same shape.
    The matrices GThard and GTsoft should be binary matrices, where the entries are
    only zeros or ones.
    The matrix S_in should have continuous values between -Inf and Inf. Higher values
    indicate higher similarity.
    The integer K>=1 defines the number of matching candidates that are selected and
    that must contain an actually matching image pair.
    """

    assert (S_in.shape == GThard.shape),"S_in and GThard must have the same shape"
    if GTsoft is not None:
        assert (S_in.shape == GTsoft.shape),"S_in and GTsoft must have the same shape"
    assert (S_in.ndim == 2),"S_in, GThard and GTsoft must be two-dimensional"
    assert (K >= 1),"K must be >=1"

    # ensure logical datatype in GT and GTsoft
    GT = GThard.astype('bool')
    if GTsoft is not None:
        GTsoft = GTsoft.astype('bool')

    # copy S and set elements that are only true in GTsoft to min(S) to ignore them during evaluation
    S = S_in.copy()
    if GTsoft is not None:
        S[GTsoft & ~GT] = S.min()

    # discard all query images without an actually matching database image
    j = GT.sum(0) > 0 # columns with matches
    S = S[:,j] # select columns with a match
    GT = GT[:,j] # select columns with a match

    # select K highest similarities
    i = S.argsort(0)[-K:,:]
    j = np.tile(np.arange(i.shape[1]), [K, 1])
    GT = GT[i, j]

    # recall@K
    RatK = np.sum(GT.sum(0) > 0) / GT.shape[1]

    return RatK


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
            continue  # Skip queries with no correct match in DB
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
        candidate_places = {}  # {place_id: best_score}
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
            writer.writerow(['Timestamp', 'Dataset', 'Descriptor', 'Method', 'Recall@1', 'Recall@3', 'Recall@5', 'Recall@10'])
        r1 = results_dict.get(1, '')
        r3 = results_dict.get(3, '')
        r5 = results_dict.get(5, '')
        r10 = results_dict.get(10, '')
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            dataset_name,
            descriptor_name,
            method_name,
            f"{r1:.2f}" if r1 != '' else '',
            f"{r3:.2f}" if r3 != '' else '',
            f"{r5:.2f}" if r5 != '' else '',
            f"{r10:.2f}" if r10 != '' else ''
        ])
