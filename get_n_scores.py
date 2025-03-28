import numpy as np
import csv
from matching import matching
from feature_extraction.feature_extractor_holistic import HDCDELF
from PIL import Image
from glob import glob
from typing import NamedTuple
from dataclasses import dataclass
from places_data import Place, places, places_matrix  # Import the Place class and data
from copy import deepcopy

@dataclass
class ScoresStruct():
    good_scores: float
    bad_scores: list
    mean_bad_scores: float
    std_dev_bad_scores: float
    filter_n: int

# The places and places_matrix arrays have been moved to places_data.py

destination = 'images/FordhamPlaces/'

# Number of runs to average
NUM_RUNS = 30

def get_descriptor(img_dir):
    feature_extractor = HDCDELF()
    get_img = sorted(glob(img_dir))
    img = [np.array(Image.open(img)) for img in get_img]
    return feature_extractor.compute_features(img)

def get_scores_for_img(this_img_i, this_img_j, img_descriptors_matrix, picked_set, test_set):
    """Calculate scores for a specific image against all other images"""
    good_scores = []
    bad_scores = []
    
    this_img_feature = img_descriptors_matrix[this_img_i, this_img_j]
    this_img_feature = this_img_feature / np.linalg.norm(this_img_feature, axis=1, keepdims=True)
    
    print(f'===== Compute cosine similarities S for p{this_img_i}/i{this_img_j} against all other images')
    
    for i in range(11):
        test_img_feature = img_descriptors_matrix[i, test_set[i]]
        test_img_feature = test_img_feature / np.linalg.norm(test_img_feature, axis=1, keepdims=True)
        
        S = np.matmul(this_img_feature, test_img_feature.transpose())
        
        if i == this_img_i:  # Same place, different image
            good_scores.append(S[0][0])
        else:  # Different place
            bad_scores.append(S[0][0])

    print(f'Calculating mean and standard deviation of bad scores for p{this_img_i}/i{this_img_j}')
    mean_bad_scores = np.mean(bad_scores)
    std_dev_bad_scores = np.std(bad_scores)
    
    print(f'Calculating filter_n for p{this_img_i}/i{this_img_j}')
    filter_n = 0
    i = 1
    
    # Using the original method to find filter_n:
    # Find the greatest value of n such that all good scores > mean_bad_scores + n * std_dev_bad_scores
    if good_scores:
        min_good_score = min(good_scores)
        while True:
            threshold = mean_bad_scores + (i * std_dev_bad_scores)
            
            # If good score is no longer greater than threshold, return i-1
            if min_good_score <= threshold:
                filter_n = i - 1
                break
            
            i += 1
            if i > 100:  # Safety break
                filter_n = 100
                break
    
    return ScoresStruct(good_scores[0] if good_scores else 0, bad_scores, mean_bad_scores, std_dev_bad_scores, filter_n)

def run_single_test():
    """Run a single test with random selection of 2 images out of 3 for each place"""
    print('===== Computing descriptors for all images')
    
    # Initialize descriptor matrix
    img_descriptors_matrix = np.ndarray(shape=(11, 3), dtype=object)
    
    # Initialize picked and test sets
    picked_set = []
    test_set = []
    
    # Randomly select 2 images out of 3 for each place
    print('===== Selecting 2 images from 3 and storing the remaining for testing')
    for i in range(11):
        numbers = set([0, 1, 2]) 
        picked_two = list(np.random.choice(list(numbers), size=2, replace=False))
        picked_set.append(picked_two)
        
        print(f'Place {i}: Picked images: {picked_set[i]}')
        
        test_one = list(numbers - set(picked_two))[0]
        test_set.append(test_one)
        
        print(f'Place {i}: Test image: {test_set[i]}')
    
    # Compute descriptors for all images
    for i in range(11):
        for j in range(3):
            print(f'===== Computing descriptors for p{i}/i{j}')
            img_descriptor = get_descriptor(destination + f'p{i}/i{j}' + '/*.jpg')
            img_descriptors_matrix[i, j] = img_descriptor
    
    # Calculate scores for picked images only
    print('===== Computing scores for picked images')
    all_scores = {}
    
    for i in range(11):
        for j in picked_set[i]:
            print(f'===== Computing scores for p{i}/i{j}')
            scores = get_scores_for_img(i, j, img_descriptors_matrix, picked_set, test_set)
            img_key = f'p{i}/i{j}'
            all_scores[img_key] = {
                'mean_bad_scores': scores.mean_bad_scores,
                'std_dev_bad_scores': scores.std_dev_bad_scores,
                'filter_n': scores.filter_n
            }
    
    return all_scores

def run_multiple_tests():
    """Run multiple tests and calculate average scores for all images across runs"""
    print(f'===== Running {NUM_RUNS} tests and calculating averages')
    
    # Store results from all runs
    all_runs_results = []
    
    # Run the tests
    for run_num in range(NUM_RUNS):
        print(f'\n===== Starting Run {run_num+1}/{NUM_RUNS} =====')
        
        # Run the test and collect results
        run_result = run_single_test()
        all_runs_results.append(run_result)
        
        # Save individual run results
        save_run_results(f'test_results_run_{run_num+1}.csv', run_result)
    
    # Calculate and save average results
    calculate_and_save_averages(all_runs_results)

def save_run_results(filename, results):
    """Save results from a single run to CSV"""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Mean Bad Scores', 'Std Deviation Bad Scores', 'Filter N'])
        
        for img_key, scores in sorted(results.items()):
            writer.writerow([
                img_key, 
                scores['mean_bad_scores'], 
                scores['std_dev_bad_scores'],
                scores['filter_n']
            ])

def calculate_and_save_averages(all_results):
    """Calculate and save average scores across all runs"""
    # Aggregate all image results
    aggregated_results = {}
    
    # Combine results from all runs
    for run_results in all_results:
        for img_key, scores in run_results.items():
            if img_key not in aggregated_results:
                aggregated_results[img_key] = {
                    'mean_bad_scores': [],
                    'std_dev_bad_scores': [],
                    'filter_n': []
                }
            
            aggregated_results[img_key]['mean_bad_scores'].append(scores['mean_bad_scores'])
            aggregated_results[img_key]['std_dev_bad_scores'].append(scores['std_dev_bad_scores'])
            aggregated_results[img_key]['filter_n'].append(scores['filter_n'])
    
    # Calculate averages
    averaged_results = {}
    for img_key, scores_list in aggregated_results.items():
        averaged_results[img_key] = {
            'mean_bad_scores': np.mean(scores_list['mean_bad_scores']),
            'std_dev_bad_scores': np.mean(scores_list['std_dev_bad_scores']),
            'filter_n': np.mean(scores_list['filter_n'])
        }
    
    # Save averaged results to CSV
    with open('averaged_results_30_runs.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Mean Bad Scores', 'Std Deviation Bad Scores', 'Filter N'])
        
        # Sort the keys for better readability
        sorted_keys = sorted(averaged_results.keys(), 
                            key=lambda x: (int(x.split('/')[0][1:]), int(x.split('/')[1][1:])))
        
        for img_key in sorted_keys:
            scores = averaged_results[img_key]
            writer.writerow([
                img_key, 
                scores['mean_bad_scores'], 
                scores['std_dev_bad_scores'],
                scores['filter_n']
            ])
    
    print("\n===== SUMMARY OF AVERAGED SCORES =====")
    print(f"Results saved to averaged_results_30_runs.csv")

# Main entry point
if __name__ == '__main__':
    run_multiple_tests()