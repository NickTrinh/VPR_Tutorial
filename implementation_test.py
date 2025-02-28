import numpy as np
import csv
from matching import matching
from feature_extraction.feature_extractor_holistic import HDCDELF
from PIL import Image
from glob import glob
from typing import NamedTuple
from dataclasses import dataclass
from places_data import Place, places, places_matrix  # Import the Place class and data

@dataclass
class ScoresStruct():
    good_scores: float
    bad_scores: list
    mean_bad_scores: float
    std_dev_bad_scores: float
    filter_n: int

# The places and places_matrix arrays have been moved to places_data.py

destination = 'images/MatchingTriplets/'

img_descriptors_matrix = np.ndarray(shape=(10,3), dtype=object)
all_scores = np.ndarray(shape=(10,3), dtype=ScoresStruct)
test_results_set = np.ndarray(shape=(10,3), dtype=object)
test_results_bool_set = np.ndarray(shape=(10,3), dtype=object)

picked_set = []
test_set = []

AVG_MEAN_BAD_SCORES = 0
AVG_STD_DEV_BAD_SCORES = 0
AVG_FILTER_N = 0

def get_descriptor(img_dir):
    feature_extractor = HDCDELF()
    get_img = sorted(glob(img_dir))
    img = [np.array(Image.open(img)) for img in get_img]
    return feature_extractor.compute_features(img)

def save_scores_to_csv_normal(filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Mean Bad Scores', 'Std Deviation Bad Scores', 'Filter N'])
        
        for i in range(10):
            for j in picked_set[i]:
                scores = all_scores[i, j]
                
                upper_limit = scores.mean_bad_scores + (scores.filter_n * scores.std_dev_bad_scores)
                lower_limit = scores.mean_bad_scores - (scores.filter_n * scores.std_dev_bad_scores)
                writer.writerow([f'p{i}/i{j}', scores.mean_bad_scores, scores.std_dev_bad_scores, scores.filter_n])

def get_scores_for_img(this_img_i, this_img_j):
    good_scores = []
    bad_scores = []
    
    this_img_feature = img_descriptors_matrix[this_img_i, this_img_j]
    this_img_feature = this_img_feature / np.linalg.norm(this_img_feature , axis=1, keepdims=True)
    
    print(f'===== Compute cosine similarities S for p{this_img_i}/i{this_img_j} against all other images')
    
    for i in range (10):
        for j in picked_set[0]:
            if i == this_img_i and j == this_img_j:
                continue
            
            # normalize descriptors and compute S-matrix
            print(f'===== Computing scores for p{this_img_i}/i{this_img_j} against p{i}/i{j}')
            
            matching_img_feature = img_descriptors_matrix[i,j]
            matching_img_feature = matching_img_feature / np.linalg.norm(matching_img_feature , axis=1, keepdims=True)
            S = np.matmul(this_img_feature , matching_img_feature.transpose())
            
            if i == this_img_i:
                good_scores.append(S[0][0])
            else:
                bad_scores.append(S[0][0])

    print(f'Calculating mean and standard deviation of bad scores for p{this_img_i}/i{this_img_j}')
    mean_bad_scores = np.mean(bad_scores)
    std_dev_bad_scores = np.std(bad_scores)
    filter_n = 0
    
    print(f'Calculating filter_n for p{this_img_i}/i{this_img_j}')
    filter_n = 0

    print(f'Calculating filter_n for p{this_img_i}/i{this_img_j}')
    i = 1
    while True:
        lower_range = mean_bad_scores - (i * std_dev_bad_scores)
        upper_range = mean_bad_scores + (i * std_dev_bad_scores)
        if good_scores[0] < lower_range or good_scores[0] > upper_range:
            filter_n = i
        else:
            break
        i += 1
    
    all_scores[this_img_i, this_img_j] = ScoresStruct(good_scores[0], bad_scores, mean_bad_scores, std_dev_bad_scores, filter_n)


def check_test_all_images():
    
    TruePositives, FalsePositives, TrueNegatives, FalseNegatives = 0, 0, 0, 0
    
    for k in range (10):   
        for i in range (10):
            for j in picked_set[i]:
                # print(f'===== Compute cosine similarities S for test image p{k}/i{test_set[k]}')
                test_img_feature = img_descriptors_matrix[k, test_set[k]]
                test_img_feature = test_img_feature / np.linalg.norm(test_img_feature , axis=1, keepdims=True)
                
                # normalize descriptors and compute S-matrix
                # print(f'===== Computing scores for test image p{k}/i{test_set[k]} against dataset image p{i}/i{j}')
                
                matching_img_feature = img_descriptors_matrix[i,j]
                matching_img_feature = matching_img_feature / np.linalg.norm(matching_img_feature , axis=1, keepdims=True)
                S = np.matmul(test_img_feature , matching_img_feature.transpose())
                
                # print(f'===== Test image p{k}/i{test_set[k]} scores against p{i}/i{j} is {S[0][0]}')
                test_results_set[i][j] = S[0][0]
                
                lower_range = 0
                upper_range = 0
                
                lower_range = all_scores[i,j].mean_bad_scores - (all_scores[i,j].filter_n * all_scores[i,j].std_dev_bad_scores)
                upper_range = all_scores[i,j].mean_bad_scores + (all_scores[i,j].filter_n * all_scores[i,j].std_dev_bad_scores)
                
                # Calculate the upper limit for the unique place score
                averaged_place_score = places[i].mean_bad_scores + (places[i].filter_n * places[i].std_dev_bad_scores)
                #averaged_place_score = places_matrix[i][j].mean_bad_scores + (places_matrix[i][j].filter_n * places_matrix[i][j].std_dev_bad_scores)

                if test_results_set[i][j] < averaged_place_score:
                    print(f'===== Test image p{k}/i{test_set[k]} is rejected when compared to p{i}/i{j}. Score: {test_results_set[i][j]}')
                    test_results_bool_set[i][j] = 'Rejected'
                    if i == k:
                        FalseNegatives += 1
                    else:
                        TrueNegatives += 1
                else:
                    print(f'===== Test image p{k}/i{test_set[k]} is accepted when compared to p{i}/i{j}. Score: {test_results_set[i][j]}')
                    test_results_bool_set[i][j] = 'Accepted'
                    if i == k:
                        TruePositives += 1
                    else:
                        FalsePositives += 1
    
    print(f'True Positives: {TruePositives}')
    print(f'False Positives: {FalsePositives}')
    print(f'True Negatives: {TrueNegatives}')
    print(f'False Negatives: {FalseNegatives}')
    
    save_scores_to_csv_normal('test_results_with_average_place_score.csv')

def run():
    print('===== Computing similarity for each images against all other images')
    print('===== Computing scores for all images')
    for i in range (10):
        print('===== Selecting 2 images from 3 and storing the remaining for testing')
        numbers = set([0, 1, 2]) 
        
        picked_two = list(np.random.choice(list(numbers), size=2, replace=False))
        picked_set.append(picked_two)
        
        print(picked_set[i])
        
        test_one = list(numbers - set(picked_two))[0]
        test_set.append(test_one)
        
        print(test_set[i])
        
        for j in range (3):
            print(f'===== Computing descriptors for p{i}/i{j}')
            img_descriptor = get_descriptor(destination + f'p{i}/i{j}' + '/*.jpg')
            #print(img_descriptor[0])
            img_descriptors_matrix[i,j] = img_descriptor
    
    for i in range (10):
        for j in picked_set[i]:
            print(f'===== Testing image is p{i}/i{test_set[i]}')
            print(f'===== Image pair is {picked_set[i]}')
            get_scores_for_img(i,j)
    
    
    check_test_all_images()


if __name__ == '__main__':
    run()