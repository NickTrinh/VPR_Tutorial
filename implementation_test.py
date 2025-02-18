import numpy as np
import csv
from matching import matching
from feature_extraction.feature_extractor_holistic import HDCDELF
from PIL import Image
from glob import glob
from typing import NamedTuple
from dataclasses import dataclass

@dataclass
class ScoresStruct():
    good_scores: float
    bad_scores: list
    mean_bad_scores: float
    std_dev_bad_scores: float
    filter_n: int

@dataclass
class Place():
    mean_bad_scores: float
    std_dev_bad_scores: float
    filter_n: float

# Manually assigning values to the Place struct instances

places = [
    Place(mean_bad_scores=0.027348, std_dev_bad_scores=0.021431, filter_n=21.266667),  # p0
    Place(mean_bad_scores=0.035565, std_dev_bad_scores=0.026386, filter_n=10.633333),  # p1
    Place(mean_bad_scores=0.049672, std_dev_bad_scores=0.030017, filter_n=3.416667),   # p2
    Place(mean_bad_scores=0.041013, std_dev_bad_scores=0.036605, filter_n=5.166667),   # p3
    Place(mean_bad_scores=0.060161, std_dev_bad_scores=0.057331, filter_n=5.444444),   # p4
    Place(mean_bad_scores=0.052739, std_dev_bad_scores=0.048123, filter_n=4.500000),   # p5
    Place(mean_bad_scores=0.045892, std_dev_bad_scores=0.040217, filter_n=6.733333),   # p6
    Place(mean_bad_scores=0.048021, std_dev_bad_scores=0.042843, filter_n=4.600000),   # p7
    Place(mean_bad_scores=0.053657, std_dev_bad_scores=0.049021, filter_n=4.500000),   # p8
    Place(mean_bad_scores=0.056789, std_dev_bad_scores=0.050902, filter_n=5.200000)    # p9
]

places_matrix = [
    [
        Place(mean_bad_scores=0.020714, std_dev_bad_scores=0.015921, filter_n=35.0),  # p0/i0
        Place(mean_bad_scores=0.022136, std_dev_bad_scores=0.016253, filter_n=24.8),  # p0/i1
        Place(mean_bad_scores=0.039192, std_dev_bad_scores=0.032119, filter_n=4.0)    # p0/i2
    ],
    [
        Place(mean_bad_scores=0.034049, std_dev_bad_scores=0.020597, filter_n=17.4),  # p1/i0
        Place(mean_bad_scores=0.031653, std_dev_bad_scores=0.023510, filter_n=11.0),  # p1/i1
        Place(mean_bad_scores=0.040993, std_dev_bad_scores=0.035052, filter_n=3.5)    # p1/i2
    ],
    [
        Place(mean_bad_scores=0.057334, std_dev_bad_scores=0.029992, filter_n=2.5),   # p2/i0
        Place(mean_bad_scores=0.045119, std_dev_bad_scores=0.028257, filter_n=5.0),   # p2/i1
        Place(mean_bad_scores=0.046564, std_dev_bad_scores=0.031802, filter_n=2.75)   # p2/i2
    ],
    [
        Place(mean_bad_scores=0.048083, std_dev_bad_scores=0.049731, filter_n=2.0),   # p3/i0
        Place(mean_bad_scores=0.032191, std_dev_bad_scores=0.028965, filter_n=8.0),   # p3/i1
        Place(mean_bad_scores=0.042766, std_dev_bad_scores=0.031119, filter_n=5.5)    # p3/i2
    ],
    [
        Place(mean_bad_scores=0.068348, std_dev_bad_scores=0.061148, filter_n=5.0),   # p4/i0
        Place(mean_bad_scores=0.053499, std_dev_bad_scores=0.054164, filter_n=7.0),   # p4/i1
        Place(mean_bad_scores=0.058637, std_dev_bad_scores=0.056680, filter_n=4.33)   # p4/i2
    ],
    [
        Place(mean_bad_scores=0.067613, std_dev_bad_scores=0.048807, filter_n=2.5),   # p5/i0
        Place(mean_bad_scores=0.040378, std_dev_bad_scores=0.041240, filter_n=7.0),   # p5/i1
        Place(mean_bad_scores=0.041706, std_dev_bad_scores=0.047086, filter_n=4.0)    # p5/i2
    ],
    [
        Place(mean_bad_scores=0.064587, std_dev_bad_scores=0.059426, filter_n=7.5),   # p6/i0
        Place(mean_bad_scores=0.043784, std_dev_bad_scores=0.052050, filter_n=6.33),  # p6/i1
        Place(mean_bad_scores=0.078152, std_dev_bad_scores=0.068723, filter_n=6.67)   # p6/i2
    ],
    [
        Place(mean_bad_scores=0.068702, std_dev_bad_scores=0.067478, filter_n=5.0),   # p7/i0
        Place(mean_bad_scores=0.069520, std_dev_bad_scores=0.062152, filter_n=5.67),  # p7/i1
        Place(mean_bad_scores=0.076234, std_dev_bad_scores=0.065766, filter_n=3.0)    # p7/i2
    ],
    [
        Place(mean_bad_scores=0.041822, std_dev_bad_scores=0.030725, filter_n=7.0),   # p8/i0
        Place(mean_bad_scores=0.042892, std_dev_bad_scores=0.026950, filter_n=7.0),   # p8/i1
        Place(mean_bad_scores=0.048624, std_dev_bad_scores=0.046853, filter_n=2.67)   # p8/i2
    ],
    [
        Place(mean_bad_scores=0.039208, std_dev_bad_scores=0.023883, filter_n=5.0),   # p9/i0
        Place(mean_bad_scores=0.029648, std_dev_bad_scores=0.024540, filter_n=10.5),  # p9/i1
        Place(mean_bad_scores=0.024691, std_dev_bad_scores=0.020590, filter_n=12.5)   # p9/i2
    ]
]


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