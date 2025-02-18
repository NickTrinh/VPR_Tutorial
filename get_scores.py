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

destination = 'images/MatchingPairs/'
img_descriptors_matrix = np.ndarray(shape=(10,2), dtype=object)
all_scores = np.ndarray(shape=(10,2), dtype=ScoresStruct)


def get_descriptor(img_dir):
    feature_extractor = HDCDELF()
    get_img = sorted(glob(img_dir))
    img = [np.array(Image.open(img)) for img in get_img]
    return feature_extractor.compute_features(img)

def get_scores_for_img(this_img_i, this_img_j):
    good_scores = []
    bad_scores = []
    
    this_img_feature = img_descriptors_matrix[this_img_i, this_img_j]
    this_img_feature = this_img_feature / np.linalg.norm(this_img_feature , axis=1, keepdims=True)
    
    print(f'===== Compute cosine similarities S for p{this_img_i}/i{this_img_j} against all other images')
    
    for i in range (10):
        for j in range (2):
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

    mean_bad_scores = np.mean(bad_scores)
    std_dev_bad_scores = np.std(bad_scores)
    filter_n = 0
    
    for i in range (4):
        lower_range = mean_bad_scores - (i * std_dev_bad_scores)
        upper_range = mean_bad_scores + (i * std_dev_bad_scores)
        if good_scores[0] < lower_range or good_scores[0] > upper_range:
            filter_n = i
            break
    
    all_scores[this_img_i, this_img_j] = ScoresStruct(good_scores[0], bad_scores, mean_bad_scores, std_dev_bad_scores, filter_n)

def save_scores_to_csv(filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Pair', 'Good Score', 'Bad Scores', 'Mean of Bad Scores', 'Standard Deviation of Bad Scores'])
        
        for i in range(10):
            for j in range(2):
                scores = all_scores[i, j]
                writer.writerow([f'p{i}/i{j}', scores.good_scores, scores.bad_scores, scores.mean_bad_scores, scores.std_dev_bad_scores])


def get_scores():
    print('===== Computing similarity for each images against all other images')
    print('===== Computing scores for all images')
    for i in range (10):
        for j in range (2):
            print(f'===== Computing descriptors for p{i}/i{j}')
            img_descriptor = get_descriptor(destination + f'p{i}/i{j}' + '/*.jpg')
            #print(img_descriptor[0])
            img_descriptors_matrix[i,j] = img_descriptor

    #print(img_descriptors_matrix)
    
    for i in range (10):
        for j in range (2):
            get_scores_for_img(i,j)
    
    for i in range (10):
        for j in range (2):
            print(f'Values for p{i}/i{j}')
            print('Good score: ' + str(all_scores[i,j].good_scores))
            print('Bad scores: ' + str(all_scores[i,j].bad_scores))
            print('Mean of bad scores: ' + str(all_scores[i,j].mean_bad_scores))
            print('Standard deviation of bad scores: ' + str(all_scores[i,j].std_dev_bad_scores))
    
    save_scores_to_csv('scores.csv')


# def new_img_check():
    


if __name__ == '__main__':
    get_scores()