import numpy as np
from PIL import Image
from glob import glob
from feature_extraction.feature_extractor_holistic import HDCDELF
from places_data import places, places_matrix

def get_descriptor(img_dir):
    feature_extractor = HDCDELF()
    get_img = sorted(glob(img_dir))
    img = [np.array(Image.open(img)) for img in get_img]
    return feature_extractor.compute_features(img)

def test_with_thresholds():
    """
    Run a test using the threshold values from places_data.py
    to calculate TP, FP, TN, FN without recalculating filter_n.
    """
    destination = 'images/MatchingTriplets/'
    
    # Initialize descriptor matrix
    img_descriptors_matrix = np.ndarray(shape=(10, 3), dtype=object)
    
    # Initialize picked and test sets
    picked_set = []
    test_set = []
    
    # Randomly select 2 images out of 3 for each place
    print('===== Selecting 2 images from 3 and storing the remaining for testing')
    for i in range(10):
        numbers = set([0, 1, 2]) 
        picked_two = list(np.random.choice(list(numbers), size=2, replace=False))
        picked_set.append(picked_two)
        
        print(f'Place {i}: Picked images: {picked_set[i]}')
        
        test_one = list(numbers - set(picked_two))[0]
        test_set.append(test_one)
        
        print(f'Place {i}: Test image: {test_set[i]}')
    
    # Compute descriptors for all images
    for i in range(10):
        for j in range(3):
            print(f'===== Computing descriptors for p{i}/i{j}')
            img_descriptor = get_descriptor(destination + f'p{i}/i{j}' + '/*.jpg')
            img_descriptors_matrix[i, j] = img_descriptor
    
    # Initialize counters for TP, FP, TN, FN
    TP, FP, TN, FN = 0, 0, 0, 0
    
    # Test each test image against all picked images
    print('===== Testing images against thresholds')
    for k in range(10):  # For each place
        test_img_feature = img_descriptors_matrix[k, test_set[k]]
        test_img_feature = test_img_feature / np.linalg.norm(test_img_feature, axis=1, keepdims=True)
        
        for i in range(10):  # Against each place
            for j in picked_set[i]:  # Against each picked image
                # Get threshold values for this image from places_matrix
                mean_bad_scores = places_matrix[i][j].mean_bad_scores
                std_dev_bad_scores = places_matrix[i][j].std_dev_bad_scores
                filter_n = places_matrix[i][j].filter_n  # Use the pre-calculated filter_n value
                
                # Compare test image against this picked image
                matching_img_feature = img_descriptors_matrix[i, j]
                matching_img_feature = matching_img_feature / np.linalg.norm(matching_img_feature, axis=1, keepdims=True)
                S = np.matmul(test_img_feature, matching_img_feature.transpose())
                score = S[0][0]
                
                # Calculate threshold using the pre-calculated filter_n value
                threshold = mean_bad_scores + (filter_n * std_dev_bad_scores)
                
                # Determine if accepted or rejected
                if score < threshold:
                    print(f'Test image p{k}/i{test_set[k]} is rejected when compared to p{i}/i{j}. Score: {score:.6f}, Threshold: {threshold:.6f}, filter_n: {filter_n}')
                    if i == k:  # Same place
                        FN += 1  # False Negative
                    else:  # Different place
                        TN += 1  # True Negative
                else:
                    print(f'Test image p{k}/i{test_set[k]} is accepted when compared to p{i}/i{j}. Score: {score:.6f}, Threshold: {threshold:.6f}, filter_n: {filter_n}')
                    if i == k:  # Same place
                        TP += 1  # True Positive
                    else:  # Different place
                        FP += 1  # False Positive
    
    # Print results
    print("\n===== TEST RESULTS =====")
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Negatives (FN): {FN}")
    
    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

if __name__ == "__main__":
    test_with_thresholds() 