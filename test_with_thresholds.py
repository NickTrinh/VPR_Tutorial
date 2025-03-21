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

def setup_test_data():
    """
    Perform the common setup tasks needed for testing:
    - Randomly select test/picked images
    - Calculate descriptors for all images
    
    Returns:
        tuple: (img_descriptors_matrix, picked_set, test_set)
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
    print('===== Computing descriptors for all images')
    for i in range(10):
        for j in range(3):
            print(f'===== Computing descriptors for p{i}/i{j}')
            img_descriptor = get_descriptor(destination + f'p{i}/i{j}' + '/*.jpg')
            img_descriptors_matrix[i, j] = img_descriptor
    
    return img_descriptors_matrix, picked_set, test_set

def test_with_place_thresholds(img_descriptors_matrix, picked_set, test_set):
    """
    Run a test using the place-level threshold values from places_data.py
    to calculate TP, FP, TN, FN.
    
    Args:
        img_descriptors_matrix: Matrix of image descriptors
        picked_set: List of picked image indices for each place
        test_set: List of test image indices for each place
        
    Returns:
        dict: Test results with TP, FP, TN, FN and metrics
    """
    # Initialize counters for TP, FP, TN, FN
    TP, FP, TN, FN = 0, 0, 0, 0
    
    # Test each test image against all picked images
    print('===== Testing images against place-level thresholds')
    for k in range(10):  # For each place
        test_img_feature = img_descriptors_matrix[k, test_set[k]]
        test_img_feature = test_img_feature / np.linalg.norm(test_img_feature, axis=1, keepdims=True)
        
        for i in range(10):  # Against each place
            # Get threshold values for this place from places array (place-level)
            mean_bad_scores = places[i].mean_bad_scores
            std_dev_bad_scores = places[i].std_dev_bad_scores
            filter_n = places[i].filter_n  # Use the pre-calculated place-level filter_n value
            
            # Calculate threshold using the place-level filter_n value
            threshold = mean_bad_scores + (filter_n * std_dev_bad_scores)
            
            for j in picked_set[i]:  # Against each picked image
                # Compare test image against this picked image
                matching_img_feature = img_descriptors_matrix[i, j]
                matching_img_feature = matching_img_feature / np.linalg.norm(matching_img_feature, axis=1, keepdims=True)
                S = np.matmul(test_img_feature, matching_img_feature.transpose())
                score = S[0][0]
                
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
    
    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Prepare results dictionary
    results = {
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1_score': f1_score
    }
    
    return results

def test_with_image_thresholds(img_descriptors_matrix, picked_set, test_set):
    """
    Run a test using the image-level threshold values from places_data.py
    to calculate TP, FP, TN, FN without recalculating filter_n.
    
    Args:
        img_descriptors_matrix: Matrix of image descriptors
        picked_set: List of picked image indices for each place
        test_set: List of test image indices for each place
        
    Returns:
        dict: Test results with TP, FP, TN, FN and metrics
    """
    # Initialize counters for TP, FP, TN, FN
    TP, FP, TN, FN = 0, 0, 0, 0
    
    # Test each test image against all picked images
    print('===== Testing images against image-level thresholds')
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
    
    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Prepare results dictionary
    results = {
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1_score': f1_score
    }
    
    return results

def print_results(results, test_type):
    """Print formatted test results"""
    print(f"\n===== TEST RESULTS ({test_type}) =====")
    print(f"True Positives (TP): {results['TP']}")
    print(f"False Positives (FP): {results['FP']}")
    print(f"True Negatives (TN): {results['TN']}")
    print(f"False Negatives (FN): {results['FN']}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")

if __name__ == "__main__":
    # Setup test data once
    img_descriptors_matrix, picked_set, test_set = setup_test_data()
    
    # Run both tests with the same data
    print("\n===== TESTING WITH PLACE-LEVEL THRESHOLDS =====")
    place_results = test_with_place_thresholds(img_descriptors_matrix, picked_set, test_set)
    print_results(place_results, "PLACE-LEVEL THRESHOLDS")
    
    print("\n===== TESTING WITH IMAGE-LEVEL THRESHOLDS =====")
    image_results = test_with_image_thresholds(img_descriptors_matrix, picked_set, test_set)
    print_results(image_results, "IMAGE-LEVEL THRESHOLDS")
    