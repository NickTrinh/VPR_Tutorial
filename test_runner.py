import numpy as np
import csv
from typing import Dict, List, Tuple
from dataclasses import dataclass

from config import DatasetConfig, get_dataset_config, auto_detect_dataset_structure
from data_utils import DatasetLoader, ResultsManager, validate_dataset_structure

@dataclass
class Place:
    mean_bad_scores: float
    std_dev_bad_scores: float
    filter_n: float

@dataclass
class TestResults:
    TP: int
    FP: int
    TN: int
    FN: int
    precision: float
    recall: float
    accuracy: float
    f1_score: float

class ThresholdDataLoader:
    """Load threshold data from CSV files"""
    
    def __init__(self, results_manager: ResultsManager):
        self.results_manager = results_manager
    
    def load_place_thresholds(self) -> List[Place]:
        """Load place-level threshold data"""
        filename = self.results_manager.get_place_averages_filename()
        places = []
        
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                place = Place(
                    mean_bad_scores=float(row['Mean Bad Scores']),
                    std_dev_bad_scores=float(row['Std Deviation Bad Scores']),
                    filter_n=float(row['Filter N'])
                )
                places.append(place)
        
        return places
    
    def load_image_thresholds(self, dataset_config: DatasetConfig) -> List[List[Place]]:
        """Load image-level threshold data"""
        filename = self.results_manager.get_image_averages_filename()
        
        # Initialize matrix
        places_matrix = [[None for _ in range(dataset_config.images_per_place)] 
                        for _ in range(dataset_config.num_places)]
        
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                img_key = row['Image']
                place_idx = int(img_key.split('/')[0][1:])  # Extract place index from 'p0'
                img_idx = int(img_key.split('/')[1][1:])    # Extract image index from 'i0'
                
                place = Place(
                    mean_bad_scores=float(row['Mean Bad Scores']),
                    std_dev_bad_scores=float(row['Std Deviation Bad Scores']),
                    filter_n=float(row['Filter N'])
                )
                places_matrix[place_idx][img_idx] = place
        
        return places_matrix

class VPRTester:
    """Main class for testing VPR performance on different datasets"""
    
    def __init__(self, dataset_config: DatasetConfig, use_cache: bool = True):
        self.dataset_config = dataset_config
        self.data_loader = DatasetLoader(dataset_config, use_cache=use_cache)
        self.results_manager = ResultsManager(dataset_config.name)
        self.threshold_loader = ThresholdDataLoader(self.results_manager)
    
    def setup_test_data(self, random_state: int = None) -> Tuple[np.ndarray, List[List[int]], List[int]]:
        """Setup test data with train/test split and load descriptors"""
        print(f'Setting up test data for {self.dataset_config.name}')
        
        # Load all descriptors
        descriptors_matrix = self.data_loader.load_all_descriptors()
        
        # Create train/test split
        picked_set, test_set = self.data_loader.create_train_test_split(random_state)
        
        return descriptors_matrix, picked_set, test_set
    
    def test_with_place_thresholds(self, descriptors_matrix: np.ndarray, 
                                   picked_set: List[List[int]], 
                                   test_set: List[int]) -> TestResults:
        """Test using place-level thresholds"""
        print('Testing with place-level thresholds')
        
        # Load place-level thresholds
        places = self.threshold_loader.load_place_thresholds()
        
        TP, FP, TN, FN = 0, 0, 0, 0
        
        for k in range(self.dataset_config.num_places):
            test_img_feature = descriptors_matrix[k, test_set[k]]
            test_img_feature = test_img_feature / np.linalg.norm(test_img_feature, axis=1, keepdims=True)
            
            for i in range(self.dataset_config.num_places):
                place_thresholds = places[i]
                threshold = (place_thresholds.mean_bad_scores + 
                           (place_thresholds.filter_n * place_thresholds.std_dev_bad_scores))
                
                for j in picked_set[i]:
                    matching_img_feature = descriptors_matrix[i, j]
                    matching_img_feature = matching_img_feature / np.linalg.norm(matching_img_feature, axis=1, keepdims=True)
                    
                    S = np.matmul(test_img_feature, matching_img_feature.transpose())
                    score = S[0][0]
                    
                    if score < threshold:
                        print(f'Test p{k}/i{test_set[k]} rejected vs p{i}/i{j}. Score: {score:.6f}, Threshold: {threshold:.6f}')
                        if i == k:
                            FN += 1
                        else:
                            TN += 1
                    else:
                        print(f'Test p{k}/i{test_set[k]} accepted vs p{i}/i{j}. Score: {score:.6f}, Threshold: {threshold:.6f}')
                        if i == k:
                            TP += 1
                        else:
                            FP += 1
        
        return self._calculate_metrics(TP, FP, TN, FN)
    
    def test_with_image_thresholds(self, descriptors_matrix: np.ndarray, 
                                   picked_set: List[List[int]], 
                                   test_set: List[int]) -> TestResults:
        """Test using image-level thresholds"""
        print('Testing with image-level thresholds')
        
        # Load image-level thresholds
        places_matrix = self.threshold_loader.load_image_thresholds(self.dataset_config)
        
        TP, FP, TN, FN = 0, 0, 0, 0
        
        for k in range(self.dataset_config.num_places):
            test_img_feature = descriptors_matrix[k, test_set[k]]
            test_img_feature = test_img_feature / np.linalg.norm(test_img_feature, axis=1, keepdims=True)
            
            for i in range(self.dataset_config.num_places):
                for j in picked_set[i]:
                    img_thresholds = places_matrix[i][j]
                    
                    # If image-level threshold data is missing, assign 0 values
                    if img_thresholds is None:
                        print(f'Warning: No image-level threshold for p{i}/i{j}, using 0 threshold')
                        img_thresholds = Place(mean_bad_scores=0.0, std_dev_bad_scores=0.0, filter_n=0.0)
                    
                    threshold = (img_thresholds.mean_bad_scores + 
                               (img_thresholds.filter_n * img_thresholds.std_dev_bad_scores))
                    
                    matching_img_feature = descriptors_matrix[i, j]
                    matching_img_feature = matching_img_feature / np.linalg.norm(matching_img_feature, axis=1, keepdims=True)
                    
                    S = np.matmul(test_img_feature, matching_img_feature.transpose())
                    score = S[0][0]
                    
                    if score < threshold:
                        print(f'Test p{k}/i{test_set[k]} rejected vs p{i}/i{j}. Score: {score:.6f}, Threshold: {threshold:.6f}')
                        if i == k:
                            FN += 1
                        else:
                            TN += 1
                    else:
                        print(f'Test p{k}/i{test_set[k]} accepted vs p{i}/i{j}. Score: {score:.6f}, Threshold: {threshold:.6f}')
                        if i == k:
                            TP += 1
                        else:
                            FP += 1
        
        return self._calculate_metrics(TP, FP, TN, FN)
    
    def _calculate_metrics(self, TP: int, FP: int, TN: int, FN: int) -> TestResults:
        """Calculate performance metrics"""
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return TestResults(TP, FP, TN, FN, precision, recall, accuracy, f1_score)
    
    def save_test_results(self, place_results: TestResults, image_results: TestResults):
        """Save test results to CSV"""
        filename = self.results_manager.get_final_results_filename()
        
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Test Type', 'TP', 'FP', 'TN', 'FN', 'Precision', 'Recall', 'Accuracy', 'F1 Score'])
            
            writer.writerow([
                'Place-level Thresholds',
                place_results.TP, place_results.FP, place_results.TN, place_results.FN,
                place_results.precision, place_results.recall, place_results.accuracy, place_results.f1_score
            ])
            
            writer.writerow([
                'Image-level Thresholds',
                image_results.TP, image_results.FP, image_results.TN, image_results.FN,
                image_results.precision, image_results.recall, image_results.accuracy, image_results.f1_score
            ])
    
    def run_full_test(self, random_state: int = None) -> Tuple[TestResults, TestResults]:
        """Run complete test with both place-level and image-level thresholds"""
        print(f'Running full test on {self.dataset_config.name}')
        
        # Setup test data
        descriptors_matrix, picked_set, test_set = self.setup_test_data(random_state)
        
        # Run tests
        place_results = self.test_with_place_thresholds(descriptors_matrix, picked_set, test_set)
        image_results = self.test_with_image_thresholds(descriptors_matrix, picked_set, test_set)
        
        # Save results
        self.save_test_results(place_results, image_results)
        
        return place_results, image_results

def print_test_results(results: TestResults, test_type: str):
    """Print formatted test results"""
    print(f"\n===== TEST RESULTS ({test_type}) =====")
    print(f"True Positives (TP): {results.TP}")
    print(f"False Positives (FP): {results.FP}")
    print(f"True Negatives (TN): {results.TN}")
    print(f"False Negatives (FN): {results.FN}")
    print(f"Precision: {results.precision:.4f}")
    print(f"Recall: {results.recall:.4f}")
    print(f"Accuracy: {results.accuracy:.4f}")
    print(f"F1 Score: {results.f1_score:.4f}")

def test_dataset(dataset_name: str, random_state: int = None, use_cache: bool = True):
    """Test a specific dataset"""
    # Get and validate dataset configuration
    dataset_config = get_dataset_config(dataset_name)
    if dataset_config.format == 'landmark':
        dataset_config = auto_detect_dataset_structure(dataset_config)
    
    if dataset_config.format == 'landmark' and not validate_dataset_structure(dataset_config):
        raise ValueError(f"Dataset structure validation failed for {dataset_name}")
    
    print(f"Testing {dataset_config.name}")
    print(f"Dataset: {dataset_config.description}")
    
    # Create and run test
    tester = VPRTester(dataset_config, use_cache=use_cache)
    place_results, image_results = tester.run_full_test(random_state)
    
    # Print results
    print_test_results(place_results, "PLACE-LEVEL THRESHOLDS")
    print_test_results(image_results, "IMAGE-LEVEL THRESHOLDS")
    
    print(f"\nTest completed for {dataset_name}")
    print(f"Results saved to: {tester.results_manager.get_final_results_filename()}")
    
    return place_results, image_results

if __name__ == "__main__":
    # Example: Test Fordham Places dataset
    test_dataset("fordham_places") 