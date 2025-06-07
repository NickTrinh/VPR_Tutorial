import numpy as np
import csv
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

from config import DatasetConfig, ExperimentConfig, get_dataset_config, auto_detect_dataset_structure
from data_utils import DatasetLoader, ResultsManager, validate_dataset_structure

@dataclass
class ScoresStruct:
    good_scores: List[float]
    bad_scores: List[float]
    mean_bad_scores: float
    std_dev_bad_scores: float
    filter_n: int

class VPRExperiment:
    """Main class for running VPR experiments on different datasets"""
    
    def __init__(self, dataset_config: DatasetConfig, experiment_config: ExperimentConfig, use_cache: bool = True):
        self.dataset_config = dataset_config
        self.experiment_config = experiment_config
        self.data_loader = DatasetLoader(dataset_config, use_cache=use_cache)
        self.results_manager = ResultsManager(dataset_config.name, experiment_config.output_dir)
    
    def calculate_scores_for_image(self, img_i: int, img_j: int, 
                                   descriptors_matrix: np.ndarray, 
                                   picked_set: List[List[int]], 
                                   test_set: List[int]) -> ScoresStruct:
        """Calculate scores for a specific image against all other images"""
        good_scores = []
        bad_scores = []
        
        this_img_feature = descriptors_matrix[img_i, img_j]
        this_img_feature = this_img_feature / np.linalg.norm(this_img_feature, axis=1, keepdims=True)
        
        print(f'Computing cosine similarities for p{img_i}/i{img_j} against all other images')
        
        for i in range(self.dataset_config.num_places):
            test_img_feature = descriptors_matrix[i, test_set[i]]
            test_img_feature = test_img_feature / np.linalg.norm(test_img_feature, axis=1, keepdims=True)
            
            S = np.matmul(this_img_feature, test_img_feature.transpose())
            
            if i == img_i:  # Same place, different image
                good_scores.append(S[0][0])
            else:  # Different place
                bad_scores.append(S[0][0])
        
        print(f'Calculating statistics for p{img_i}/i{img_j}')
        mean_bad_scores = np.mean(bad_scores)
        std_dev_bad_scores = np.std(bad_scores)
        
        # Calculate filter_n using the original method
        filter_n = 0
        if good_scores:
            min_good_score = min(good_scores)
            i = 1
            while True:
                threshold = mean_bad_scores + (i * std_dev_bad_scores)
                
                if min_good_score <= threshold:
                    filter_n = i - 1
                    break
                
                i += 1
                if i > 100:  # Safety break
                    filter_n = 100
                    break
        
        return ScoresStruct(good_scores, bad_scores, mean_bad_scores, std_dev_bad_scores, filter_n)
    
    def run_single_experiment(self, run_number: int) -> Dict[str, Dict[str, float]]:
        """Run a single experiment with random train/test split"""
        print(f'\n===== Running experiment {run_number} on {self.dataset_config.name} =====')
        
        # Load descriptors for all images
        descriptors_matrix = self.data_loader.load_all_descriptors()
        
        # Create train/test split
        picked_set, test_set = self.data_loader.create_train_test_split(
            random_state=self.experiment_config.random_seed + run_number
        )
        
        # Calculate scores for picked images only
        all_scores = {}
        
        for i in range(self.dataset_config.num_places):
            for j in picked_set[i]:
                print(f'Computing scores for p{i}/i{j}')
                scores = self.calculate_scores_for_image(i, j, descriptors_matrix, picked_set, test_set)
                img_key = f'p{i}/i{j}'
                all_scores[img_key] = {
                    'mean_bad_scores': scores.mean_bad_scores,
                    'std_dev_bad_scores': scores.std_dev_bad_scores,
                    'filter_n': scores.filter_n
                }
        
        # Save results for this run
        self.save_run_results(run_number, all_scores)
        
        return all_scores
    
    def run_multiple_experiments(self) -> Tuple[Dict, Dict]:
        """Run multiple experiments and calculate averages"""
        print(f'Running {self.experiment_config.num_runs} experiments on {self.dataset_config.name}')
        
        all_runs_results = []
        
        for run_num in range(1, self.experiment_config.num_runs + 1):
            run_result = self.run_single_experiment(run_num)
            all_runs_results.append(run_result)
        
        # Calculate and save averages
        image_averages, place_averages = self.calculate_and_save_averages(all_runs_results)
        
        return image_averages, place_averages
    
    def save_run_results(self, run_number: int, results: Dict[str, Dict[str, float]]):
        """Save results from a single run to CSV"""
        filename = self.results_manager.get_run_filename(run_number)
        
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
    
    def calculate_and_save_averages(self, all_results: List[Dict]) -> Tuple[Dict, Dict]:
        """Calculate and save average scores across all runs"""
        # Aggregate results
        aggregated_results = {}
        
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
        
        # Calculate image-level averages
        image_averages = {}
        for img_key, scores_list in aggregated_results.items():
            image_averages[img_key] = {
                'mean_bad_scores': np.mean(scores_list['mean_bad_scores']),
                'std_dev_bad_scores': np.mean(scores_list['std_dev_bad_scores']),
                'filter_n': np.mean(scores_list['filter_n'])
            }
        
        # Calculate place-level averages
        place_data = {}
        for img_key, result in image_averages.items():
            place = img_key.split('/')[0]  # Extract place (e.g., 'p0')
            if place not in place_data:
                place_data[place] = {
                    'mean_bad_scores': [],
                    'std_dev_bad_scores': [],
                    'filter_n': []
                }
            
            place_data[place]['mean_bad_scores'].append(result['mean_bad_scores'])
            place_data[place]['std_dev_bad_scores'].append(result['std_dev_bad_scores'])
            place_data[place]['filter_n'].append(result['filter_n'])
        
        place_averages = {}
        for place, data in place_data.items():
            place_averages[place] = {
                'mean_bad_scores': np.mean(data['mean_bad_scores']),
                'std_dev_bad_scores': np.mean(data['std_dev_bad_scores']),
                'filter_n': np.mean(data['filter_n'])
            }
        
        # Save averages
        self.save_image_averages(image_averages)
        self.save_place_averages(place_averages)
        
        return image_averages, place_averages
    
    def save_image_averages(self, averages: Dict):
        """Save image-level averages to CSV"""
        filename = self.results_manager.get_image_averages_filename()
        
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Image', 'Mean Bad Scores', 'Std Deviation Bad Scores', 'Filter N'])
            
            sorted_keys = sorted(averages.keys(),
                                key=lambda x: (int(x.split('/')[0][1:]), int(x.split('/')[1][1:])))
            
            for img_key in sorted_keys:
                result = averages[img_key]
                writer.writerow([
                    img_key,
                    result['mean_bad_scores'],
                    result['std_dev_bad_scores'],
                    result['filter_n']
                ])
    
    def save_place_averages(self, averages: Dict):
        """Save place-level averages to CSV"""
        filename = self.results_manager.get_place_averages_filename()
        
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Place', 'Mean Bad Scores', 'Std Deviation Bad Scores', 'Filter N'])
            
            for place in sorted(averages.keys(), key=lambda x: int(x[1:])):
                result = averages[place]
                writer.writerow([
                    place,
                    result['mean_bad_scores'],
                    result['std_dev_bad_scores'],
                    result['filter_n']
                ])

def run_experiment_on_dataset(dataset_name: str, experiment_config: ExperimentConfig = None, use_cache: bool = True):
    """Run complete experiment on a specific dataset"""
    if experiment_config is None:
        from config import DEFAULT_EXPERIMENT
        experiment_config = DEFAULT_EXPERIMENT
    
    # Get and validate dataset configuration
    dataset_config = get_dataset_config(dataset_name)
    dataset_config = auto_detect_dataset_structure(dataset_config)
    
    if not validate_dataset_structure(dataset_config):
        raise ValueError(f"Dataset structure validation failed for {dataset_name}")
    
    print(f"Running experiment on {dataset_config.name}")
    print(f"Dataset: {dataset_config.description}")
    print(f"Places: {dataset_config.num_places}, Images per place: {dataset_config.images_per_place}")
    
    # Create and run experiment
    experiment = VPRExperiment(dataset_config, experiment_config, use_cache=use_cache)
    image_averages, place_averages = experiment.run_multiple_experiments()
    
    print(f"\nExperiment completed for {dataset_name}")
    print(f"Results saved to: {experiment.results_manager.dataset_dir}")
    
    return image_averages, place_averages

if __name__ == "__main__":
    # Example: Run experiment on Fordham Places dataset
    run_experiment_on_dataset("fordham_places") 