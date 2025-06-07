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
        
        # Calculate filter_n using selected method
        if self.experiment_config.threshold_method == "original":
            # Original method
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
        
        elif self.experiment_config.threshold_method == "ensemble":
            filter_n = self.calculate_ensemble_threshold(good_scores, bad_scores)
        
        else:
            # Use alternative or advanced methods
            alt_thresholds = self.calculate_alternative_thresholds(good_scores, bad_scores)
            adv_thresholds = self.calculate_advanced_thresholds(good_scores, bad_scores)
            precision_thresholds = self.calculate_precision_focused_thresholds(good_scores, bad_scores)
            balanced_thresholds = self.calculate_balanced_thresholds(good_scores, bad_scores)
            all_methods = {**alt_thresholds, **adv_thresholds, **precision_thresholds, **balanced_thresholds}
            
            if self.experiment_config.threshold_method in all_methods:
                threshold_value = all_methods[self.experiment_config.threshold_method]
                # Convert to filter_n format
                if std_dev_bad_scores > 0:
                    filter_n = (threshold_value - mean_bad_scores) / std_dev_bad_scores
                else:
                    filter_n = 1.0
            else:
                # Fallback to original method
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
        
        # Apply threshold multiplier for tuning
        filter_n = filter_n * self.experiment_config.threshold_multiplier
        
        return ScoresStruct(good_scores, bad_scores, mean_bad_scores, std_dev_bad_scores, filter_n)
    
    def calculate_alternative_thresholds(self, good_scores: List[float], bad_scores: List[float]) -> Dict[str, float]:
        """Calculate alternative threshold strategies for comparison"""
        if not good_scores or not bad_scores:
            return {}
        
        results = {}
        mean_bad = np.mean(bad_scores)
        std_bad = np.std(bad_scores)
        
        # Strategy 1: Percentile-based (95th percentile of bad scores)
        results['percentile_95'] = np.percentile(bad_scores, 95)
        
        # Strategy 2: Median + 2*IQR of bad scores
        q75, q25 = np.percentile(bad_scores, [75, 25])
        iqr = q75 - q25
        results['median_2iqr'] = np.median(bad_scores) + 2 * iqr
        
        # Strategy 3: Halfway between bad and good score means
        mean_good = np.mean(good_scores)
        results['halfway'] = (mean_bad + mean_good) / 2
        
        # Strategy 4: Optimal F1 threshold (simple approximation)
        # Find threshold that maximizes F1 score
        thresholds = np.linspace(min(bad_scores), max(good_scores), 100)
        best_f1 = 0
        best_threshold = mean_bad + std_bad
        
        for thresh in thresholds:
            tp = sum(1 for score in good_scores if score >= thresh)
            fp = sum(1 for score in bad_scores if score >= thresh)
            fn = len(good_scores) - tp
            
            if tp + fp > 0 and tp + fn > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = thresh
        
        results['optimal_f1'] = best_threshold
        
        return results
    
    def calculate_advanced_thresholds(self, good_scores: List[float], bad_scores: List[float]) -> Dict[str, float]:
        """Calculate advanced threshold strategies"""
        if not good_scores or not bad_scores:
            return {}
        
        results = {}
        mean_bad = np.mean(bad_scores)
        std_bad = np.std(bad_scores)
        
        # Strategy 5: ROC-based optimal threshold (maximizes TPR - FPR)
        thresholds = np.linspace(min(bad_scores), max(good_scores), 200)
        best_roc = -1
        best_roc_threshold = mean_bad + std_bad
        
        for thresh in thresholds:
            tp = sum(1 for score in good_scores if score >= thresh)
            fp = sum(1 for score in bad_scores if score >= thresh)
            tn = len(bad_scores) - fp
            fn = len(good_scores) - tp
            
            if tp + fn > 0 and tn + fp > 0:
                tpr = tp / (tp + fn)  # Sensitivity/Recall
                fpr = fp / (fp + tn)  # False Positive Rate
                roc_metric = tpr - fpr  # Youden's J statistic
                
                if roc_metric > best_roc:
                    best_roc = roc_metric
                    best_roc_threshold = thresh
        
        results['youden_j'] = best_roc_threshold
        
        # Strategy 6: Cost-sensitive threshold (assuming FN is 2x worse than FP)
        fn_cost = 2.0  # Missing a true match is worse than false alarm
        fp_cost = 1.0
        best_cost = float('inf')
        best_cost_threshold = mean_bad + std_bad
        
        for thresh in thresholds:
            tp = sum(1 for score in good_scores if score >= thresh)
            fp = sum(1 for score in bad_scores if score >= thresh)
            fn = len(good_scores) - tp
            
            total_cost = fn * fn_cost + fp * fp_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_cost_threshold = thresh
        
        results['cost_sensitive'] = best_cost_threshold
        
        # Strategy 7: Statistical separation threshold
        # Find threshold at intersection of good/bad score distributions
        try:
            from scipy import stats
            
            # Fit normal distributions
            good_mean, good_std = np.mean(good_scores), np.std(good_scores)
            bad_mean, bad_std = np.mean(bad_scores), np.std(bad_scores)
            
            # Find intersection point of two normal distributions
            if good_std > 0 and bad_std > 0 and good_mean != bad_mean:
                a = 1/(2*good_std**2) - 1/(2*bad_std**2)
                b = bad_mean/(bad_std**2) - good_mean/(good_std**2)
                c = good_mean**2/(2*good_std**2) - bad_mean**2/(2*bad_std**2) - np.log(bad_std/good_std)
                
                if a != 0:
                    discriminant = b**2 - 4*a*c
                    if discriminant >= 0:
                        intersection1 = (-b + np.sqrt(discriminant)) / (2*a)
                        intersection2 = (-b - np.sqrt(discriminant)) / (2*a)
                        
                        # Choose intersection closest to middle of distributions
                        mid_point = (good_mean + bad_mean) / 2
                        if abs(intersection1 - mid_point) < abs(intersection2 - mid_point):
                            results['gaussian_intersection'] = intersection1
                        else:
                            results['gaussian_intersection'] = intersection2
        except ImportError:
            # Fallback if scipy not available
            results['gaussian_intersection'] = (np.mean(good_scores) + np.mean(bad_scores)) / 2
        except:
            results['gaussian_intersection'] = (np.mean(good_scores) + np.mean(bad_scores)) / 2
        
        # Strategy 8: Quantile-based threshold (where X% of good scores are above threshold)
        results['quantile_90'] = np.percentile(good_scores, 10)  # 90% of good scores above this
        results['quantile_80'] = np.percentile(good_scores, 20)  # 80% of good scores above this
        
        # Strategy 9: Otsu's method adaptation (minimize intra-class variance)
        all_scores = sorted(good_scores + bad_scores)
        labels = [1] * len(good_scores) + [0] * len(bad_scores)
        combined = list(zip(all_scores, labels))
        combined.sort()
        
        best_otsu_variance = float('inf')
        best_otsu_threshold = mean_bad + std_bad
        
        for i in range(1, len(combined) - 1):
            threshold = combined[i][0]
            
            # Split into two groups
            group1_scores = [s for s, l in combined if s < threshold]
            group2_scores = [s for s, l in combined if s >= threshold]
            
            if len(group1_scores) > 0 and len(group2_scores) > 0:
                var1 = np.var(group1_scores) * len(group1_scores)
                var2 = np.var(group2_scores) * len(group2_scores)
                weighted_variance = (var1 + var2) / len(combined)
                
                if weighted_variance < best_otsu_variance:
                    best_otsu_variance = weighted_variance
                    best_otsu_threshold = threshold
        
        results['otsu_adapted'] = best_otsu_threshold
        
        return results
    
    def calculate_precision_focused_thresholds(self, good_scores: List[float], bad_scores: List[float]) -> Dict[str, float]:
        """Calculate precision-focused threshold strategies to reduce false positives"""
        if not good_scores or not bad_scores:
            return {}
        
        results = {}
        thresholds = np.linspace(min(bad_scores), max(good_scores), 200)
        
        # Strategy 10: Target specific precision levels
        target_precisions = [0.8, 0.9, 0.95]  # 80%, 90%, 95% precision
        
        for target_prec in target_precisions:
            best_threshold = max(good_scores)  # Start with very conservative
            best_diff = float('inf')
            
            for thresh in thresholds:
                tp = sum(1 for score in good_scores if score >= thresh)
                fp = sum(1 for score in bad_scores if score >= thresh)
                
                if tp + fp > 0:
                    precision = tp / (tp + fp)
                    diff = abs(precision - target_prec)
                    
                    if diff < best_diff and precision >= target_prec:
                        best_diff = diff
                        best_threshold = thresh
            
            results[f'precision_{int(target_prec*100)}'] = best_threshold
        
        # Strategy 11: Conservative F1 (weight precision more heavily)
        best_weighted_f1 = 0
        best_conservative_threshold = max(good_scores)
        
        for thresh in thresholds:
            tp = sum(1 for score in good_scores if score >= thresh)
            fp = sum(1 for score in bad_scores if score >= thresh)
            fn = len(good_scores) - tp
            
            if tp + fp > 0 and tp + fn > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                
                # Weighted F1 favoring precision (beta < 1)
                beta = 0.5  # Precision weighted 2x more than recall
                if precision + recall > 0:
                    weighted_f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
                    
                    if weighted_f1 > best_weighted_f1:
                        best_weighted_f1 = weighted_f1
                        best_conservative_threshold = thresh
        
        results['conservative_f1'] = best_conservative_threshold
        
        # Strategy 12: False Positive Rate constraint (max 5% FPR)
        max_fpr = 0.05
        best_fpr_threshold = max(good_scores)
        best_recall_at_fpr = 0
        
        for thresh in thresholds:
            tp = sum(1 for score in good_scores if score >= thresh)
            fp = sum(1 for score in bad_scores if score >= thresh)
            tn = len(bad_scores) - fp
            fn = len(good_scores) - tp
            
            if tn + fp > 0 and tp + fn > 0:
                fpr = fp / (fp + tn)
                recall = tp / (tp + fn)
                
                if fpr <= max_fpr and recall > best_recall_at_fpr:
                    best_recall_at_fpr = recall
                    best_fpr_threshold = thresh
        
        results['max_fpr_5'] = best_fpr_threshold
        
        # Strategy 13: Percentile of bad scores (more conservative)
        results['percentile_99'] = np.percentile(bad_scores, 99)  # Only top 1% of bad scores accepted
        results['percentile_99_5'] = np.percentile(bad_scores, 99.5)  # Only top 0.5% accepted
        
        # Strategy 14: Mean + N standard deviations (more conservative)
        mean_bad = np.mean(bad_scores)
        std_bad = np.std(bad_scores)
        results['mean_plus_2std'] = mean_bad + 2 * std_bad
        results['mean_plus_3std'] = mean_bad + 3 * std_bad
        
        return results
    
    def calculate_balanced_thresholds(self, good_scores: List[float], bad_scores: List[float]) -> Dict[str, float]:
        """Calculate balanced thresholds targeting specific precision-recall trade-offs"""
        if not good_scores or not bad_scores:
            return {}
        
        results = {}
        thresholds = np.linspace(min(bad_scores), max(good_scores), 200)
        
        # Strategy 15: Balanced F-beta scores
        beta_values = [0.5, 0.75, 1.25, 1.5, 2.0]  # Different precision/recall weights
        
        for beta in beta_values:
            best_fbeta = 0
            best_fbeta_threshold = np.mean(bad_scores) + np.std(bad_scores)
            
            for thresh in thresholds:
                tp = sum(1 for score in good_scores if score >= thresh)
                fp = sum(1 for score in bad_scores if score >= thresh)
                fn = len(good_scores) - tp
                
                if tp + fp > 0 and tp + fn > 0:
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    
                    if precision + recall > 0:
                        fbeta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
                        
                        if fbeta > best_fbeta:
                            best_fbeta = fbeta
                            best_fbeta_threshold = thresh
            
            # Name based on what it emphasizes
            if beta < 1:
                name = f'precision_focused_{beta}'.replace('.', '_')
            elif beta > 1:
                name = f'recall_focused_{beta}'.replace('.', '_')
            else:
                name = 'balanced_f1'
            
            results[name] = best_fbeta_threshold
        
        return results
    
    def calculate_ensemble_threshold(self, good_scores: List[float], bad_scores: List[float]) -> float:
        """Calculate ensemble threshold by averaging multiple methods"""
        if not good_scores or not bad_scores:
            return np.mean(bad_scores) + np.std(bad_scores)
        
        # Get all threshold methods
        alt_thresholds = self.calculate_alternative_thresholds(good_scores, bad_scores)
        adv_thresholds = self.calculate_advanced_thresholds(good_scores, bad_scores)
        precision_thresholds = self.calculate_precision_focused_thresholds(good_scores, bad_scores)
        balanced_thresholds = self.calculate_balanced_thresholds(good_scores, bad_scores)
        
        # Combine all thresholds
        all_thresholds = []
        for method_dict in [alt_thresholds, adv_thresholds, precision_thresholds, balanced_thresholds]:
            all_thresholds.extend(method_dict.values())
        
        if all_thresholds:
            # Use median to avoid outlier influence
            ensemble_threshold = np.median(all_thresholds)
            
            # Convert back to filter_n format
            mean_bad = np.mean(bad_scores)
            std_bad = np.std(bad_scores)
            
            if std_bad > 0:
                return (ensemble_threshold - mean_bad) / std_bad
            else:
                return 1.0
        
        return 1.0
    
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