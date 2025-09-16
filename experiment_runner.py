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
        self.data_loader = DatasetLoader(dataset_config, use_cache=use_cache, descriptor_name=experiment_config.descriptor)
        # Save results under results/<DatasetName>/<descriptor>/
        descriptor_subdir = os.path.join(dataset_config.name, experiment_config.descriptor)
        self.results_manager = ResultsManager(descriptor_subdir, experiment_config.output_dir)
    
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
        
        # Calculate filter_n using original method
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
    

    

    

    
    def run_single_experiment(self, run_number: int) -> Dict[str, Dict[str, float]]:
        """Run a single experiment with random train/test split"""
        print(f'\n===== Running experiment {run_number} on {self.dataset_config.name} =====')
        
        # Load descriptors for all images
        descriptors_matrix = self.data_loader.load_all_descriptors()
        
        # Create train/test split
        picked_set, test_set = self.data_loader.create_train_test_split(
            random_state=self.experiment_config.random_seed + run_number
        )
        
        # Calculate scores for picked images only (vectorized across all images)
        all_scores = self._calculate_scores_vectorized(descriptors_matrix, picked_set, test_set)
        
        # Save results for this run
        self.save_run_results(run_number, all_scores)
        
        return all_scores

    def _calculate_scores_vectorized(
        self,
        descriptors_matrix: np.ndarray,
        picked_set: List[List[int]],
        test_set: List[int],
        batch_size: int = 4096,
    ) -> Dict[str, Dict[str, float]]:
        """Vectorized computation of per-image mean_bad, std_bad, and filter_n.

        This replaces the nested Python loops with a single matrix multiply (optionally on GPU)
        and vectorized statistics, providing large speedups on big datasets.
        """
        # Build test matrix (one test image per place)
        test_features_list: List[np.ndarray] = []
        for place_index in range(self.dataset_config.num_places):
            test_img_index = test_set[place_index]
            feat = descriptors_matrix[place_index, test_img_index]
            feat = feat.reshape(-1) if feat.ndim > 1 else feat
            test_features_list.append(feat)
        test_matrix = np.stack(test_features_list, axis=0).astype(np.float32)
        # L2 normalize rows
        test_norms = np.linalg.norm(test_matrix, axis=1, keepdims=True) + 1e-12
        test_matrix = test_matrix / test_norms

        # Build train matrix (all picked images) and bookkeeping
        train_features_list: List[np.ndarray] = []
        train_place_indices: List[int] = []
        image_keys: List[str] = []
        for place_index in range(self.dataset_config.num_places):
            for img_idx in picked_set[place_index]:
                feat = descriptors_matrix[place_index, img_idx]
                feat = feat.reshape(-1) if feat.ndim > 1 else feat
                train_features_list.append(feat)
                train_place_indices.append(place_index)
                image_keys.append(f'p{place_index}/i{img_idx}')
        if not train_features_list:
            return {}

        train_matrix = np.stack(train_features_list, axis=0).astype(np.float32)
        train_norms = np.linalg.norm(train_matrix, axis=1, keepdims=True) + 1e-12
        train_matrix = train_matrix / train_norms

        num_train = train_matrix.shape[0]
        num_places = test_matrix.shape[0]
        num_bad = max(num_places - 1, 1)

        # Prepare outputs
        mean_bad_all = np.empty((num_train,), dtype=np.float32)
        std_bad_all = np.empty((num_train,), dtype=np.float32)
        filter_n_all = np.empty((num_train,), dtype=np.float32)

        # Try GPU via PyTorch for big speedups; fallback to NumPy if not available
        use_torch = False
        try:
            import torch  # type: ignore
            use_torch = torch.cuda.is_available()
        except Exception:
            use_torch = False

        if use_torch:
            import torch  # type: ignore
            # Enable fast matmul on modern GPUs
            try:
                torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
                if hasattr(torch, 'set_float32_matmul_precision'):
                    torch.set_float32_matmul_precision("high")  # type: ignore
            except Exception:
                pass
            device = torch.device('cuda')
            with torch.inference_mode():
                test_t = torch.from_numpy(test_matrix).to(device, non_blocking=True)
                # Try to keep the full train matrix on GPU for best throughput
                train_t = None
                try:
                    train_t = torch.from_numpy(train_matrix).to(device, non_blocking=True)
                except Exception:
                    # Fallback: keep on CPU and transfer per batch (slower)
                    train_cpu = torch.from_numpy(train_matrix).pin_memory()
                place_idx_all = torch.tensor(train_place_indices, device=device, dtype=torch.long)
                for start in range(0, num_train, batch_size):
                    end = min(start + batch_size, num_train)
                    if train_t is not None:
                        batch = train_t[start:end]
                    else:
                        batch = train_cpu[start:end].to(device, non_blocking=True)
                    # Similarity matrix for this batch: (B, P)
                    s_batch = torch.matmul(batch, test_t.T)
                    # Row stats
                    sum_all = torch.sum(s_batch, dim=1)
                    sumsq_all = torch.sum(s_batch * s_batch, dim=1)
                    # Gather good score at the correct place column per row
                    good = s_batch[torch.arange(end - start, device=device), place_idx_all[start:end]]
                    mean_bad = (sum_all - good) / num_bad
                    var_bad = (sumsq_all - good * good) / num_bad - mean_bad * mean_bad
                    var_bad = torch.clamp(var_bad, min=0.0)
                    std_bad = torch.sqrt(var_bad + 1e-12)
                    # Original filter_n approximation (vectorized)
                    raw_n = (good - mean_bad) / (std_bad + 1e-12)
                    raw_n = torch.floor(raw_n)
                    raw_n = torch.clamp(raw_n, min=0.0, max=100.0)
                    mean_bad_all[start:end] = mean_bad.detach().cpu().numpy().astype(np.float32)
                    std_bad_all[start:end] = std_bad.detach().cpu().numpy().astype(np.float32)
                    filter_n_all[start:end] = raw_n.detach().cpu().numpy().astype(np.float32)
                    # Free only local references; avoid empty_cache()
                    del batch, s_batch, sum_all, sumsq_all, good, mean_bad, var_bad, std_bad, raw_n
        else:
            test_t = test_matrix.T  # (D, P)
            for start in range(0, num_train, batch_size):
                end = min(start + batch_size, num_train)
                batch = train_matrix[start:end]  # (B, D)
                s_batch = np.matmul(batch, test_t)  # (B, P)
                sum_all = np.sum(s_batch, axis=1)
                sumsq_all = np.sum(s_batch * s_batch, axis=1)
                good = s_batch[np.arange(end - start), np.array(train_place_indices[start:end])]
                mean_bad = (sum_all - good) / num_bad
                var_bad = (sumsq_all - good * good) / num_bad - mean_bad * mean_bad
                var_bad[var_bad < 0.0] = 0.0
                std_bad = np.sqrt(var_bad + 1e-12)
                raw_n = np.floor((good - mean_bad) / (std_bad + 1e-12))
                raw_n = np.clip(raw_n, 0.0, 100.0)
                mean_bad_all[start:end] = mean_bad.astype(np.float32)
                std_bad_all[start:end] = std_bad.astype(np.float32)
                filter_n_all[start:end] = raw_n.astype(np.float32)

        # Assemble the expected results dict
        all_scores: Dict[str, Dict[str, float]] = {}
        for idx, img_key in enumerate(image_keys):
            all_scores[img_key] = {
                'mean_bad_scores': float(mean_bad_all[idx]),
                'std_dev_bad_scores': float(std_bad_all[idx]),
                'filter_n': float(filter_n_all[idx])
            }
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
        for img_key, result in aggregated_results.items():
            place = img_key.split('/')[0]  # Extract place (e.g., 'p0')
            if place not in place_data:
                place_data[place] = {
                    'mean_bad_scores': [],
                    'std_dev_bad_scores': [],
                    'filter_n': []
                }
            
            # Note: We are aggregating the *results* from each run, not the *averages*.
            place_data[place]['mean_bad_scores'].extend(result['mean_bad_scores'])
            place_data[place]['std_dev_bad_scores'].extend(result['std_dev_bad_scores'])
            place_data[place]['filter_n'].extend(result['filter_n'])
        
        place_averages = {}
        for place, data in place_data.items():
            mean_bads = np.array(data['mean_bad_scores'])
            std_devs = np.array(data['std_dev_bad_scores'])
            filter_ns = np.array(data['filter_n'])

            # Legacy mode: thresholds are mean of bad scores only (no margin)
            if getattr(self.experiment_config, 'threshold_method', 'original') == 'legacy_mean_bad':
                thresholds = mean_bads
            else:
                # Original (current) mode: per-sample thresholds t_j = mean_bad_j + filter_n_j * std_dev_bad_j
                thresholds = mean_bads + filter_ns * std_devs

            # Method 1: Simple average over all t_j
            simple_avg_threshold = float(np.mean(thresholds))

            # Method 2: Inverse-variance weighted average over all t_j
            weight_power = 2  # change to 1 to use 1/sd weighting
            weights = 1.0 / (np.power(std_devs, weight_power) + 1e-9)
            weighted_avg_threshold = float(np.sum(weights * thresholds) / np.sum(weights))

            place_averages[place] = {
                'simple_avg_threshold': simple_avg_threshold,
                'weighted_avg_threshold': weighted_avg_threshold,
                'std_dev_of_thresholds': float(np.std(thresholds)),
                'avg_filter_n': float(np.mean(filter_ns))
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
        # --- Force output to the correct directory name for demo.py ---
        output_dir = self.results_manager.dataset_dir
        if os.path.basename(output_dir) == "gardens_point_mini":
            output_dir = os.path.join(os.path.dirname(output_dir), "GardensPoint_Mini")
            os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.join(output_dir, "place_averages.csv")
        # --- End of fix ---

        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['place', 'simple_avg_threshold', 'weighted_avg_threshold', 'std_dev_of_thresholds', 'avg_filter_n'])
            
            for place in sorted(averages.keys(), key=lambda x: int(x[1:])):
                result = averages[place]
                writer.writerow([
                    place,
                    result['simple_avg_threshold'],
                    result['weighted_avg_threshold'],
                    result['std_dev_of_thresholds'],
                    result['avg_filter_n']
                ])

def run_experiment_on_dataset(dataset_name: str, experiment_config: ExperimentConfig = None, use_cache: bool = True):
    """Run complete experiment on a specific dataset"""
    if experiment_config is None:
        from config import DEFAULT_EXPERIMENT
        experiment_config = DEFAULT_EXPERIMENT
    
    # Get and validate dataset configuration
    dataset_config = get_dataset_config(dataset_name)
    if dataset_config.format == 'landmark':
        dataset_config = auto_detect_dataset_structure(dataset_config)
    
    # The validation needs to be format-aware as well.
    # For now, let's assume sequential datasets are valid if path exists.
    if dataset_config.format == 'landmark' and not validate_dataset_structure(dataset_config):
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