import numpy as np
import os
import pickle
import hashlib
from PIL import Image
from glob import glob
from typing import List, Tuple, Dict
from config import DatasetConfig
from feature_extraction.feature_extractor_holistic import HDCDELF

class DatasetLoader:
    """Utility class for loading and processing datasets"""
    
    def __init__(self, dataset_config: DatasetConfig, use_cache: bool = True):
        self.config = dataset_config
        self.feature_extractor = HDCDELF()
        self.use_cache = use_cache
        self.cache_dir = os.path.join("cache", dataset_config.name)
        
        # Create cache directory if it doesn't exist
        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_image_cache_path(self, place_idx: int, img_idx: int) -> str:
        """Get cache file path for a specific image"""
        return os.path.join(self.cache_dir, f"p{place_idx}_i{img_idx}_descriptor.pkl")
    
    def _get_image_hash(self, img_dir: str) -> str:
        """Get hash of all images in directory to detect changes"""
        img_files = sorted(glob(os.path.join(img_dir, self.config.image_extension)))
        hash_md5 = hashlib.md5()
        
        for img_file in img_files:
            with open(img_file, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _load_cached_descriptor(self, place_idx: int, img_idx: int, img_dir: str) -> np.ndarray:
        """Load descriptor from cache if valid, otherwise compute and cache it"""
        cache_path = self._get_image_cache_path(place_idx, img_idx)
        
        # Check if cache file exists
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Verify the hash matches (images haven't changed)
                current_hash = self._get_image_hash(img_dir)
                if cache_data['hash'] == current_hash:
                    print(f'Loaded cached descriptor for p{place_idx}/i{img_idx}')
                    return cache_data['descriptor']
                else:
                    print(f'Cache invalid for p{place_idx}/i{img_idx} (images changed)')
            except Exception as e:
                print(f'Cache read error for p{place_idx}/i{img_idx}: {e}')
        
        # Compute descriptor if not cached or cache invalid
        print(f'Computing descriptor for p{place_idx}/i{img_idx}')
        descriptor = self.get_descriptor(img_dir)
        
        # Save to cache
        if self.use_cache:
            cache_data = {
                'descriptor': descriptor,
                'hash': self._get_image_hash(img_dir)
            }
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f'Cached descriptor for p{place_idx}/i{img_idx}')
            except Exception as e:
                print(f'Cache write error for p{place_idx}/i{img_idx}: {e}')
        
        return descriptor
    
    def get_descriptor(self, img_dir: str) -> np.ndarray:
        """Extract features from images in a directory"""
        img_files = sorted(glob(os.path.join(img_dir, self.config.image_extension)))
        if not img_files:
            raise ValueError(f"No images found in {img_dir}")
        
        images = [np.array(Image.open(img)) for img in img_files]
        return self.feature_extractor.compute_features(images)
    
    def load_all_descriptors(self) -> np.ndarray:
        """Load descriptors for all images in the dataset with caching"""
        descriptors_matrix = np.ndarray(
            shape=(self.config.num_places, self.config.images_per_place), 
            dtype=object
        )
        
        print(f'Loading descriptors for {self.config.name} dataset (caching enabled: {self.use_cache})')
        for i in range(self.config.num_places):
            for j in range(self.config.images_per_place):
                img_path = os.path.join(self.config.path, f'p{i}', f'i{j}')
                
                if self.use_cache:
                    descriptors_matrix[i, j] = self._load_cached_descriptor(i, j, img_path)
                else:
                    print(f'Computing descriptors for p{i}/i{j}')
                    descriptors_matrix[i, j] = self.get_descriptor(img_path)
        
        return descriptors_matrix
    
    def clear_cache(self):
        """Clear all cached descriptors for this dataset"""
        if os.path.exists(self.cache_dir):
            import shutil
            shutil.rmtree(self.cache_dir)
            print(f"Cleared cache for {self.config.name}")
    
    def create_train_test_split(self, random_state: int = None) -> Tuple[List[List[int]], List[int]]:
        """Create train/test split for each place"""
        if random_state is not None:
            np.random.seed(random_state)
        # If random_state is None, use current system state (truly random)
        
        picked_set = []
        test_set = []
        
        print(f'Creating train/test split for {self.config.name} (random_state: {random_state})')
        for i in range(self.config.num_places):
            # All possible image indices for this place
            all_indices = list(range(self.config.images_per_place))
            
            # Randomly select images for training (default: 2 out of 3)
            train_size = min(2, self.config.images_per_place - 1)  # Leave at least 1 for testing
            picked_indices = list(np.random.choice(all_indices, size=train_size, replace=False))
            picked_set.append(picked_indices)
            
            # Remaining images for testing
            remaining_indices = list(set(all_indices) - set(picked_indices))
            test_index = remaining_indices[0] if remaining_indices else all_indices[-1]
            test_set.append(test_index)
            
            print(f'Place {i}: Train images: {picked_indices}, Test image: {test_index}')
        
        return picked_set, test_set

class ResultsManager:
    """Utility class for managing experiment results"""
    
    def __init__(self, dataset_name: str, output_dir: str = "results"):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.dataset_dir = os.path.join(output_dir, dataset_name)
        
        # Create output directories
        os.makedirs(self.dataset_dir, exist_ok=True)
    
    def get_run_filename(self, run_number: int) -> str:
        """Get filename for a specific run"""
        return os.path.join(self.dataset_dir, f'test_results_run_{run_number}.csv')
    
    def get_averaged_filename(self) -> str:
        """Get filename for averaged results"""
        return os.path.join(self.dataset_dir, 'averaged_results.csv')
    
    def get_place_averages_filename(self) -> str:
        """Get filename for place-level averages"""
        return os.path.join(self.dataset_dir, 'place_averages.csv')
    
    def get_image_averages_filename(self) -> str:
        """Get filename for image-level averages"""
        return os.path.join(self.dataset_dir, 'image_averages.csv')
    
    def get_final_results_filename(self) -> str:
        """Get filename for final test results"""
        return os.path.join(self.dataset_dir, 'final_test_results.csv')

def validate_dataset_structure(dataset_config: DatasetConfig) -> bool:
    """Validate that the dataset has the expected structure"""
    if not os.path.exists(dataset_config.path):
        print(f"Dataset path does not exist: {dataset_config.path}")
        return False
    
    # Check if we have the expected number of places
    place_dirs = [d for d in os.listdir(dataset_config.path) 
                  if os.path.isdir(os.path.join(dataset_config.path, d)) and d.startswith('p')]
    
    if len(place_dirs) != dataset_config.num_places:
        print(f"Expected {dataset_config.num_places} places, found {len(place_dirs)}")
        return False
    
    # Check if each place has the expected number of image directories
    for i in range(dataset_config.num_places):
        place_path = os.path.join(dataset_config.path, f'p{i}')
        if not os.path.exists(place_path):
            print(f"Place directory does not exist: {place_path}")
            return False
        
        image_dirs = [d for d in os.listdir(place_path) 
                      if os.path.isdir(os.path.join(place_path, d)) and d.startswith('i')]
        
        if len(image_dirs) != dataset_config.images_per_place:
            print(f"Place p{i}: Expected {dataset_config.images_per_place} image dirs, found {len(image_dirs)}")
            return False
    
    return True 