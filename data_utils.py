import numpy as np
import os
import pickle
import hashlib
from PIL import Image
from glob import glob
from typing import List, Tuple, Dict
from config import DatasetConfig
from feature_extraction.feature_extractor_eigenplaces import EigenPlacesFeatureExtractor
from datasets.load_dataset import StLuciaDataset, SFUDataset # Import dataset loaders
from scipy.sparse import csgraph

class DatasetLoader:
    """Utility class for loading and processing datasets"""
    
    def __init__(self, dataset_config: DatasetConfig, use_cache: bool = True):
        self.config = dataset_config
        self.feature_extractor = EigenPlacesFeatureExtractor()
        self.use_cache = use_cache
        self.cache_dir = os.path.join("cache", dataset_config.name)
        self.place_map = [] # Will hold the structure for sequential datasets
        
        # Create cache directory if it doesn't exist
        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        if self.config.format == 'sequential':
            self._load_or_generate_place_map()
        elif self.config.format == 'landmark_grouped':
            self._generate_place_map_from_grouped_landmark()
    
    def _generate_place_map_from_grouped_landmark(self, step_size=10):
        """
        Builds a place_map in memory for landmark datasets with a flat structure,
        based on filename conventions (e.g., first 3 of every 10 images).
        """
        print(f"Generating in-memory place map for {self.config.name}...")
        
        conditions = ["day_left", "day_right", "night_right"]
        
        # We need a master list of all unique image paths to create consistent indices
        all_image_paths = []
        for cond in conditions:
            paths = sorted(glob(os.path.join(self.config.path, cond, "*.jpg")))
            all_image_paths.extend(paths)
        all_image_paths = sorted(list(set(all_image_paths)))
        
        path_to_idx = {path: i for i, path in enumerate(all_image_paths)}
        
        # Use a dictionary to build places to handle images from different conditions
        places_dict = {}

        for path in all_image_paths:
            try:
                # Extract the number from filename like '.../Image012.jpg'
                img_num = int(os.path.basename(path).replace('Image', '').replace('.jpg', ''))
                
                # Integer division by step_size gives the place ID
                place_id = img_num // step_size
                
                if place_id not in places_dict:
                    places_dict[place_id] = []
                
                places_dict[place_id].append(path_to_idx[path])

            except (ValueError, IndexError):
                print(f"Warning: Could not parse image number from filename: {path}")

        # Convert the dictionary of places to a list of lists (the place_map)
        # We sort by the place_id to ensure the order is consistent
        self.place_map = [sorted(list(set(places_dict[pid]))) for pid in sorted(places_dict.keys())]
        
        self.config.num_places = len(self.place_map)
        self.config.images_per_place = max(len(p) for p in self.place_map) if self.place_map else 0
        print(f"-> Generated place map with {self.config.num_places} places.")

    def _get_place_map_cache_path(self) -> str:
        """Get cache file path for the place map"""
        return os.path.join(self.cache_dir, "place_map.pkl")

    def _get_gt_file_hash(self) -> str:
        """Get hash of the ground truth file to detect changes"""
        gt_path = os.path.join(self.config.path, 'GT.npz')
        if not os.path.exists(gt_path):
            return None
            
        hash_md5 = hashlib.md5()
        with open(gt_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _load_or_generate_place_map(self):
        """Load place map from cache or generate it from ground truth"""
        cache_path = self._get_place_map_cache_path()
        gt_hash = self._get_gt_file_hash()

        # Check if cache file exists and is valid
        if self.use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                if cache_data.get('hash') == gt_hash:
                    print(f'Loaded cached place map for {self.config.name}')
                    self.place_map = cache_data['place_map']
                    # Update config with cached info
                    self.config.num_places = len(self.place_map)
                    self.config.images_per_place = max(len(p) for p in self.place_map) if self.place_map else 0
                    return
            except Exception as e:
                print(f'Cache read error for place map: {e}')
        
        # Generate place map if not cached or cache invalid
        print(f'Generating place map for {self.config.name}')
        
        # Use the appropriate loader from VPR_Tutorial
        if self.config.name == "StLuciaSmall":
            loader = StLuciaDataset(self.config.path)
        elif self.config.name == "SFU":
            loader = SFUDataset(self.config.path)
        else:
            raise NotImplementedError(f"No sequential loader implemented for {self.config.name}")
            
        imgs_db, imgs_q, GThard, _ = loader.load()
        # For simplicity, we'll just use the db images for now to form places
        all_images = imgs_db

        # Find connected components in the ground truth matrix
        n_components, labels = csgraph.connected_components(csgraph=GThard, directed=False, return_labels=True)
        
        place_map = [[] for _ in range(n_components)]
        for img_idx, place_label in enumerate(labels):
            # Assuming image filenames are indexed from 1, e.g., image001.jpg
            # This logic will need to be robust to the actual filename format
            # For now, we store the index.
            place_map[place_label].append(img_idx)
            
        self.place_map = place_map
        # Update config with generated info
        self.config.num_places = len(self.place_map)
        self.config.images_per_place = max(len(p) for p in self.place_map) if self.place_map else 0

        # Save to cache
        if self.use_cache:
            cache_data = {
                'place_map': self.place_map,
                'hash': gt_hash
            }
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f'Cached place map for {self.config.name}')
            except Exception as e:
                print(f'Cache write error for place map: {e}')

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
        if self.config.format == 'landmark':
            return self._load_descriptors_landmark()
        elif self.config.format in ['sequential', 'landmark_grouped']:
            return self._load_descriptors_from_place_map()

    def _load_descriptors_landmark(self) -> np.ndarray:
        """Load descriptors for landmark-based datasets"""
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

    def _load_descriptors_from_place_map(self) -> np.ndarray:
        """Load descriptors for datasets that use a place_map (sequential, landmark_grouped)"""
        
        # This function needs a comprehensive list of all image paths
        all_image_paths = []
        if self.config.format == 'sequential':
            if self.config.name == "StLuciaSmall":
                loader = StLuciaDataset(self.config.path)
            elif self.config.name == "SFU":
                loader = SFUDataset(self.config.path)
            else:
                raise NotImplementedError(f"No sequential loader implemented for {self.config.name}")
            all_image_paths = sorted(glob(os.path.join(self.config.path, loader.fns_db_path, '*.jpg')))
        
        elif self.config.format == 'landmark_grouped':
            conditions = ["day_left", "day_right", "night_right"]
            for cond in conditions:
                paths = sorted(glob(os.path.join(self.config.path, cond, "*.jpg")))
                all_image_paths.extend(paths)
            all_image_paths = sorted(list(set(all_image_paths)))


        # Collect all unique image indices that need processing
        all_required_indices = sorted(list(set(idx for place in self.place_map for idx in place)))
        
        # Check cache for all required images first
        cached_descriptors = {}
        images_to_compute = []
        paths_to_compute = []

        for idx in all_required_indices:
            cache_path = os.path.join(self.cache_dir, f"img_{idx}_descriptor.pkl")
            if self.use_cache and os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cached_descriptors[idx] = pickle.load(f)['descriptor']
            else:
                images_to_compute.append(idx)
                # Use the master path list to find the correct path for the index
                paths_to_compute.append(all_image_paths[idx])

        # Compute features in a single batch for all images that are not cached
        if paths_to_compute:
            print(f"Computing features for {len(paths_to_compute)} images...")
            images = [np.array(Image.open(p)) for p in paths_to_compute]
            computed_features = self.feature_extractor.compute_features(images)
            
            # Add computed features to cache and to our lookup dictionary
            for i, idx in enumerate(images_to_compute):
                descriptor = computed_features[i:i+1, :] # Keep dimensions
                cached_descriptors[idx] = descriptor
                if self.use_cache:
                    cache_path = os.path.join(self.cache_dir, f"img_{idx}_descriptor.pkl")
                    with open(cache_path, 'wb') as f:
                        pickle.dump({'descriptor': descriptor}, f)

        # Now, populate the descriptors_matrix from our cached/computed descriptors
        max_images_per_place = self.config.images_per_place
        descriptors_matrix = np.ndarray(
            shape=(self.config.num_places, max_images_per_place),
            dtype=object
        )
        
        for place_idx, image_indices in enumerate(self.place_map):
            for img_map_idx, original_img_idx in enumerate(image_indices):
                descriptors_matrix[place_idx, img_map_idx] = cached_descriptors[original_img_idx]
        
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

        if self.config.format == 'landmark':
            images_per_place = self.config.images_per_place
            for i in range(self.config.num_places):
                # All possible image indices for this place
                all_indices = list(range(images_per_place))
                
                # Randomly select images for training (default: 2 out of 3)
                train_size = min(2, images_per_place - 1)  # Leave at least 1 for testing
                picked_indices = list(np.random.choice(all_indices, size=train_size, replace=False))
                picked_set.append(picked_indices)
                
                # Remaining images for testing
                remaining_indices = list(set(all_indices) - set(picked_indices))
                test_index = remaining_indices[0] if remaining_indices else all_indices[-1]
                test_set.append(test_index)
                
                print(f'Place {i}: Train images: {picked_indices}, Test image: {test_index}')
        
        elif self.config.format in ['sequential', 'landmark_grouped']:
            for i, place_images in enumerate(self.place_map):
                num_images_in_place = len(place_images)
                all_indices = list(range(num_images_in_place))
                
                if num_images_in_place < 2:
                    # Cannot create a train/test split, use all for training and one for test
                    picked_indices = all_indices
                    test_index = all_indices[0] if all_indices else -1 # handle empty case
                else:
                    train_size = min(2, num_images_in_place - 1)
                    picked_indices = list(np.random.choice(all_indices, size=train_size, replace=False))
                    remaining_indices = list(set(all_indices) - set(picked_indices))
                    test_index = remaining_indices[0]
                
                picked_set.append(picked_indices)
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