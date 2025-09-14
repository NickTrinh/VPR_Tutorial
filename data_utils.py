import numpy as np
import os
import pickle
import hashlib
import re
from PIL import Image
from glob import glob
from typing import List, Tuple, Dict
from config import DatasetConfig
from feature_extraction.feature_extractor_eigenplaces import EigenPlacesFeatureExtractor
from feature_extraction.feature_extractor_cosplace import CosPlaceFeatureExtractor
from feature_extraction.feature_extractor_holistic import AlexNetConv3Extractor, HDCDELF, SAD
from datasets.load_dataset import StLuciaDataset, SFUDataset # Import dataset loaders
from scipy.sparse import csgraph

class DatasetLoader:
    """Utility class for loading and processing datasets"""
    
    def __init__(self, dataset_config: DatasetConfig, use_cache: bool = True, descriptor_name: str = "eigenplaces"):
        self.config = dataset_config
        self.feature_extractor = self._init_feature_extractor(descriptor_name)
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

    def _init_feature_extractor(self, name: str):
        n = (name or "").lower()
        if n == "cosplace":
            return CosPlaceFeatureExtractor()
        if n in ("eigenplaces", "eigenplace"):
            return EigenPlacesFeatureExtractor()
        if n == "alexnet":
            return AlexNetConv3Extractor()
        if n in ("hdc-delf", "hdcdelf", "hdc_delf"):
            return HDCDELF()
        if n == "sad":
            return SAD()
        if n in ("netvlad", "patchnetvlad"):
            try:
                import os
                import configparser
                from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR
                from feature_extraction.feature_extractor_patchnetvlad import PatchNetVLADFeatureExtractor
                if n == "netvlad":
                    configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/netvlad_extract.ini')
                else:
                    configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/speed.ini')
                cfg = configparser.ConfigParser()
                cfg.read(configfile)
                return PatchNetVLADFeatureExtractor(cfg)
            except Exception as e:
                raise ImportError("PatchNetVLAD is not available. Install 'patchnetvlad' and its models, or choose a different --descriptor.") from e
        return EigenPlacesFeatureExtractor()
    
    def _generate_place_map_from_grouped_landmark(self):
        """
        Builds a place_map for grouped datasets.
        Preferred: parse unified filenames Place####_CondCC_GG.jpg.
        Fallback: if no unified naming is found, generate by index using
        grouping_step_size and grouping_group_size from config (e.g., take 3 skip 7).
        """
        print(f"Generating in-memory place map for {self.config.name}...")

        # Resolve conditions from config if available; otherwise infer subfolders
        if self.config.conditions:
            conditions = self.config.conditions
        else:
            conditions = [d for d in sorted(os.listdir(self.config.path))
                          if os.path.isdir(os.path.join(self.config.path, d))]

        # Build a stable list of all image paths per condition and combined
        per_condition_paths: List[List[str]] = []
        for cond in conditions:
            pat = self.config.image_extension or '*.jpg'
            per_condition_paths.append(sorted(glob(os.path.join(self.config.path, cond, pat))))
        all_image_paths: List[str] = sorted(list(set(p for plist in per_condition_paths for p in plist)))

        path_to_idx = {path: i for i, path in enumerate(all_image_paths)}

        # Try unified naming first
        place_pattern = re.compile(r"Place(\d+)_Cond\d{2}_G\d{2}\.(?:jpg|jpeg|png)$", re.IGNORECASE)
        places_dict: Dict[int, List[int]] = {}
        for path in all_image_paths:
            filename = os.path.basename(path)
            m = place_pattern.search(filename)
            if m:
                place_id = int(m.group(1))
                places_dict.setdefault(place_id, []).append(path_to_idx[path])

        if places_dict:
            # Unified naming path
            self.place_map = [sorted(list(set(places_dict[pid]))) for pid in sorted(places_dict.keys())]
            self.config.num_places = len(self.place_map)
            self.config.images_per_place = max(len(p) for p in self.place_map) if self.place_map else 0
            print(f"-> Generated place map with {self.config.num_places} places (unified filenames).")
            return

        # Fallback: index-based grouping using config parameters
        step_size = self.config.grouping_step_size or 10
        group_size = self.config.grouping_group_size or 3
        # Determine the minimum number of images across conditions
        min_len = min(len(plist) for plist in per_condition_paths) if per_condition_paths else 0
        place_map: List[List[int]] = []
        place_idx = 0
        idx = 0
        while idx + group_size - 1 < min_len:
            group_indices: List[int] = []
            for g in range(group_size):
                index = idx + g
                # Add images across all conditions for this index
                for cond_list in per_condition_paths:
                    if index < len(cond_list):
                        group_indices.append(path_to_idx[cond_list[index]])
            place_map.append(sorted(list(set(group_indices))))
            place_idx += 1
            idx += step_size

        self.place_map = place_map
        self.config.num_places = len(self.place_map)
        self.config.images_per_place = max((len(p) for p in self.place_map), default=0)
        print(f"-> Generated place map with {self.config.num_places} places (index-based grouping).")

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
            # Resolve conditions from config or infer
            if self.config.conditions:
                conditions = self.config.conditions
            else:
                conditions = [d for d in sorted(os.listdir(self.config.path))
                              if os.path.isdir(os.path.join(self.config.path, d))]
            img_pat = self.config.image_extension or '*.jpg'
            for cond in conditions:
                paths = sorted(glob(os.path.join(self.config.path, cond, img_pat)))
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
                all_indices = list(range(images_per_place))
                if images_per_place < 2:
                    picked_indices = all_indices
                    test_index = all_indices[0] if all_indices else -1
                else:
                    # Select exactly 1 test index; use all remaining for training
                    test_index = int(np.random.choice(all_indices, size=1, replace=False))
                    picked_indices = [idx for idx in all_indices if idx != test_index]
                picked_set.append(picked_indices)
                test_set.append(test_index)
                print(f'Place {i}: Train images: {picked_indices}, Test image: {test_index}')
        
        elif self.config.format in ['sequential', 'landmark_grouped']:
            for i, place_images in enumerate(self.place_map):
                num_images_in_place = len(place_images)
                all_indices = list(range(num_images_in_place))
                if num_images_in_place < 2:
                    picked_indices = all_indices
                    test_index = all_indices[0] if all_indices else -1
                else:
                    test_index = int(np.random.choice(all_indices, size=1, replace=False))
                    picked_indices = [idx for idx in all_indices if idx != test_index]
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