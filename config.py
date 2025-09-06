from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os

@dataclass
class DatasetConfig:
    """Configuration for a dataset"""
    name: str
    path: str
    format: str  # 'landmark', 'sequential', or 'landmark_grouped'
    num_places: int
    images_per_place: int
    image_extension: str = "*.jpg"
    description: str = ""
    conditions: Optional[List[str]] = None  # list of subfolders representing conditions
    # Optional grouping parameters for landmark_grouped datasets without unified naming
    grouping_step_size: Optional[int] = None
    grouping_group_size: Optional[int] = None

@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    num_runs: int = 30
    random_seed: int = 42
    train_test_split: Tuple[int, int] = (2, 1)  # (train_images, test_images) per place
    output_dir: str = "results"
    threshold_multiplier: float = 1.0  # Multiplier for threshold adjustment (< 1.0 = more lenient)
    threshold_method: str = "original"  # "original" threshold method

# Dataset configurations
DATASETS = {
    "fordham_places": DatasetConfig(
        name="FordhamPlaces",
        path="images/FordhamPlaces/",
        format="landmark",
        num_places=11,
        images_per_place=3,
        description="Fordham Places dataset with 11 places, 3 images each"
    ),
    "matching_triplets": DatasetConfig(
        name="MatchingTriplets",
        path="images/MatchingTriplets/",
        format="landmark",
        num_places=10,
        images_per_place=3,
        description="Matching Triplets dataset with 10 places, 3 images each"
    ),
    "gardens_point_mini": DatasetConfig(
        name="GardensPoint_Mini",
        path="images/GardensPoint_Mini/",
        format="landmark_grouped",
        # These will be determined by the loader, placeholders are fine
        num_places=20, 
        images_per_place=6,
        description="Mini Gardens Point built via group-and-skip over conditions.",
        conditions=["day_left", "day_right", "night_right"],
        grouping_step_size=10,
        grouping_group_size=3
    ),
    "gardens_point_mini_2": DatasetConfig(
        name="GardensPoint_Mini_2",
        path="images/GardensPoint_Mini_2/",
        format="landmark_grouped",
        num_places=0,
        images_per_place=0,
        description="Mini Gardens Point (choose 2, skip 8).",
        conditions=["day_left", "day_right", "night_right"],
        grouping_step_size=10,
        grouping_group_size=2
    ),
    "gardens_point_mini_3": DatasetConfig(
        name="GardensPoint_Mini_3",
        path="images/GardensPoint_Mini_3/",
        format="landmark_grouped",
        num_places=0,
        images_per_place=0,
        description="Mini Gardens Point (choose 3, skip 7).",
        conditions=["day_left", "day_right", "night_right"],
        grouping_step_size=10,
        grouping_group_size=3
    ),
    "google_landmarks_micro": DatasetConfig(
        name="GoogleLandmarksMicro",
        path="images/GoogleLandmarksMicro/",
        format="landmark",
        num_places=20,       # Will be auto-detected, but good to have a placeholder
        images_per_place=50,  # Will be auto-detected
        description="A micro subset of the Google Landmarks v2 dataset."
    ),
    "st_lucia": DatasetConfig(
        name="StLuciaSmall",
        path="images/StLucia_small/",
        format="sequential",
        num_places=0,  # Will be detected automatically
        images_per_place=0,  # Will be detected automatically
        description="St Lucia small dataset",
        conditions=["100909_0845", "180809_1545"]
    ),
    "gardens_point": DatasetConfig(
        name="GardensPoint",
        path="images/GardensPoint/",
        format="sequential",
        num_places=0,  # Will be detected automatically
        images_per_place=0,  # Will be detected automatically
        description="Gardens Point dataset",
        conditions=["day_left", "day_right", "night_right"]
    ),
    "sfu": DatasetConfig(
        name="SFU",
        path="images/SFU/",
        format="sequential",
        num_places=0,  # Will be detected automatically
        images_per_place=0,  # Will be detected automatically
        description="SFU dataset",
        conditions=["dry", "jan"]
    ),
    "tokyo247": DatasetConfig(
        name="Tokyo247",
        path="mini_VPR_datasets/Tokyo24_7/tokyo247_vpr_format/",
        format="landmark",
        num_places=99,
        images_per_place=3,
        image_extension="*.jpg",
        description="Tokyo 24/7 dataset, formatted for VPR experiments"
    )
}

# Default experiment configuration
DEFAULT_EXPERIMENT = ExperimentConfig()

def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get dataset configuration by name"""
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {list(DATASETS.keys())}")
    return DATASETS[dataset_name]

def validate_dataset_path(dataset_config: DatasetConfig) -> bool:
    """Validate that dataset path exists"""
    return os.path.exists(dataset_config.path)

def auto_detect_dataset_structure(dataset_config: DatasetConfig) -> DatasetConfig:
    """Automatically detect number of places and images per place"""
    if not validate_dataset_path(dataset_config):
        raise FileNotFoundError(f"Dataset path not found: {dataset_config.path}")
    
    places = [d for d in os.listdir(dataset_config.path) 
              if os.path.isdir(os.path.join(dataset_config.path, d)) and d.startswith('p')]
    
    if not places:
        raise ValueError(f"No place directories found in {dataset_config.path}")
    
    # Update num_places if it's 0 (auto-detect)
    if dataset_config.num_places == 0:
        dataset_config.num_places = len(places)
    
    # Auto-detect images per place from first place
    if dataset_config.images_per_place == 0:
        first_place_path = os.path.join(dataset_config.path, places[0])
        image_dirs = [d for d in os.listdir(first_place_path) 
                      if os.path.isdir(os.path.join(first_place_path, d)) and d.startswith('i')]
        dataset_config.images_per_place = len(image_dirs)
    
    return dataset_config 