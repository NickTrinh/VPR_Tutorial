import os
import shutil
from glob import glob

def prepare_gardens_point_landmark(
    source_base_path="images/GardensPoint/",
    output_base_path="images/GardensPoint_Landmark/",
    group_size=3,
    skip_size=10
):
    """
    Reorganizes the sequential Gardens Point dataset into a landmark-based format.
    
    Creates groups of 'group_size' consecutive images, then skips 'skip_size' images
    to ensure clear separation between places. Pulls from all conditions for each place.
    """
    
    # Define source folders
    conditions = ["day_left", "day_right", "night_right"]
    source_dirs = [os.path.join(source_base_path, c) for c in conditions]

    # Clean up previous runs
    if os.path.exists(output_base_path):
        print(f"Removing existing directory: {output_base_path}")
        shutil.rmtree(output_base_path)
    
    os.makedirs(output_base_path)
    print(f"Created output directory: {output_base_path}")

    # Get the total number of images in one of the source directories
    try:
        num_images_total = len(glob(os.path.join(source_dirs[0], "*.jpg")))
        if num_images_total == 0:
            print(f"Error: No images found in {source_dirs[0]}")
            return
    except IndexError:
        print(f"Error: Source directory not found or empty: {source_dirs[0]}")
        return

    place_counter = 0
    start_idx = 1  # Image names are 1-based (e.g., Image001.jpg)

    while start_idx + group_size <= num_images_total:
        place_path = os.path.join(output_base_path, f"p{place_counter}")
        os.makedirs(place_path)
        
        image_in_place_counter = 0
        
        # Collect images for the current place from all conditions
        for i in range(group_size):
            current_image_num = start_idx + i
            
            for source_dir in source_dirs:
                # Assuming the file naming convention is ImageXXX.jpg
                src_img_name = f"Image{current_image_num:03d}.jpg"
                src_path = os.path.join(source_dir, src_img_name)
                
                if os.path.exists(src_path):
                    dst_img_name = f"i{image_in_place_counter}.jpg"
                    dst_path = os.path.join(place_path, dst_img_name)
                    shutil.copy(src_path, dst_path)
                    image_in_place_counter += 1
                else:
                    print(f"Warning: Source image not found, skipping: {src_path}")

        print(f"Created Place {place_counter} with {image_in_place_counter} images.")
        
        # Move to the next place
        place_counter += 1
        start_idx += group_size + skip_size

    print("\\n--- Preparation Summary ---")
    print(f"Successfully created {place_counter} places.")
    print(f"Each place contains images from {len(conditions)} conditions.")
    print("---------------------------\\n")


if __name__ == "__main__":
    prepare_gardens_point_landmark()
