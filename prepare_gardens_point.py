import os
import shutil
from glob import glob

def prepare_gardens_point_mini(
    source_base_path="images/GardensPoint/",
    output_base_path="images/GardensPoint_Mini/",
    group_size=3,
    step_size=10
):
    """
    Creates a smaller, cleaner version of the Gardens Point dataset by selecting
    the first 'group_size' images from every 'step_size' images.
    
    This creates a new dataset with the same flat structure as the original,
    which will be handled by a custom data loader.
    """
    
    conditions = ["day_left", "day_right", "night_right"]
    
    if os.path.exists(output_base_path):
        print(f"Removing existing directory: {output_base_path}")
        shutil.rmtree(output_base_path)
    
    print(f"Creating output directory structure at: {output_base_path}")
    for condition in conditions:
        os.makedirs(os.path.join(output_base_path, condition))

    try:
        num_images_total = len(glob(os.path.join(source_base_path, conditions[0], "*.jpg")))
    except IndexError:
        print(f"Error: Source directory not found or empty.")
        return

    images_copied_count = 0
    start_idx = 0

    while start_idx < num_images_total:
        for i in range(group_size):
            current_image_num = start_idx + i
            if current_image_num >= num_images_total:
                continue

            src_img_name = f"Image{current_image_num:03d}.jpg"
            
            for condition in conditions:
                src_path = os.path.join(source_base_path, condition, src_img_name)
                dst_path = os.path.join(output_base_path, condition, src_img_name)
                
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
                    images_copied_count += 1
                else:
                    # This warning is important in case of missing files
                    print(f"Warning: Source image not found, skipping: {src_path}")

        start_idx += step_size

    print("\\n--- Preparation Summary ---")
    # Each image is copied 3 times (one for each condition)
    print(f"Successfully copied {images_copied_count // len(conditions)} unique image indices across {len(conditions)} conditions.")
    print(f"Total files created: {images_copied_count}")
    print(f"The new dataset is ready at: {output_base_path}")
    print("---------------------------\\n")

if __name__ == "__main__":
    prepare_gardens_point_mini()
