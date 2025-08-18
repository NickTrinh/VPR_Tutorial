import pandas as pd
import os
import sys
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def prepare_google_landmarks_subset(num_places=20, min_images=5, max_images=50):
    """
    Analyzes the Google Landmarks v2 metadata to create a smaller, balanced subset
    and prepares it for download.
    """
    
    meta_path = "gldv2_meta/train.csv"
    if not os.path.exists(meta_path):
        print(f"Error: Metadata file not found at '{meta_path}'")
        print("Please download 'train.csv' first.")
        return

    print("Reading metadata, this may take a moment...")
    df = pd.read_csv(meta_path)

    print("Analyzing landmark distribution...")
    landmark_counts = df['landmark_id'].value_counts()
    
    # Filter for landmarks that have a reasonable number of images
    eligible_landmarks = landmark_counts[(landmark_counts >= min_images) & (landmark_counts <= max_images)]
    
    if len(eligible_landmarks) < num_places:
        print(f"Warning: Found only {len(eligible_landmarks)} landmarks with {min_images}-{max_images} images.")
        print("Using all of them. Consider adjusting min/max image counts.")
        num_places = len(eligible_landmarks)

    # Select the top 'num_places' landmarks from the eligible list
    selected_landmark_ids = eligible_landmarks.head(num_places).index.tolist()
    
    print(f"\nSelected {len(selected_landmark_ids)} landmarks for the micro-dataset.")
    
    # Filter the main dataframe to get only the images for our selected landmarks
    subset_df = df[df['landmark_id'].isin(selected_landmark_ids)]

    # --- Create Directory Structure ---
    output_dir = "images/GoogleLandmarksMicro"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a mapping from landmark_id to a place index (p0, p1, ...)
    landmark_to_place_map = {landmark_id: f"p{i}" for i, landmark_id in enumerate(selected_landmark_ids)}

    download_plan = []
    print("Creating directory structure and download plan...")
    for landmark_id, place_folder in landmark_to_place_map.items():
        place_path = os.path.join(output_dir, place_folder)
        os.makedirs(place_path, exist_ok=True)
        
        # Get all images for this landmark
        images_for_landmark = subset_df[subset_df['landmark_id'] == landmark_id]
        
        for i, row in enumerate(images_for_landmark.itertuples()):
            # Define image path p{place}/i{img_num}
            img_folder_path = os.path.join(place_path, f"i{i}")
            os.makedirs(img_folder_path, exist_ok=True)
            
            # Get file extension from URL
            file_ext = os.path.splitext(row.url)[1]
            if not file_ext: file_ext = '.jpg' # Default extension
            
            final_img_path = os.path.join(img_folder_path, f"image{file_ext}")
            
            download_plan.append({
                "url": row.url,
                "path": final_img_path
            })

    # Save the download plan
    plan_df = pd.DataFrame(download_plan)
    plan_path = "gldv2_meta/download_plan.csv"
    plan_df.to_csv(plan_path, index=False)
    
    print(f"\nSuccessfully created directory structure in '{output_dir}'.")
    print(f"Download plan saved to '{plan_path}'.")
    print(f"\nNext step: Run this script with the --download flag to get the images.")


def download_image(url, path):
    """Downloads a single image and saves it."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=10, headers=headers)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            # Print the non-200 status code for debugging
            # To avoid clutter, we can comment this out later.
            print(f"Failed URL: {url} | Status Code: {response.status_code}", file=sys.stderr)
            return False
    except requests.exceptions.RequestException as e:
        # Print specific request-related errors
        # To avoid clutter, we can comment this out later.
        print(f"Failed URL: {url} | Error: {e}", file=sys.stderr)
        return False
    return False

def execute_download_plan(max_workers=10):
    """Downloads all images from the download_plan.csv."""
    plan_path = "gldv2_meta/download_plan.csv"
    if not os.path.exists(plan_path):
        print("Error: Download plan not found. Please run the script without flags first.")
        return

    plan_df = pd.read_csv(plan_path)
    
    urls_paths = list(zip(plan_df['url'], plan_df['path']))

    print(f"Starting download of {len(urls_paths)} images with {max_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(lambda p: download_image(*p), urls_paths), total=len(urls_paths)))

    success_count = sum(1 for r in results if r)
    print(f"\nDownload complete. Successfully downloaded {success_count} / {len(urls_paths)} images.")


if __name__ == "__main__":
    if "--download" in sys.argv:
        execute_download_plan()
    else:
        prepare_google_landmarks_subset()
