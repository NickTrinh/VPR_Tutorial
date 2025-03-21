import csv
import os
import numpy as np
from collections import defaultdict

def average_test_results():
    """
    Read all test_results_run_X.csv files, calculate averages for each image,
    and save the results to a single averaged CSV file.
    """
    # Define the directory to search for CSV files
    directory = "."  # Current directory
    
    # Initialize data structures to hold accumulated values
    all_images = set()
    image_data = defaultdict(lambda: {
        'mean_bad_scores': [],
        'std_dev_bad_scores': [],
        'filter_n': []
    })
    
    # Count files found for reporting
    file_count = 0
    
    # Find and process all test result CSV files
    for file_name in os.listdir(directory):
        if file_name.startswith("test_results_run_") and file_name.endswith(".csv"):
            file_path = os.path.join(directory, file_name)
            file_count += 1
            
            # Read data from CSV file
            with open(file_path, 'r', newline='') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    image = row['Image']
                    all_images.add(image)
                    
                    # Extract and store values
                    try:
                        image_data[image]['mean_bad_scores'].append(float(row['Mean Bad Scores']))
                        image_data[image]['std_dev_bad_scores'].append(float(row['Std Deviation Bad Scores']))
                        image_data[image]['filter_n'].append(float(row['Filter N']))
                    except (ValueError, KeyError) as e:
                        print(f"Error processing {file_name}, row for image {image}: {e}")
    
    print(f"Processed {file_count} test result files")
    
    # Calculate averages
    averaged_results = {}
    for image in all_images:
        data = image_data[image]
        
        # Only calculate averages if we have data
        if data['mean_bad_scores']:
            averaged_results[image] = {
                'mean_bad_scores': np.mean(data['mean_bad_scores']),
                'std_dev_bad_scores': np.mean(data['std_dev_bad_scores']),
                'filter_n': np.mean(data['filter_n']),
                'count': len(data['mean_bad_scores'])  # Number of samples
            }
    
    # Generate a sorted list of images for consistent output
    # Sorting by place number first, then image number
    sorted_images = sorted(all_images, key=lambda x: (int(x.split('/')[0][1:]), int(x.split('/')[1][1:])))
    
    # Save the averaged results to a new CSV file
    output_file = "image_averages_summary.csv"
    with open(output_file, 'w', newline='') as csv_file:
        fieldnames = ['Image', 'Mean Bad Scores', 'Std Deviation Bad Scores', 'Filter N', 'Sample Count']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        for image in sorted_images:
            if image in averaged_results:
                result = averaged_results[image]
                writer.writerow({
                    'Image': image,
                    'Mean Bad Scores': result['mean_bad_scores'],
                    'Std Deviation Bad Scores': result['std_dev_bad_scores'],
                    'Filter N': result['filter_n'],
                    'Sample Count': result['count']
                })
    
    print(f"Averaged results saved to {output_file}")
    
    # Also output a summary for place-level averages
    place_data = defaultdict(lambda: {
        'mean_bad_scores': [],
        'std_dev_bad_scores': [],
        'filter_n': []
    })
    
    # Group data by place
    for image, result in averaged_results.items():
        place = image.split('/')[0]  # Extract place (e.g., 'p0')
        place_data[place]['mean_bad_scores'].append(result['mean_bad_scores'])
        place_data[place]['std_dev_bad_scores'].append(result['std_dev_bad_scores'])
        place_data[place]['filter_n'].append(result['filter_n'])
    
    # Calculate place-level averages
    place_averages = {}
    for place, data in place_data.items():
        place_averages[place] = {
            'mean_bad_scores': np.mean(data['mean_bad_scores']),
            'std_dev_bad_scores': np.mean(data['std_dev_bad_scores']),
            'filter_n': np.mean(data['filter_n'])
        }
    
    # Save place-level averages
    place_output_file = "place_averages_summary.csv"
    with open(place_output_file, 'w', newline='') as csv_file:
        fieldnames = ['Place', 'Mean Bad Scores', 'Std Deviation Bad Scores', 'Filter N']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        for place in sorted(place_data.keys(), key=lambda x: int(x[1:])):
            result = place_averages[place]
            writer.writerow({
                'Place': place,
                'Mean Bad Scores': result['mean_bad_scores'],
                'Std Deviation Bad Scores': result['std_dev_bad_scores'],
                'Filter N': result['filter_n']
            })
    
    print(f"Place-level averages saved to {place_output_file}")
    
    return averaged_results, place_averages

if __name__ == "__main__":
    average_test_results() 