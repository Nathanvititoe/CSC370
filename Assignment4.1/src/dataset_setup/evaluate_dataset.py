import os
from collections import defaultdict

# function to count # of photos for each class
def dataset_evaluation(dataset_dir, class_map, class_names):
    # init count vars
    per_folder_counts = {}
    class_totals = defaultdict(int)

    # iterate through dataset folders
    for folder in sorted(os.listdir(dataset_dir)):
        full_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(full_path):
            continue

        # get label from the class map
        label = class_map.get(folder)

        # skip folders that arent defined in the class map
        if label is None:
            print(f"        skipping unknown folder: {folder}")
            continue

        # Count image files
        num_images = len(os.listdir(full_path))

        # apply count for each folder to its folder name
        per_folder_counts[folder] = num_images
        class_totals[label] += num_images
    
    # images per class type
    print("\n\n-------Total images per class--------\n")
    for class_id, total in class_totals.items():
        print(f"    {class_names[class_id]:<10}: {total} images")
    print("\n-------------------------------------\n")

