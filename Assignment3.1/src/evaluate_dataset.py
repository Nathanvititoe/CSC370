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

        label = class_map.get(folder)
        if label is None:
            print(f"skipping unknown folder: {folder}")
            continue

        # Count image files
        num_images = len(os.listdir(full_path))

        # apply count for each folder to its folder name
        per_folder_counts[folder] = num_images
        class_totals[label] += num_images

    # Images per aircraft type
    print("Image count per folder:")
    for folder, count in per_folder_counts.items():
        print(f"  {folder:6}: {count} images")

    # images per class type [fighter, bomber, helicopter]
    print("\nTotal images per class:")
    for class_id, total in class_totals.items():
        print(f"  {class_names[class_id]:<10}: {total} images")
    
    # total images overall
    print(f"\nTotal images in dataset: {sum(class_totals.values())}")
