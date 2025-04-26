import os
from collections import defaultdict, Counter
import random

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

def balance_dataset(image_labels_list, class_names):
    # Group images by CLASS (not by folder)
    class_groups = defaultdict(list)
    for img_path, label in image_labels_list:
        class_groups[label].append((img_path, label))

    # Find the smallest class size
    smallest_class_size = min(len(images) for images in class_groups.values())
    print(f"\nSmallest class size is: {smallest_class_size}\n")

    # Sample evenly across classes
    balanced_list = []
    for class_id, images in class_groups.items():
        selected_images = random.sample(images, smallest_class_size)
        balanced_list.extend(selected_images)

    random.shuffle(balanced_list)  # Shuffle to mix classes

    # Print new class distribution
    label_counter = Counter(label for _, label in balanced_list)
    print("\n--- Balanced Class Distribution ---")
    for label, count in label_counter.items():
        class_name = class_names[label]  # <-- FIX: look up true class name
        print(f"{class_name:10}: {count} images")
    print("-----------------------------------\n")

    return balanced_list
