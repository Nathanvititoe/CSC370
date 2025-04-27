import os
from collections import defaultdict, Counter

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
            print(f"skipping unknown folder: {folder}")
            continue

        # Count image files
        num_images = len(os.listdir(full_path))

        # apply count for each folder to its folder name
        per_folder_counts[folder] = num_images
        class_totals[label] += num_images

    # Images per aircraft type
    print("\n--------Image count per folder--------")
    for folder, count in per_folder_counts.items():
        print(f"  {folder:6}: {count} images")

    # images per class type [fighter, bomber, helicopter]
    print("\n-------Total images per class--------\n")
    for class_id, total in class_totals.items():
        print(f"  {class_names[class_id]:<10}: {total} images")
    print("\n-------------------------------------")

# function to determine class weights, allows us to use all data without bias
def get_class_weights(image_labels_list, num_classes):
    # count how many images/labels per class
    label_counts = Counter(label for _, label in image_labels_list)

    # toal number of images in dataset
    total_images = sum(label_counts.values())

    # get the class weights
    class_weights = {
        class_id: total_images / (num_classes * count)
        for class_id, count in label_counts.items()
    }

     # output the class weights
    print("\n---------- Class Weights ----------\n")
    for class_id, weight in class_weights.items():
        print(f"Class {class_id}: {weight:.4f}")
    print("\n------------------------------------\n")