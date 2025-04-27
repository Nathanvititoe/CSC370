from sklearn.model_selection import train_test_split # type: ignore
from collections import defaultdict
import random

# function to split the ds into train/validation sets
def dataset_split(label_list):
   # use sklearn train_test_split to split ds 50/50
   return train_test_split(
        label_list, # list of img path and label pairs
        test_size=0.5, # what % of ds in val set
        stratify=[label for _, label in label_list], # maintain class distribution
        random_state=42 # random seeding
    )

# def balanced_split(label_list, test_size=0.5):
#     # Group images by class
#     class_to_images = defaultdict(list)
#     for path, label in label_list:
#         class_to_images[label].append((path, label))

#     # Find smallest class size
#     min_class_size = min(len(lst) for lst in class_to_images.values())

#     # Sample min_class_size images from each class
#     balanced_list = []
#     for label, imgs in class_to_images.items():
#         balanced_list.extend(random.sample(imgs, min_class_size))

#     # Shuffle balanced list
#     random.shuffle(balanced_list)

#     # Then split into train/val
#     train_list, val_list = train_test_split(
#         balanced_list,
#         test_size=test_size,
#         stratify=[label for _, label in balanced_list],
#         random_state=42
#     )

#     return train_list, val_list