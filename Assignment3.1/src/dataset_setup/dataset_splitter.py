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