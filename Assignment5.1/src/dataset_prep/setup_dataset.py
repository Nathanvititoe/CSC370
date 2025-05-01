from src.dataset_prep.dataset_loader import build_dataset_list
from src.dataset_prep.evaluate_dataset import visualize_audio_stats
import pandas as pd

# logic control for dataset preparation
def setup_dataset(CSV_PATH, DATASET_DIR, val_split=0.2):
    metadata = pd.read_csv(CSV_PATH) # get metadata from csv

    # get human readable labels for visualizations
    label_names = list(metadata.drop_duplicates("classID").sort_values("classID")["class"])
    all_folds = list(range(1,11)) # get all folds 
    # build dataset and collect stats
    dataset, durations, sizes, files_per_fold, labels = build_dataset_list(DATASET_DIR, metadata, folds=all_folds)

    # visualize audio stats to identify outliers, clean data
    visualize_audio_stats(durations, sizes, labels, label_names, files_per_fold)

    # split folds into train/val based on val_split
    val_count = int(len(all_folds) * val_split)
    val_folds = all_folds[:val_count]
    train_folds = all_folds[val_count:]

    # create train/val ds's using fold counts
    train_ds = [item[:2] for item in dataset if item[2] in train_folds]
    val_ds = [item[:2] for item in dataset if item[2] in val_folds]

    return train_ds, val_ds, label_names
