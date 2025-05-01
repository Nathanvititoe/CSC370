from src.dataset_prep.dataset_loader import build_dataset_list, get_datasets
from src.dataset_prep.evaluate_dataset import visualize_audio_stats
from src.ui.visualization import show_spectrogram
import pandas as pd

# logic control for dataset preparation
def setup_dataset(DATASET_DIR, CSV_PATH, BATCH_SIZE, SAMPLE_RATE, duration_length, val_split=0.2):
    metadata = pd.read_csv(CSV_PATH) # get metadata from csv

    # get human readable labels for visualizations
    label_names = list(metadata.drop_duplicates("classID").sort_values("classID")["class"])
    all_folds = list(range(1,11)) # get all folds 

    # build dataset list and collect stats
    dataset_list, durations, sizes, files_per_fold, labels = build_dataset_list(DATASET_DIR, SAMPLE_RATE, metadata, folds=all_folds)

    
    # visualize audio stats to identify outliers, clean data
    # visualize_audio_stats(durations, sizes, labels, label_names, files_per_fold)

    # split folds into train/val based on val_split
    val_count = int(len(all_folds) * val_split)
    val_folds = all_folds[:val_count]
    train_folds = all_folds[val_count:]

    # create train/val list's using fold counts
    train_list = [item for item in dataset_list if item[2] in train_folds]
    val_list = [item for item in dataset_list if item[2] in val_folds]

    # create datasets
    train_ds, val_ds = get_datasets(train_list, val_list, BATCH_SIZE, SAMPLE_RATE, duration_length)

    # display example spectrogram to show what the model will use for classification
    # show_spectrogram(train_ds, label_names)
    
    return train_ds, val_ds, label_names
