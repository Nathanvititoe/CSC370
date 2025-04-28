from src.dataset_setup.evaluate_dataset import dataset_evaluation
from src.dataset_setup.dataset_loader import build_dataset_list, create_dataset
from src.dataset_setup.dataset_splitter import split_dataset
from collections import Counter

# logic center for dataset preparation
def setup_dataset(DATASET_DIR, CLASS_MAP, CLASS_NAMES, IMG_SIZE, BATCH_SIZE, NUM_CLASSES, NUM_WORKERS): 
    # get image counts, to verify class distribution and ds size
    dataset_evaluation(DATASET_DIR, CLASS_MAP, CLASS_NAMES)

    # get list of image paths and labels
    image_labels_list = build_dataset_list(DATASET_DIR, CLASS_MAP)

    # # split the model 
    train_list, val_list = split_dataset(image_labels_list)

    # get datasets 
    # train_loader, val_loader = get_dataloaders(train_ds, val_ds, NUM_CLASSES, IMG_SIZE, BATCH_SIZE, NUM_WORKERS)
    train_ds = create_dataset(train_list, IMG_SIZE)
    val_ds = create_dataset(val_list, IMG_SIZE)

    # verify there are no duplicates in train/val
    train_paths = set(img_path for img_path, _ in train_ds)
    val_paths = set(img_path for img_path, _ in val_ds)
    overlap = train_paths.intersection(val_paths)
    print(f"\n\nNumber of duplicated images between train and val: {len(overlap)}\n\n")

    # output class distribution for verification
    print("----- Class distribution counts -----\n")
    print(f"Train: {Counter(label for _, label in train_ds)}")
    print(f"Val  : {Counter(label for _, label in val_ds)}")
    print("--------------------------------------")

    return train_ds, val_ds