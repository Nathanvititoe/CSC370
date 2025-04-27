from src.dataset_setup.evaluate_dataset import dataset_evaluation, get_class_weights
from src.dataset_setup.dataset_loader import build_dataset_list, get_datasets
from src.dataset_setup.dataset_splitter import dataset_split, balanced_split
from collections import Counter

# logic center for dataset preparation
def setup_dataset(DATASET_DIR, CLASS_MAP, CLASS_NAMES, img_size, batch_size, num_classes): 
    # get image counts, to verify class distribution and ds size
    dataset_evaluation(DATASET_DIR, CLASS_MAP, CLASS_NAMES)

    # get list of image paths and labels for training
    image_labels_list = build_dataset_list(DATASET_DIR, CLASS_MAP)

    # get class weights
    class_weights = get_class_weights(image_labels_list, num_classes)

    # split the model 
    train_ds, val_ds = dataset_split(image_labels_list)

    # get datasets 
    final_train_ds, final_val_ds = get_datasets(train_ds, val_ds, img_size, batch_size)

    # verify there are no duplicates in train/val
    train_paths = set(img_path for img_path, _ in train_ds)
    val_paths = set(img_path for img_path, _ in val_ds)
    overlap = train_paths.intersection(val_paths)
    print(f"Number of duplicated images between train and val: {len(overlap)}\n\n")

    # output class distribution for verification
    print("----- Class distribution counts -----\n")
    print(f"Train: {Counter(label for _, label in train_ds)}")
    print(f"Val  : {Counter(label for _, label in val_ds)}")
    print("\n--------------------------------------")

    return final_train_ds, final_val_ds, class_weights,