from src.dataset_setup.evaluate_dataset import dataset_evaluation
from src.dataset_setup.dataset_loader import build_dataset_list, create_dataset

# logic center for dataset preparation
def setup_dataset(DATASET_DIR, CLASS_MAP, CLASS_NAMES, transform): 
    # get image counts, to verify class distribution and ds size
    dataset_evaluation(DATASET_DIR, CLASS_MAP, CLASS_NAMES)

    # get list of image paths and labels
    image_labels_list = build_dataset_list(DATASET_DIR, CLASS_MAP)
   
    # get datasets 
    dataset = create_dataset(image_labels_list, transform)
    
    return dataset

