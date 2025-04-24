import keras_core as keras # type: ignore
keras.backend.set_backend("torch")

from src.evaluate_dataset import dataset_evaluation 
from src.dataset_loader import build_dataset_list, get_loaders
from src.dataset_splitter import dataset_split
# Root dataset directory
DATASET_DIR = "/home/nathanvititoe/CSC370/AircraftClassifier/aircraft_dataset/small_subset"

# apply labels to classes (6 of each, [fighter, bomber, helicopter])
CLASS_MAP = {
    # Fighters
    "F16": 0, "F22": 0, "J35": 0, "Mig29": 0, "Su57": 0, "YF23": 0,
    # Bombers
    "B1": 1, "B2": 1, "B21": 1, "B52": 1, "Tu95": 1, "Tu160": 1,
    # Helicopters
    "AH64": 2, "CH47": 2, "Ka27": 2, "Ka52": 2, "Mi24": 2, "Mi26": 2, "Mi28": 2, "UH60": 2
}

CLASS_NAMES = {0: "Fighter", 1: "Bomber", 2: "Helicopter"}

def setup_dataset(): 
    # get image counts
    dataset_evaluation(DATASET_DIR, CLASS_MAP, CLASS_NAMES)

    # get list of image paths and labels for training
    image_labels_list = build_dataset_list(DATASET_DIR, CLASS_MAP)

    # split the model 
    train_ds, val_ds = dataset_split(image_labels_list)

    # get dataset loaders
    return get_loaders(train_ds, val_ds)
