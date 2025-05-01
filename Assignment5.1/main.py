from src.dataset_prep.setup_dataset import setup_dataset
import tensorflow as tf
# directory paths
DATASET_DIR = './dataset/dataset_folds'
CSV_PATH = './dataset/UrbanAudio8k.csv'

# config variables
valid_split = 0.2


def main():
    print("introduction here")
    
    gpus = tf.config.list_physical_devices('GPU') # list devices tf sees
    # log if using gpu for debugging
    if gpus:
        print(f"TensorFlow is using GPU: {[gpu.name for gpu in gpus]}")
    else:
        print("TensorFlow is NOT using a GPU.")

    print("-------------------------------------------")
    print("Prepare the Datasets...")
    print("-------------------------------------------\n")

    train_ds, val_ds, label_names = setup_dataset(DATASET_DIR, CSV_PATH, valid_split)