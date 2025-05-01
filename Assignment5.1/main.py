from src.dataset_prep.setup_dataset import setup_dataset
import tensorflow as tf
import warnings
import os

# suppress warnings
warnings.filterwarnings("ignore") 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# directory paths
DATASET_DIR = './dataset/dataset_folds'
CSV_PATH = './dataset/UrbanSound8K.csv'

# config variables
valid_split = 0.2 # % of dataset to use for validation
BATCH_SIZE = 32 # num of files per sample 
SAMPLE_RATE = 22050 # default sample rate (from kaggle examples)
duration_length = 4 # duration of audio (seconds)


def main():
    print("introduction here")

    gpus = tf.config.list_physical_devices('GPU') # list devices tf sees
    # log if using gpu for debugging
    if gpus:
        print(f"TensorFlow is using GPU: {[gpu.name for gpu in gpus]}")
    else:
        print("TensorFlow is NOT using a GPU.")

    print("\n\n\n-------------------------------------------")
    print("Prepare the Datasets...")
    print("-------------------------------------------\n")

    train_ds, val_ds, label_names = setup_dataset(DATASET_DIR, CSV_PATH, BATCH_SIZE, SAMPLE_RATE, duration_length, valid_split)

    print("-------------------------------------------")
    print("COMPLETED")
    print("-------------------------------------------")

main()