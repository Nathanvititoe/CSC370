# suppress warnings and TF logs
import warnings
import os
warnings.filterwarnings("ignore") 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

from src.prep_data.preprocess import load_data_from_folds
from src.audio_classifier.build_and_train import create_classifier

# force gpu usage
assert tf.config.list_physical_devices('GPU'), "No GPU available. Exiting."

# TODO: 
# turn on spectrograms when done training
# add more clean visualizations
# create more logs/print statements

# directory paths
AUDIO_PATH = './dataset/dataset_folds'
CSV_PATH = './dataset/UrbanSound8K.csv'

# config variables
valid_split = 0.2 # % of dataset to use for validation
BATCH_SIZE = 32 # num of files per sample 
SAMPLE_RATE = 16000 # sample rate to downsample to
DURATION_SEC = 4 # time length of audio file (seconds)
TARGET_SAMPLES = SAMPLE_RATE * DURATION_SEC # technical length of file (samples per sec * # of seconds)
NUM_EPOCHS = 20

header_title_indent = " " * 18
header_line_indent = " " * 15
def main():
    print("\n\nintroduction here\n\n")

    check_gpu() # check if gpu is being used

    print(f"\n{header_title_indent}Prepare the Datasets")
    print(f"{header_line_indent}--------------------------")
    features, labels = load_data_from_folds(AUDIO_PATH, CSV_PATH, SAMPLE_RATE, TARGET_SAMPLES)
    num_classes = len(np.unique(labels))

    # split the dataset
    train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)

    print(f"\n\n{header_title_indent}Build and Train the Audio Classifier\n\n")
    audio_classifier = create_classifier(num_classes)

    classifier_history = audio_classifier.fit(
        train_features, train_labels,
        validation_data=(val_features, val_labels),
        epochs=20,
        batch_size=32
        )
    #TODO: use history to create visualizations

    print(f"\n\n\n{header_title_indent}Test the Audio Classifier")
    print(f"{header_line_indent}-------------------------------\n")
    print("Final Evaluation:")
    loss, acc = audio_classifier.evaluate(val_features, val_labels)
    print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {acc:.4f}\n\n")

    print("Exiting...")


# function to check if gpu is being used
def check_gpu():
    gpus = tf.config.list_physical_devices('GPU') # list devices tf sees
    if gpus:
        details = tf.config.experimental.get_device_details(gpus[0])
        gpu_name = details.get("device_name", "GPU:0")  # use name or GPU:0 
        print(f"TensorFlow is using device: {gpu_name}\n\n")
    else:
        print("TensorFlow is NOT using a GPU.\n\n")

# run the main function
main()
