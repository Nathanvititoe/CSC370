
# %% [markdown]
# Install all dependencies (dont log output)
# %%
import subprocess
from pathlib import Path
try:
    reqs = Path.home() / "CSC370" / "Assignment5.1" / "requirements.txt"
    success = subprocess.run(["pip", "install", "-r", str(reqs)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if success:
        print("Successfully installed all dependencies")
except Exception as e:
    print(f"Failed to install dependencies: {e}")
# %% [markdown]
# Setup - imports and variables
# %%
# suppress warnings and TF logs
import warnings
import os
warnings.filterwarnings("ignore") 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import libraries and files
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from termcolor import colored
import pandas as pd
from src.ui.introduction import introduction, get_acc_color
from src.prep_data.evaluate_dataset import plot_dataset
from src.ui.visualization import visualize_stats, plot_confusion_matrix
from src.prep_data.preprocess import load_data_from_folds
from src.audio_classifier.build_model import create_classifier, train_classifier
from src.ui.cleanup import final_cleanup
from src.ui.user_input import get_prediction

# force gpu usage
assert tf.config.list_physical_devices('GPU'), "No GPU available. Exiting."

# TODO: 
# fix classifier layer
# cleanup requirements.txt
# color prediction label and actual label


# directory paths
AUDIO_ROOT_PATH = './dataset/dataset_folds'
CSV_PATH = './dataset/UrbanSound8K.csv'

# config variables
valid_split = 0.2 # % of dataset to use for validation
BATCH_SIZE = 16 # num of files per sample(32 default--good result) 
SAMPLE_RATE = 16000 # sample rate to downsample to
DURATION_SEC = 4 # time length of audio file (seconds)

NUM_EPOCHS = 100
label_names = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
]

# function to check if gpu is being used
def check_gpu():
    gpus = tf.config.list_physical_devices('GPU') # list devices tf sees
    if gpus:
        details = tf.config.experimental.get_device_details(gpus[0])
        gpu_name = details.get("device_name", "GPU:0")  # use name or GPU:0 
        print(f"\n\nTensorFlow is using device: {gpu_name}\n")
    else:
        print("\n\nTensorFlow is NOT using a GPU.\n")

# %% [markdown]
# Introduction & Check Device 
# %%
# check device and introduction
check_gpu() # check if gpu is being used
introduction() # output description to user

# %% [markdown]
# Prepare and Plot Data

# plot dataset
plot_dataset(CSV_PATH) # visualize file and class distribution

# %% [markdown]
# Prepare and Visualize Data

# %%
# load, split Dataset and plot audio
df = pd.read_csv(CSV_PATH) # load metadata

# save 5% of the dataset as unseen files for user predictions, split remainder 80/20 for train/val
trainval_df, user_predict_df = train_test_split(df, test_size=0.05, stratify=df["classID"], random_state=42)
train_df, val_df = train_test_split(trainval_df, test_size=0.2, stratify=trainval_df["classID"], random_state=42)

# load data for each split
train_features, train_labels = load_data_from_folds(AUDIO_ROOT_PATH, train_df, SAMPLE_RATE, DURATION_SEC, "Training") # training features/labels
val_features, val_labels     = load_data_from_folds(AUDIO_ROOT_PATH, val_df, SAMPLE_RATE, DURATION_SEC, "Validation") # validation features/labels
num_classes = len(np.unique(train_labels)) # get total number of classes


# %% [markdown]
# Build and Train the Classifier

# %%
# create the custom classifier top layer
audio_classifier = create_classifier(num_classes)
classifier_history = train_classifier(audio_classifier, train_features, train_labels, val_features, val_labels, NUM_EPOCHS, BATCH_SIZE)
    
# %% [markdown]
# Evaluate and Plot Results

# %%
# output eval and results
print("Final Evaluation:")
loss, acc = audio_classifier.evaluate(val_features, val_labels) # evaluate model
acc_color = get_acc_color(acc)
print(colored(f"\nValidation Loss:     {loss:.4f}\n", "blue"))
print(colored(f"Validation Accuracy: {acc:.4f}\n", acc_color))

# plot training results
plot_confusion_matrix(audio_classifier, val_features, val_labels, label_names) # create confusion matrix
visualize_stats(classifier_history) # visualize the loss/acc

# %% [markdown]
# Get User Input and Output Prediction

# %%
# take user input, get prediction, display to user
get_prediction(audio_classifier, SAMPLE_RATE, DURATION_SEC, label_names, user_predict_df, AUDIO_ROOT_PATH)

final_cleanup()
print("Exiting...")
