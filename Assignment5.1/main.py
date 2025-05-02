# suppress warnings and TF logs
import warnings
import os
warnings.filterwarnings("ignore") 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.random.set_seed(42) 
tf.get_logger().setLevel('FATAL')
tf.autograph.set_verbosity(0,True)

from src.dataset_prep.setup_dataset import setup_dataset
from src.model.build_and_train import build_model, train_model, compile_model


# force gpu usage
assert tf.config.list_physical_devices('GPU'), "No GPU available. Exiting."
# Lock TensorFlow to use only the first GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')  # hide CPU
    tf.config.experimental.set_memory_growth(gpus[0], True)

# TODO: 
# turn on spectrograms after debug
# consider using kapre for spectrogram creation

# directory paths
DATASET_DIR = './dataset/dataset_folds'
CSV_PATH = './dataset/UrbanSound8K.csv'

# config variables
valid_split = 0.2 # % of dataset to use for validation
BATCH_SIZE = 32 # num of files per sample 
SAMPLE_RATE = 22050 # default sample rate (from kaggle examples)
duration_length = 4 # duration of audio (seconds)
NUM_EPOCHS = 20

header_title_indent = " " * 18
header_line_indent = " " * 15
def main():
    print("introduction here\n\n")

    check_gpu() # check if gpu is being used

    print(f"\n{header_title_indent}Prepare the Datasets")
    print(f"{header_line_indent}--------------------------")

    train_ds, val_ds, label_names = setup_dataset(DATASET_DIR, CSV_PATH, BATCH_SIZE, valid_split)
    NUM_CLASSES = len(label_names)

     # get input shape from one sample
    for waveform, _ in train_ds.take(1):
        input_shape = waveform.shape[1:]
        break


    print(f"\n\n{header_title_indent}Build and Train the Model")
    model = build_model(input_shape, NUM_CLASSES)
    compile_model(model)
    train_model(model, train_ds, val_ds, epochs=NUM_EPOCHS)


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
