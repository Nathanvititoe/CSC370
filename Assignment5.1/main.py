from src.dataset_prep.setup_dataset import setup_dataset
from src.model.build_and_train import build_model, train_model, compile_model
import tensorflow as tf
import warnings
import os

# suppress warnings
warnings.filterwarnings("ignore") 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TODO: 
# turn on spectrograms after debug

# directory paths
DATASET_DIR = './dataset/dataset_folds'
CSV_PATH = './dataset/UrbanSound8K.csv'

# config variables
valid_split = 0.2 # % of dataset to use for validation
BATCH_SIZE = 32 # num of files per sample 
SAMPLE_RATE = 22050 # default sample rate (from kaggle examples)
duration_length = 4 # duration of audio (seconds)
NUM_EPOCHS = 20
def main():
    print("introduction here")

    gpus = tf.config.list_physical_devices('GPU') # list devices tf sees
    # log if using gpu for debugging
    if gpus:
        print(f"TensorFlow is using GPU: {[gpu.name for gpu in gpus]}")
    else:
        print("TensorFlow is NOT using a GPU.")

    print("\n\n\n-------------------------------------------")
    print("Prepare the Datasets")
    print("----------------------------------------\n")

    train_ds, val_ds, label_names = setup_dataset(DATASET_DIR, CSV_PATH, BATCH_SIZE, SAMPLE_RATE, duration_length, valid_split)
    NUM_CLASSES = len(label_names)

     # get input shape from one sample
    for spec, _ in train_ds.take(1):
        input_shape = spec.shape[1:]
        break

    print("\n\n\n-------------------------------------------")
    print("Build the Model")
    print("------------------------------------------\n")
    model = build_model(input_shape, NUM_CLASSES)
    compile_model(model)

    print("\n\n\n-------------------------------------------")
    print("Train the Model")
    print("------------------------------------------\n")
    train_model(model, train_ds, val_ds, epochs=NUM_EPOCHS)


    print("Exiting...")
    print("-------------------------------------------")

main()