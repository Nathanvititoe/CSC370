from pathlib import Path
import torch
import warnings
import os

from src.dataset_setup.setup import setup_dataset
from src.dataset_setup.dataset_loader import build_dataset_list, create_dataset,get_transforms
from src.user_experience.ux import introduction, visualize_stats
from src.model_training.build_model import build_model, train_model
from src.model_training.cleanup import cleanup

# suppress warnings
warnings.filterwarnings("ignore") 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# dataset paths
train_dataset = './dataset/seg_train/seg_train'
val_dataset = './dataset/seg_test/seg_test'
pred_dataset = './dataset/seg_pred/seg_pred'

# define dataset variables
DATASET_DIR = train_dataset
CLASS_NAMES = [d.name for d in Path(DATASET_DIR).iterdir() if d.is_dir()] # get class names from subfolder headings
CLASS_MAP = {name: index for index, name in enumerate(sorted(CLASS_NAMES))}
IMG_SIZE = (224, 224) # image resolution
BATCH_SIZE = 256 # num of images per sample
NUM_CLASSES = len(CLASS_NAMES) # total number of classes
NUM_EPOCHS = 15 # num of epochs to run

# force model to use GPU or throw error
DEVICE = 'cuda' if torch.cuda.is_available() else None
if DEVICE is None:
    raise RuntimeError("GPU not available.")

# main logic flow control
def main():
    # output description of the program to the user
    introduction()
    
    print("\n\n---------------------------------------------------------")
    # log what device we are training with
    print(f"Running on device: {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("---------------------------------------------------------")


    print("\n\n---------------------------------------------------------")
    print("Step 1: Preparing the Datasets...")
    print("---------------------------------------------------------\n\n")

    # prepare the training dataset
    train_transform, val_transform = get_transforms(IMG_SIZE, DEVICE) # apply augmentation
    dataset = setup_dataset(DATASET_DIR, CLASS_MAP, CLASS_NAMES, train_transform) # build full dataset

    # prepare a dataset exclusively for validation
    test_list = build_dataset_list(val_dataset, CLASS_MAP)
    test_ds = create_dataset(test_list, val_transform)    
    
    print("\n---------------------------------------------------------")
    print("Step 2: Building and Training...")
    print("---------------------------------------------------------\n\n")
    # build the model
    classifier = build_model(DEVICE, dataset, NUM_CLASSES, NUM_EPOCHS, BATCH_SIZE)

    # Train the classifier on the dataset
    classifier = train_model(classifier, dataset)    
    
    # Test the Model on the exclusive validation set
    print("\n\n---------------------------------------------------------")
    print("Step 3: Evaluate Model Performance...")
    print("---------------------------------------------------------")
    y_test = [x[1] for x in test_ds]  # all the labels
    accuracy = classifier.score(test_ds, y=y_test) # get model accuracy

    # output test scores (val acc, val loss)
    final_val_loss = classifier.history[-1]['valid_loss']
    print(f"\n    Final Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"    Final Validation Loss: {final_val_loss:.4f}")

    # graph loss v. acc (training and validation)
    print("\n\n---------------------------------------------------------")    
    print("Step 5: Visualizing Performance...")
    print("---------------------------------------------------------\n\n")
    visualize_stats(classifier)

    print("---------------------------------------------------------")
    print("Running Final Cleanup...")
    print("---------------------------------------------------------")
    cleanup(classifier, is_final=True)
    
    print("\n\nExiting...")
    print("----------------------------------------\n")

main()