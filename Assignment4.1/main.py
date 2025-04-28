from pathlib import Path
import torch
import warnings
import os

from src.dataset_setup.setup import setup_dataset
from src.dataset_setup.dataset_loader import build_dataset_list, create_dataset
from src.user_experience.ux import introduction, visualize_stats, visualize_predictions
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
IMG_SIZE = (96, 96) # image resolution
BATCH_SIZE = 16 # num of images per sample
NUM_CLASSES = len(CLASS_NAMES) # total number of classes
NUM_WORKERS = 4 # cpu processes
NUM_EPOCHS = 5 # num of epochs to run
#TODO:
#   class weights?
#   get running
#   break 95% accuracy
#   fix dataset, no sea or street in ds, uneven mountain set
#   include train acc in visualizer

# main logic flow control
def main():
    # ------------------------------
    # output description of the program to the user
    introduction()
    # ------------------------------

    print("---------------------------------------------------------")
    
    # log what device we are training with
    print(f"Running on device: {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    print("---------------------------------------------------------")

    print("\n\nStep 1: Prepare the Datasets...\n")
    # ------------------------------
    # prepare the training dataset
    print("\n Preparing the Training Dataset...\n")
    train_ds, training_val_ds = setup_dataset(DATASET_DIR, CLASS_MAP, CLASS_NAMES, IMG_SIZE, BATCH_SIZE,NUM_CLASSES, NUM_WORKERS)
    # ------------------------------

    # ------------------------------
    # prepare a dataset exclusively for validation
    print("\nPreparing the Validation Dataset...\n")
    test_list = build_dataset_list(val_dataset, CLASS_MAP)
    # test_loader = create_dataset(test_list, IMG_SIZE, BATCH_SIZE, NUM_WORKERS)
    test_ds = create_dataset(test_list, IMG_SIZE)

    # ------------------------------
    
    print("---------------------------------------------------------")
    
    # ------------------------------
    print("\n\nStep 2: Building and Training...")
    # build the model
    model = build_model(train_ds, training_val_ds, NUM_CLASSES, NUM_EPOCHS, BATCH_SIZE)

    # Train the model on the dataset
    model = train_model(model, train_ds)
    # ------------------------------
    
    print("---------------------------------------------------------")
    
    # ------------------------------
    # Test the Model on the exclusive validation set
    print("\n\nStep 3: Evaluate Model Performance...\n")
    y_test = [x[1] for x in test_ds]  # all the labels
    accuracy = model.score(test_ds, y=y_test) # get model accuracy
    #              # get model loss? 
    # output test scores
    print(f"Final Validation Accuracy: {accuracy * 100:.2f}%")
    # print(f" Final Validation Loss: {loss:.4f}")
    # ------------------------------

    print("---------------------------------------------------------\n")

    # ------------------------------
    # # get predictions on prediction dataset
    # print("\n\nStep 4: Get Model Predictions on the Prediction Dataset...\n")
    # pred_paths = [str(p) for p in Path(pred_dataset).rglob("*.jpg")]
    # pred_ds = create_dataset(pred_paths, IMG_SIZE)
    # all_preds = model.predict(pred_ds)
    # visualize_predictions(all_preds, pred_paths, CLASS_NAMES)
    # ------------------------------

    print("---------------------------------------------------------\n")    
    # ------------------------------
    # graph loss v. acc
    print("\n\nStep 5: Visualizing Performance...\n")
    visualize_stats(model)
    # ------------------------------

    print("---------------------------------------------------------")

    # ------------------------------
    print("\nRunning Final Cleanup...\n")
    cleanup(model, is_final=True)
    # ------------------------------
    
    print("\nExiting...")
    print("\n----------------------------------------\n")

main()