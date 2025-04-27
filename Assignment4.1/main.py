from pathlib import Path

from src.dataset_setup.setup import setup_dataset
from src.dataset_setup.dataset_loader import build_dataset_list, create_dataset
from src.user_experience.ux import introduction, visualize_stats, visualize_predictions
from src.model_training.build_model import build_model, train_model
from src.model_training.cleanup import cleanup

# dataset paths
train_dataset = './dataset/seg_train/seg_train'
val_dataset = './dataset/seg_test/seg_test'
pred_dataset = './dataset/seg_pred/seg_pred'

# define dataset variables
DATASET_DIR = train_dataset
CLASS_NAMES = [d.name for d in Path(DATASET_DIR).iterdir() if d.is_dir()] # get class names from subfolder headings
CLASS_MAP = {name: index for index, name in enumerate(sorted(CLASS_NAMES))}
IMG_SIZE = (150, 150)
BATCH_SIZE = 16
NUM_CLASSES = len(CLASS_NAMES)
NUM_WORKERS = 4
NUM_EPOCHS = 10

# main logic flow control
def main():
    # ------------------------------
    # output description of the program to the user
    introduction()
    # ------------------------------
    
    print("---------------------------------------------------------")
    
    print("\n\nStep 1: Prepare the Datasets...\n")
    # ------------------------------
    # prepare the training dataset
    print("\n Preparing the Training Dataset...\n")
    train_loader, training_val_loader = setup_dataset(DATASET_DIR, CLASS_MAP, CLASS_NAMES, IMG_SIZE, BATCH_SIZE, NUM_WORKERS)
    # ------------------------------

    # ------------------------------
    # prepare a dataset exclusively for validation
    print("\nPreparing the Validation Dataset...\n")
    test_list = build_dataset_list(val_dataset, CLASS_MAP)
    test_loader = create_dataset(test_list, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, class_weights=None)
    # ------------------------------
    
    print("---------------------------------------------------------")
    
    # ------------------------------
    print("\n\nStep 2: Building and Training...")
    # build the model
    model = build_model(NUM_CLASSES, input_shape=(*IMG_SIZE, 3))

    # Train the model on the dataset
    model_history = train_model(model, train_loader, training_val_loader, BATCH_SIZE, NUM_EPOCHS)
    # ------------------------------
    
    print("---------------------------------------------------------")
    
    # ------------------------------
    # Test the Model on the exclusive validation set
    print("\n\nStep 3: Evaluate Model Performance...\n")
    loss, accuracy = model.evaluate(test_loader, verbose=1)
    # output test scores
    print(f"\n Final Validation Accuracy: {accuracy * 100:.2f}%")
    print(f" Final Validation Loss: {loss:.4f}")
    # ------------------------------

    print("---------------------------------------------------------")

    # ------------------------------
    # get predictions on prediction dataset
    print("\n\nStep 4: Get Model Predictions on the Prediction Dataset...\n")
    pred_list = [(str(p), None) for p in Path(pred_dataset).rglob("*.jpg")]
    pred_loader = create_dataset(pred_list, IMG_SIZE, BATCH_SIZE, NUM_WORKERS, class_weights=None)
    all_preds = model.predict(pred_loader)
    visualize_predictions(all_preds, pred_list, CLASS_NAMES)
    # ------------------------------

    print("---------------------------------------------------------")    

    # ------------------------------
    # graph loss v. acc
    print("\n\nStep 5: Visualizing Performance...\n")
    visualize_stats(model_history)
    # ------------------------------

    print("---------------------------------------------------------")

    # ------------------------------
    print("\nRunning Final Cleanup...\n")
    cleanup(model, is_final=True)
    # ------------------------------
    
    print("\nExiting...")
    print("\n----------------------------------------\n")

main()