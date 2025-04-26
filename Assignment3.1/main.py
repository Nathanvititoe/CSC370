from src.dataset_setup.setup import setup_dataset
from src.user_experience.ux import introduction, visualize_history, visualize_predictions
from src.model_training.model_builder import build_model, compile_and_train
# dataset directory
DATASET_DIR = "./small_subset"

# apply integer labels to aircraft types
CLASS_MAP = {
    # Fighters
    "F22": 0, "J35": 0, "Mig29": 0, "Su57": 0, "YF23": 0,

    # Bombers
    "B1": 1, "B2": 1, "B21": 1, "B52": 1, "Tu95": 1, "Tu160": 1, "Tu22M": 1,

    # Helicopters
    "AH64": 2, "CH47": 2, "Ka27": 2, "Ka52": 2,
    "Mi24": 2, "Mi26": 2, "Mi28": 2, "UH60": 2,
    "Z10": 2, "Z19": 2
}

# apply integer labels to each class of aircraft
CLASS_NAMES = {0: "Fighter", 1: "Bomber", 2: "Helicopter"}

# set img size, input shape for tensor flow and the size of the batches for training
img_size = (224, 224)
input_shape = (224, 224, 3) # tensor shape (224x224) w/ all 3 color channels (RGB)
batch_size = 64
num_classes = len(CLASS_NAMES)


# control main program logic/flow
def main():
    # output description of the program to the user
    introduction()
    
    # prepare the dataset
    print("Preparing the Dataset...")
    final_train_ds, final_val_ds = setup_dataset(DATASET_DIR, CLASS_MAP, CLASS_NAMES, img_size, batch_size)

    # build the model
    print("\n\nBuilding the Model...")
    model = build_model(input_shape, num_classes)

    # Train the model on the dataset
    print("\nStep 3: Compiling and Training the Model...")
    model_history = compile_and_train(model, final_train_ds, final_val_ds)
    
    # Test the Model on a validation set
    print("\nStep 4: Evaluating Model Performance...")
    loss, accuracy = model.evaluate(final_val_ds, verbose=1)
    
    # output test scores
    print(f"\n Final Validation Accuracy: {accuracy * 100:.2f}%")
    print(f" Final Validation Loss: {loss:.4f}")
    
    print("Step 5: Visualize...")

    # graph val loss v. val acc
    print("Visualizing Performance...")
    visualize_history(model_history)

    # visualize the predictions
    print("Visualizing Predictions...")
    visualize_predictions(model, final_val_ds, CLASS_NAMES, num_images=12)


main()
