import keras_core as keras # type: ignore
import tensorflow as tf
import numpy as np

from src.dataset_setup.setup import setup_dataset
from src.user_experience.ux import introduction, visualize_history, visualize_predictions
from src.model_training.model_builder import build_model, compile_and_train
# dataset directory
DATASET_DIR = "./small_subset"

# apply labels to classes
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

CLASS_NAMES = {0: "Fighter", 1: "Bomber", 2: "Helicopter"}
img_size = (224, 224)
input_shape = (224, 224, 3)
batch_size = 64
num_classes = len(CLASS_NAMES)


# control main program logic/flow
def main():
    introduction()
    
    # prepare the dataset
    print("Preparing the Dataset...")
    train_loader, val_loader, train_ds, val_ds = setup_dataset(DATASET_DIR, CLASS_MAP, CLASS_NAMES, img_size, batch_size)

    images, labels = next(iter(train_loader))
    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)
    print("Unique labels in batch:", np.unique(labels.numpy()))

    # build the model
    print("\n\nBuilding the Model...")
    model = build_model(input_shape, num_classes)

    print("CLASS_MAP used:", CLASS_MAP)
    print("Label 0 folder(s):", [k for k, v in CLASS_MAP.items() if v == 0])
    print("Label 1 folder(s):", [k for k, v in CLASS_MAP.items() if v == 1])
    print("Label 2 folder(s):", [k for k, v in CLASS_MAP.items() if v == 2])

    # Peek at a single batch
    for batch_images, batch_labels in train_loader.take(1):
        print("Batch shape:", batch_images.shape)
        print("Batch dtype:", batch_images.dtype)
        print("Min/Max pixel values:", tf.reduce_min(batch_images).numpy(), tf.reduce_max(batch_images).numpy())
        print("Sample labels:", batch_labels.numpy())


    # Train the model on the dataset
    print("\nStep 3: Compiling and Training the Model...")
    model_history = compile_and_train(model, train_loader, val_loader)
    
    # graph val loss v. val acc
    print("Visualizing Performance...")
    visualize_history(model_history)

    # visualize the predictions
    print("Visualizing Predictions...")
    visualize_predictions(model, val_loader, CLASS_NAMES, num_images=12)

    # Test the Model on a validation set
    print("\nStep 4: Evaluating Model Performance...")
    loss, accuracy = model.evaluate(val_loader, verbose=1)

    # output test scores
    print(f"\n Final Validation Accuracy: {accuracy * 100:.2f}%")
    print(f" Final Validation Loss: {loss:.4f}")

    print("\nTraining and evaluation complete!")


main()
