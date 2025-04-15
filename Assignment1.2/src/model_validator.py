# allows us to load a model from file
from tensorflow.keras.models import load_model # type: ignore
# use the dataset_loader so that the dataset is prepared the same as during training
from src.dataset_loader import load_dataset

# function to validate the trained models accuracy w/ the validation dataset
def model_validator(data_dir, img_size, batch_size):
    # load trained model from file
    model = load_model("trainedModels/aircraft_classifier.keras")

    # load the validation dataset
    _, val_ds, _ = load_dataset(data_dir, img_size, batch_size, cache=False)

    # evaluate the model, returning loss (err) and accuracy
    loss, acc = model.evaluate(val_ds)

    # output the accuracy/loss
    print(f"\nValidation Accuracy: {acc:.2%}\n")
    print(f"\nValidation Loss (confidence): {loss:.2f}\n Loss > 3.5 is considered guessing")
