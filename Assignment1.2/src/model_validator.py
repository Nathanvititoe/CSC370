# allows us to load a model from file
from tensorflow.keras.models import load_model # type: ignore
# use the dataset_loader so that the dataset is prepared the same as during training
from src.dataset_loader import load_dataset

# function to validate the trained models accuracy w/ the validation dataset
def model_validator():
    data_path = "./aircraft_dataset/crop" # path to dataset
    img_size = (180, 180) # size of images
    batch_size = 32 # num of imgs per batch

    # load trained model from file
    model = load_model("trainedModel/aircraft_classifier.h5")

    # load the validation dataset
    _, val_ds, class_names = load_dataset(data_path, img_size, batch_size)

    # evaluate the model, returning loss (err) and accuracy
    loss, acc = model.evaluate(val_ds)

    # output the accuracy
    print(f"\nValidation Accuracy: {acc:.2%}\n")
