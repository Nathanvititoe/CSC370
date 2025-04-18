# import libraries
import tensorflow as tf
from src.dataset_loader import load_dataset # dataset loader function
from src.model_builder import build_pretrained_model  # function that builds the pretrained model
from src.model_builder import build_custom_model  # function that builds the custom model
from src.model_visualizer import visualize_predictions # function to visualize predictions
from src.model_validator import model_validator # function to validate trained models accuracy
from src.custom_prediction import predict_image # function to run prediction on user image

# allow dynamic learning rate based on validation loss
from tensorflow.keras.callbacks import ReduceLROnPlateau # type: ignore

# stops training early if model isnt learning
from tensorflow.keras.callbacks import EarlyStopping # type: ignore 

# output description to the user
def introduction():
    print("\n--- Military Aircraft Image Classifier ---")
    print("This program uses machine learning to identify the type of military aircraft in a picture.")
    print("It trains a model using labeled aircraft pictures, evaluates the model’s accuracy,")
    print("and visualizes how well it can predict different aircraft types.")
    print("After training, you'll be able to test the model using your own image.")
    print("\n------------------------------------------------------------\n")

# main function, controls the model process from start to finish
#   Assigns labels using tensorflow
#   Builds the model using keras, from tensorflow
#   Trains the model using model.fit() from tensorflow
#   Saves model to file using model.save() from tensorflow
#   Validates the model against validation data using model.evaluate() from tensorflow
#   Visualizes predictions using MatPlotLib and tensorflow predictions
def train_model():
    introduction() # description of project for user
    full_dataset_path = "./aircraft_dataset/crop"  # path after dataset is unzipped
    subset_path = "./aircraft_dataset/aircraft_subset" # subset of aircraft classes
    small_subset ="./aircraft_dataset/small_subset"

    data_path = full_dataset_path # what dataset we are using

    img_size = (180,180) # ensure all images are the same size
    batch_size = 32 # num of imgs in each training batch

    print("\nStep 1: Assign Labels...")
    print("Loading aircraft images and assigning labels based on folder names...\n")
    # load images and give them labels
    train_ds, val_ds, class_names = load_dataset(data_path, img_size, batch_size)

    print(f"\nLearning on {len(class_names)} aircraft types")
    print(f"\nAircraft types: {class_names}")

    print("\nStep 2: Build the Model...")
    print("Creating a CNN (Convulutional Neural Network) model to learn patterns in aircraft images.\n")
    # Build the model
    #   input_shape is the image size and we want all 3 colors (RGB)
    #   num_classes is the num of different aircraft types we will predict
    model = build_pretrained_model(input_shape=img_size + (3,), num_classes=len(class_names))


    # stops training early if validation accuracy doesn't improve for 3 epochs
    early_stop = EarlyStopping(
        monitor='val_accuracy', # what to watch for
        min_delta=0.001, # will stop early if < .1% improvement between patience epochs
        patience=3, # number of epochs improvement must be seen across
        restore_best_weights=True # roll back to best weights
    )

    # # reduce learning rate if validation loss isnt improving
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_delta=0.005,
        cooldown=1,
        min_lr=1e-6,
        verbose=1
    )

    print("\nStep 3: Train the Model...")
    print("Pass over the dataset 30 times (30 epochs), please wait... \n")

    # Train the model
    #   defines the validation set, and how many epochs (passes over the training data)
    model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[early_stop, reduce_lr])

    print("\nStep 4: Save the Model...")
    print("Saving the trained model to a folder to use later...\n")
    # save the trained model to the trainedModel directory
    model.save("trainedModels/aircraft_classifier.keras")

    print("\nStep 5: Test the model...")
    print("Testing the model with images it hasn't seen before...\n")
    # test the model w/ the validation dataset
    model_validator(data_path, img_size, batch_size)

    print("\nStep 6: Visualize the Data...")
    print("Displaying sample predictions, Green(Correct), Red(incorrect)...")
    # visualize the model predictions w/ a grid
    visualize_predictions(model, val_ds, class_names)
    print("Check the Visualizations folder for a png file visualization.\n")

    print("-----------------------------------")
    print("Training and Testing Completed.")
    print("------------------------------------\n")

    print("You can now test the model yourself by uploading your own image of a military aircraft.\n")
    predict_image(model, class_names, img_size)

# call our main function
train_model()
