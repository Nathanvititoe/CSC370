import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore

# This function loads an image file and uses the trained model to predict its class
def predict_image(model, class_names):
    print("\nWould you like to classify a new image? (yes/no)")
    user_response = input("Enter 'yes' to continue: ").strip().lower()

    if user_response != "yes":
        print("Exiting...")
        return

    # prompt the user for an image
    img_path = input("Enter the full path to the aircraft image (.jpg or .png): ").strip()

    try:
        # load the user's image and preprocess
        img = image.load_img(img_path, target_size=(180, 180)) # load and resize
        img_array = image.img_to_array(img) # convert img to arr for tensorflow
        
        # simulate a batch size of one and normalize pixel values
        img_array = np.expand_dims(img_array, axis=0) / 255.0 

        # have the model predict
        predictions = model.predict(img_array) 
        predicted_class = class_names[np.argmax(predictions)] # get the prediction as a class name

        # output prediction result
        print("---------------------------------")
        print(f"\nThis aircraft is a/an: {predicted_class}")
        print("----------------------------------")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the file path is correct and the image is valid.")
