# import tf library and keras
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers # type: ignore


# Convulutional neural networks: Uses filters to scan an image, producing a feature map that's passed to the next layer
# builds a CNN model for image classification (convulutional neural network)
def build_model(input_shape, num_classes):
    # creates sequential model
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=input_shape), # normalize pixels (0-1 range instead of 0-255)
        layers.Conv2D(32, (3, 3), activation='relu'), # applies filters to help detect features
        layers.MaxPooling2D(), # downsamples images (shrinks the detail, keeping important features)
        layers.Conv2D(64, (3, 3), activation='relu'), # applies 64 filters now for more complex patterns
        layers.MaxPooling2D(), # more downsampling
        layers.Flatten(), # converts 3d into 1d vector
        layers.Dense(128, activation='relu'), # look for relationship between the flattened features
        layers.Dense(num_classes, activation='softmax') # outputs a probability for each class of plane
    ])

    # compile the model
    #   adam adjusts the learning rate 
    #   sparse_categorical_crossentropy will convert labels to one hot vectors (0,1)
    #   accuracy - metric to track performance
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # return completed model to be trained
    return model