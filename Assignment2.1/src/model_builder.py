# import tf library and keras
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy # type: ignore


# Convulutional neural networks: Uses filters to scan an image, producing a feature map that's passed to the next layer
# builds a CNN model for image classification (convulutional neural network)
def build_model(input_shape, num_classes):
     # augment data to prevent overfitting
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),# flip images
        layers.RandomFlip("vertical"),# flip images
        layers.RandomRotation(0.05), # random image rotations
        # layers.RandomContrast(0.05) # slight shadows/lighting filter
    ])

    # creates sequential CNN model
    model = keras.Sequential([
        keras.Input(shape=input_shape),# define input shape
        data_augmentation, # augment training data to prevent overfitting
        layers.Rescaling(1./255), # normalize pixels (0-1 range instead of 0-255)


        # first convolution
        layers.Conv2D(16, (3, 3), activation='relu'),

        # second 
        layers.Conv2D(32, (3, 3), activation='relu'),
        # layers.MaxPooling2D(), # downsamples images (shrinks the detail, keeping important features)

        # second convolution
        layers.Conv2D(64, (3, 3), activation='relu'),   

        # layers.MaxPooling2D(), # downsamples images (shrinks the detail, keeping important features)

        # third convolution
        layers.Conv2D(128, (3, 3), activation='relu'),

        layers.MaxPooling2D(), # more downsampling

        # flatten and connect to dense layers 
        layers.Flatten(), # converts 3d into 1d vector
        # layers.GlobalAveragePooling2D(), # takes average of 3d vectors 
        layers.Dense(128, activation='relu'), # look for relationship between the flattened features
        layers.Dropout(0.1), # randomly disables 10% of nodes during training (reduce overfitting)
        # Test removing dropout and increasing augmentation
        # layers.Dropout(0.2), # disable 20% of nodes to prevent overfitting

        # output layer
        layers.Dense(num_classes, activation='softmax') # outputs a probability for each class of plane
    ])

    metrics=['accuracy', SparseTopKCategoricalAccuracy(k=3)]   

    # compile the model
    #   optimizer calculates which direction would reduce error and updates weights
    #   sparse_categorical_crossentropy will convert labels to one hot vectors (0,1)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=metrics)
    
    # return completed model to be trained
    return model