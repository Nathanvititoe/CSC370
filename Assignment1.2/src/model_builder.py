# import tf library and keras
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers # type: ignore


# Convulutional neural networks: Uses filters to scan an image, producing a feature map that's passed to the next layer
# builds a CNN model for image classification (convulutional neural network)
def build_model(input_shape, num_classes):
     # augment data to prevent overfitting
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),# flip images
        layers.RandomRotation(0.15), # random image rotations
        layers.RandomZoom(0.15), # zoom images in/out
        layers.RandomTranslation(0.1,0.1), # slight x/y shift
        layers.RandomContrast(0.1) # slight shadows/lighting filter
    ])

    # creates sequential CNN model
    model = keras.Sequential([
        keras.Input(shape=input_shape),# define input shape to prevent warnings
        data_augmentation, # augment training data to prevent overfitting
        layers.Rescaling(1./255, input_shape=input_shape), # normalize pixels (0-1 range instead of 0-255)

        # Testing back to back layers before downsampling to try to learn more complex edge combos
        # first convolutional block (2 layers, then downsample)
        layers.Conv2D(32, (3, 3), activation='relu'), # applies filters to help detect features
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(), # downsamples images (shrinks the detail, keeping important features)

        # second convolutional block (2 layers, then downsample)
        layers.Conv2D(64, (3, 3), activation='relu'), # applies 64 filters now for more complex patterns
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(), # more downsampling

        # third convolutional block (1 layer, then downsample)
        # testing for improved accuracy
        layers.Conv2D(128, (3,3), activation='relu'), # applies 129 filters for complexity
        layers.MaxPooling2D(), # more downsampling

        # flatten and connect to dense layers 
        layers.Flatten(), # converts 3d into 1d vector
        layers.Dense(128, activation='relu'), # look for relationship between the flattened features

        # Test removing dropout and increasing augmentation
        # layers.Dropout(0.1), # disable 10% of nodes to prevent overfitting
        # layers.Dropout(0.025) # test disable 2.5% of nodes during training

        # output layer
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