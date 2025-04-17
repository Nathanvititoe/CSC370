# import tf library and keras
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.optimizers import Adamax # type: ignore
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy # type: ignore
from tensorflow.keras.applications import EfficientNetB3 # type: ignore

from tensorflow.keras.models import Model # type: ignore

# from online example
from tensorflow.keras.layers import Dense, Dropout, Flatten # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# data augmentation
train_gen = ImageDataGenerator(
        rescale=1.0/255,          
        rotation_range=30,         
        width_shift_range=0.2,     
        height_shift_range=0.2,    
        shear_range=0.2,           
        zoom_range=0.2,            
        horizontal_flip=True,      
        vertical_flip=True,        
        brightness_range=[0.7, 1.3], 
        fill_mode='nearest' 
)
test_gen = ImageDataGenerator(rescale = 1.0/255)

def build_pretrained_model(input_shape, num_classes, dropout_rate, learning_rate):
    # Load a pretrained model (exclude top layer)
    base_model = EfficientNetB3(input_shape=input_shape, include_top=False, weights='imagenet', pooling='avg')
    base_model.trainable = False  # freeze for now

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # inputs = keras.Input(shape=input_shape)
    # x = preprocess_input(inputs)  
    # # x = layers.Rescaling(1./255)(inputs) # not for efficientNet
    # x = base_model(x, training=False)
    # x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Dropout(dropout_rate)(x)
    # outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    optimizer  = Adamax(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Convulutional neural networks: Uses filters to scan an image, producing a feature map that's passed to the next layer
# builds a CNN model for image classification (convulutional neural network)
def build_custom_model(input_shape, num_classes):
     # augment data to prevent overfitting
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),# flip images
        layers.RandomRotation(0.2), # random image rotations
        # layers.RandomContrast(0.05) # slight shadows/lighting filter
    ])

    # creates sequential CNN model
    model = keras.Sequential([
        keras.Input(shape=input_shape),# define input shape
        data_augmentation, # augment training data to prevent overfitting
        layers.Rescaling(1./255), # normalize pixels (0-1 range instead of 0-255)

        # first convolutional layer
        layers.Conv2D(16, (3, 3), activation='relu'),

        # second convolutional layer
        layers.Conv2D(32, (3, 3), activation='relu'),
        # layers.MaxPooling2D(), # downsamples images (shrinks the detail, keeping important features)

        # third convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),   

        # layers.MaxPooling2D(), # downsamples images (shrinks the detail, keeping important features)
        # fourth convolutional layer
        layers.Conv2D(128, (3, 3), activation='relu'),

        layers.MaxPooling2D(), # more downsampling

        # flatten and connect to dense layers 
        layers.Flatten(), # converts 3d into 1d vector
        # layers.GlobalAveragePooling2D(), # takes average of 3d vectors 
        layers.Dense(128, activation='relu'), # look for relationship between the flattened features
        layers.Dropout(0.1), # randomly disables 10% of nodes during training (reduce overfitting)

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