# import tf library and keras
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers # type: ignore


# Convulutional neural networks: Uses filters to scan an image, producing a feature map that's passed to the next layer
# builds a CNN model for image classification (convulutional neural network)
def build_model(input_shape, num_classes):
     # augment data to prevent overfitting
    # data_augmentation = keras.Sequential([
    #     layers.RandomFlip("horizontal"),# flip images
    #     layers.RandomRotation(0.05), # random image rotations
    #     # layers.RandomTranslation(0.7,0.7), # slight x/y shift
    #     layers.RandomContrast(0.05) # slight shadows/lighting filter
    # ])

    # creates sequential CNN model
    model = keras.Sequential([
        keras.Input(shape=input_shape),# define input shape
        # data_augmentation, # augment training data to prevent overfitting
        layers.Rescaling(1./255), # normalize pixels (0-1 range instead of 0-255)

        # first convolutional block (2 layers, then downsample)
        #   first layer of 1st convolution (remove bias since we are batch normalizing)
        layers.SeparableConv2D(16, (3, 3), use_bias=False), # applies 16 filters to start general learning
        layers.BatchNormalization(), # attempt to normalize between layers, to speed up training
        layers.Activation('relu'), # keeps positive values, converts negative values to 0

        #   second layer of 1st convolution
        layers.SeparableConv2D(16, (3, 3), use_bias=False),
        layers.BatchNormalization(), # attempt to normalize between layers, to speed up training
        layers.Activation('relu'),
        layers.MaxPooling2D(), # downsamples images (shrinks the detail, keeping important features)

        # second convolutional block (2 layers, then downsample)
        #   first layer of 2nd convolution   
        layers.SeparableConv2D(32, (3, 3), use_bias=False), # applies 32 filters to help detect features
        layers.BatchNormalization(), # attempt to normalize between layers, to speed up training
        layers.Activation('relu'),

        #   second layer of 2nd convolution
        layers.SeparableConv2D(32, (3, 3), use_bias=False),
        layers.BatchNormalization(), # attempt to normalize between layers, to speed up training
        layers.Activation('relu'),
        layers.MaxPooling2D(), # downsamples images (shrinks the detail, keeping important features)

        # third convolutional block (2 layer, then downsample)
        #   first layer of 3rd Convolution
        layers.SeparableConv2D(64, (3, 3), use_bias=False), # applies 64 filters now for more complex patterns
        layers.BatchNormalization(), # attempt to normalize between layers, to speed up training
        layers.Activation('relu'),

        #   second layer of 3rd convolution
        layers.SeparableConv2D(64, (3, 3), use_bias=False),
        layers.BatchNormalization(), # attempt to normalize between layers, to speed up training
        layers.Activation('relu'),

        layers.MaxPooling2D(), # more downsampling

        # flatten and connect to dense layers 
        # layers.Flatten(), # converts 3d into 1d vector
        layers.GlobalAveragePooling2D(), # takes average of 3d vectors 
        layers.Dense(128, activation='relu'), # look for relationship between the flattened features

        # Test removing dropout and increasing augmentation
        # layers.Dropout(0.2), # disable 20% of nodes to prevent overfitting
        layers.Dropout(0.1), # randomly disables 10% of nodes during training (reduce overfitting)

        # output layer
        layers.Dense(num_classes, activation='softmax') # outputs a probability for each class of plane
    ])

    # test out rms instead of adam
    optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
    # optimizer='adam'

    # compile the model
    #   optimizer calculates which direction would reduce error and updates weights
    #   sparse_categorical_crossentropy will convert labels to one hot vectors (0,1)
    #   accuracy - metric to track performance
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # return completed model to be trained
    return model