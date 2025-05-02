from tensorflow.keras import layers, models # type: ignore


def create_classifier(num_classes):
    # build classifier w/ input shape same as yamnet output shape
    audio_classifier = models.Sequential([
        layers.Input(shape=(1024,)),  # input yamnet 1024-embedding (feature vector)

        # custom classifier head
        layers.Dense(256, activation='relu'), # dense layer, 256 neurons 
        layers.Dropout(0.3), # randomly drop 30% of neurons (prevnts overfitting)
        layers.Dense(num_classes, activation='softmax')  # dense output layer, output num_classes
    ])

    # compile the model
    audio_classifier.compile(
        optimizer='adam', # use adam for optimization
        loss='sparse_categorical_crossentropy', # categorical crossentropy for multi-classification
        metrics=['accuracy'] # measure accuracy
    )

    return audio_classifier