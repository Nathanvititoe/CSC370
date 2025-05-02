import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks  # type: ignore
from src.ui.pretty_logging import RichProgressCallback
from kapre.composed import get_melspectrogram_layer
from tensorflow.keras.applications import EfficientNetB0 # type: ignore

# Build a simple CNN for spectrogram classification
def build_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    # Convert waveform -> spectrogram
    x = get_melspectrogram_layer(
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        return_decibel=True,
        input_data_format='channels_last',
        output_data_format='channels_last'
    )(inputs)

    x = layers.Resizing(224, 224)(x)
    x = layers.Conv2D(3, (3, 3), padding='same')(x) 

    # Load EfficientNet with known input shape
    base_model = EfficientNetB0(include_top=False,weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True
        
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)


    return models.Model(inputs=inputs, outputs=outputs)
    
# Compile the model
def compile_model(model, learning_rate=1e-2):
    model.compile(
        optimizer=optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model



# Train the model
def train_model(model, train_ds, val_ds, epochs=20):
    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=0, # turn off default logs
    callbacks=[
        RichProgressCallback(),
        callbacks.EarlyStopping(patience=3,  min_delta=1e-5,restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
    ]
    )
   
    return history