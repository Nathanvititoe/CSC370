from keras_core import layers, models, optimizers # type: ignore
from keras_core.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from keras_core.applications import MobileNetV2


# function to build the CNN layers and filters
def build_model(input_shape, num_classes=3):
    base_model = MobileNetV2(
        include_top=False,  # Don't include final classifier layer
        input_shape=input_shape,
        pooling='avg',
        weights='imagenet'  # Load pretrained weights
    )
   
    base_model.trainable = False  # Freeze base for now

    model = models.Sequential([
        base_model,
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# function to compile and train the model
def compile_and_train(model, train_loader, val_loader):
    print("\nCompiling the Model...")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks to improve training
    early_stop = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-6)

    print("\nTraining the Model...")
    model_history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=10,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )
    return model_history
