from keras import layers, models, optimizers # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from keras.applications import EfficientNetB0 # type: ignore

# function to build the CNN layers and filters
    # use transfer learning to get a more efficient model (base: efficientNetB0, classification layer: custom)
def build_model(input_shape, num_classes):
    base_model = EfficientNetB0(
        include_top=False,  # dont use pretrained classification layer
        input_shape=input_shape, # use our defined input shape
        pooling='avg', # global avg pooling to flatten output
        weights='imagenet'  # use weights from training on imagenet
    )
   
    base_model.trainable = False  # freeze the pretrained base (dont let it learn)
    
    inputs = layers.Input(shape=input_shape) # define inputs/input shape
    x = base_model(inputs) # pretrained model w/o top layer
    x = layers.Dense(128, activation='relu')(x) # dense layer to learn specific features for this goal
    # x = layers.Dropout(0.1)(x) # drop 10% of neurons to prevent overfitting
    outputs = layers.Dense(num_classes, activation='softmax')(x) # final output layer

    model = models.Model(inputs, outputs)

    return model
    
# function to compile and train the model
def compile_and_train(model, final_train_ds, final_val_ds, class_weights):
    print("\n    Compiling the Model...")
    model.compile(
        optimizer=optimizers.Adamax(), # adaptive learning rate optimizer
        loss="sparse_categorical_crossentropy", # used for int labels and multi-classification tasks
        metrics=["accuracy"], # monitor the accuracy
    )

    # callbacks to improve training performance and accuracy
    #   Early stopping will quit early when the val_accuracy doesnt improve
    early_stop = EarlyStopping(
        monitor="val_accuracy", # what value to monitor
        patience=4,  # how many epochs to wait before stopping
        restore_best_weights=True # roll back to the epoch w/ the highest val_acc
        )
    
    # LR reducer will divide the LR by 2 whenever it hasnt improved over 3 epochs
    #   Only necessary for higher epoch counts bc Adamax already dynamically adjusts LR
    reduce_lr = ReduceLROnPlateau(
        monitor="val_accuracy", # what value to monitor
        factor=0.5, # how much to divide LR by
        patience=2, # how many epochs to wait before dropping
        min_lr=1e-7 # set minimum LR (.0000001)
        )

    print("\n    Training the Model...\n")
    model_history = model.fit(
        final_train_ds, # training data
        validation_data=final_val_ds, # validation data
        epochs=10, # number of epochs to run
        callbacks=[early_stop, reduce_lr], # define callbacks (Early stop, LR reducer)
        class_weight=class_weights, # tell model class distribution
        verbose=1,  # output logs
    )
    return model_history
