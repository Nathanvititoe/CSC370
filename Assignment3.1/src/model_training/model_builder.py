from keras import layers, models, optimizers # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from keras.applications import MobileNetV2 # type: ignore


# function to build the CNN layers and filters
# use transfer learning to get a more efficient model (base: mobileNet, classification layer: custom)
def build_model(input_shape, num_classes=3):
    base_model = MobileNetV2(
        include_top=False,  # dont use mobileNet classification layer
        input_shape=input_shape, # use our defined input shape
        pooling='avg', # global avg pooling ot flatten output
        weights='imagenet'  # use mobileNet weights from training on imagenet
    )
   
    base_model.trainable = False  # freeze the pretrained base (dont let it learn)

    model = models.Sequential([
        base_model, # start w mobileNet base

        # custom classification layer
        layers.Dense(128, activation='relu'), # dense layer to learn specific features for this goal
        layers.Dropout(0.34), # drop 34% of neurons to prevent overfitting
        layers.Dense(num_classes, activation='softmax') # final output layer
    ])
    return model

# function to compile and train the model
def compile_and_train(model, final_train_ds, final_val_ds):
    print("\nCompiling the Model...")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4), # 0.0001 Learning rate w/ Adam as optimizer
        loss="sparse_categorical_crossentropy", # used for int labels and multi-classification tasks
        metrics=["accuracy"], # monitor the accuracy
    )

    # callbacks to improve training performance and accuracy
    #   Early stopping will quit early when the val_accuracy doesnt improve
    early_stop = EarlyStopping(
        monitor="val_accuracy", # what value to monitor
        patience=5,  # how many epochs to wait before stopping
        restore_best_weights=True # roll back to the epoch w/ the highest val_acc
        )
    
    #   LR reducer will divide the LR by 2 whenever it hasnt improved over 3 epochs
    reduce_lr = ReduceLROnPlateau(
        monitor="val_accuracy", # what value to monitor
        factor=0.5, # how much to divide LR by
        patience=3, # how many epochs to wait before dropping
        min_lr=1e-6 # set minimum LR
        )

    print("\nTraining the Model...")
    model_history = model.fit(
        final_train_ds, # training data
        validation_data=final_val_ds, # validation data
        epochs=10, # number of epochs to run
        callbacks=[early_stop, reduce_lr], # define callbacks (Early stop, LR reducer)
        verbose=1,  # output logs
    )
    return model_history
