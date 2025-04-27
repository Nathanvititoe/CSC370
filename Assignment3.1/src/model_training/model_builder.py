from keras import layers, models, optimizers, Sequential # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from keras.applications import EfficientNetB0,MobileNetV3Small,MobileNetV3Large, EfficientNetB3, EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2# type: ignore

# preprocessing methods for different pretrained models
from keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess # type: ignore
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess # type: ignore
# Diff models validation accuracy
# provide more epochs for larger models to converge

    # EfficientNetB0(224x224) : 93.5% accuracy- small_subset/10 epochs
    # EfficientNetB0(96x96) : 85.4% accuracy- small_subset/10 epochs

    # more epochs? 
    # EfficientNetB3(224x224) : 91.3% accuracy- small_subset/10 epochs
    # EfficientNetB3(96x96) : 86% accuracy- small_subset/10 epochs   
    
    # EfficientNetV2B0(224x224) : __% accuracy- small_subset/10 epochs
    # EfficientNetV2B0(96x96) : 86.9% accuracy- small_subset/10 epochs 

    # EfficientNetV2B1(224x224) : __% accuracy- small_subset/10 epochs
    # EfficientNetV2B1(96x96) : 87.5% accuracy- small_subset/10 epochs 

    # EfficientNetV2B2(224x224) : __% accuracy- small_subset/10 epochs
    # EfficientNetV2B2(96x96) : 86.3% accuracy- small_subset/10 epochs 

    # MobileNetV3Small(224x224) : 88% - small_subset/10 epochs
    # MobileNetV3Small(96x96) : 83.6% accuracy - small_subset/10 epochs
    # MobileNetV3Small(96x96) : 84.6% accuracy - small_subset/30 epochs

    # more epochs? 
    # MobileNetV3Large(224x224) : 91% accuracy - small_subset/10 epochs
    # MobileNetV3Large(96x96) : 87.3% accuracy - small_subset/10 epochs


# function to build the CNN layers and filters
# use transfer learning to get a more efficient model (base: mobileNet, classification layer: custom)
def build_model(input_shape, num_classes):
    # add some augmentation to images to encourage generalization/prevent overfitting
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal"), # randomly horizontal flip
        layers.RandomRotation(0.1),# Rotate randomly +/- 10%
        layers.RandomZoom(0.1), # Zoom in/out randomly by +/- 10%
        layers.RandomContrast(0.1), # randomly adjust contrast by +/- 10%
    ], name="data_augmentation")

    base_model = EfficientNetV2B2(
        include_top=False,  # dont use pretrained classification layer
        input_shape=input_shape, # use our defined input shape
        pooling='avg', # global avg pooling to flatten output
        weights='imagenet'  # use weights from training on imagenet
    )
   
    base_model.trainable = False  # freeze the pretrained base (dont let it learn)
    
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = efficientnet_preprocess(x)  # preprocess for the pretrained model
    x = base_model(x, training=False) # pretrained model w/o top layer
    x = layers.Dense(128, activation='relu')(x) # dense layer to learn specific features for this goal
    x = layers.Dropout(0.4)(x) # drop 40% of neurons to prevent overfitting
    outputs = layers.Dense(num_classes, activation='softmax')(x) # final output layer

    model = models.Model(inputs, outputs)

    return model

# function to compile and train the model
def compile_and_train(model, final_train_ds, final_val_ds, class_weights):
    print("\nCompiling the Model...")
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

    print("\nTraining the Model...")
    model_history = model.fit(
        final_train_ds, # training data
        validation_data=final_val_ds, # validation data
        epochs=10, # number of epochs to run
        callbacks=[early_stop, reduce_lr], # define callbacks (Early stop, LR reducer)
        class_weight=class_weights, # tell model class distribution
        verbose=1,  # output logs
    )
    return model_history
