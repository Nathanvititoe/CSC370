# pytorch
import torch   # type: ignore

# pytorch neural network methods
import torch.nn as nn  # type: ignore

# pretrained models import
import torchvision.models as models  # type: ignore

# Function to build pretrained base, custom head
def build_model(num_classes, dropout_rate=0.2, dense_units=256):
    # Load EfficientNetB3 trained on imageNet
    # - EfficientNet will identify which features are important
    base_model = models.efficientnet_b3(weights='IMAGENET1K_V1')
    
    pretrained_feature_output = base_model.classifier[1].in_features # get num of features from base_model
    
    # Strip off the original top layer to add a custom one
    base_model.classifier = nn.Identity()

    # create a custom top layer classifier
    #   - custom classifier determines which features mean what
    classifier = nn.Sequential(
        nn.BatchNorm1d(pretrained_feature_output), # normalize
        nn.Linear(pretrained_feature_output, dense_units), # 
        nn.ReLU(), # apply activation function for non-linearity
        nn.Dropout(dropout_rate), # drop a % of neurons every pass
        nn.Linear(dense_units, num_classes) # converts raw output to output classes 
    )

    # combined class for easy calling elsewhere
    class CombinedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = base_model # init pretrained base
            self.classifier = classifier # init custom classifier

        def forward(self, x):
            x = self.base_model(x) # extract important features w/ efficientNet
            x = self.classifier(x) # determine what they mean w/ custom layers
            return x

    return CombinedModel()
    

    # dont use tf model anymore, too complex for gpu acceleration
    # # import tf library and keras
# from tensorflow import keras # type: ignore
# from tensorflow.keras.models import Sequential, Model # type: ignore
# from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization # type: ignore
# from tensorflow.keras.applications import EfficientNetB3 # type: ignore
# from tensorflow.keras.optimizers import Adamax # type: ignore
# from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
# from tensorflow.keras import regularizers # type: ignore

# def build_pretrained_model(input_shape, num_classes, dropout_rate, learning_rate, dense_units):
#     # try loading a pretrained model (exclude top layer)
#     # Load EfficientNetB3 as base model
#     base_model = keras.applications.EfficientNetB3(
#         include_top=False,
#         weights="imagenet",
#         input_shape=input_shape,
#         pooling="max"
#     )

#     # build the convNet
#     model = Sequential([
#         base_model, # pretrained  model
#         BatchNormalization(), # normalize
        
#         # hidden custom layers w/ 256 filters and L2 regularization (penalize large weights to prevent overfit)
#         # Dense(256, activation='relu', kernel_regularizer= regularizers.l2(0.01)),
#         Dense(dense_units, activation='relu', kernel_regularizer= regularizers.l2(0.01)), # dynamic dense filters for optuna study
        
#         # Dropout(0.2),
#         Dropout(dropout_rate), # use dynamic dropout for optuna study
        
#         Dense(num_classes, activation='softmax')  # change depending on dataset
#     ])

#     # Compile the model
#     # model.compile(optimizer=Adamax(learning_rate=0.001),
#     model.compile(optimizer=Adamax(learning_rate=learning_rate), # use dynamic LR for optuna study
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])

#     model.summary()
#     return model