import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, ReduceLROnPlateau

from cleanup import EpochCleanupCallback

# create the model (base: EfficientNetB0, classification layer: custom)
class Model(nn.Module):
    # class constructor
    def __init__(self, num_classes):
        super(Model, self).__init__()
        # load EfficientNetB0 model as base
        self.model = models.efficientnet_b0(pretrained=True)
        
        # freeze all layers in the EfficientNetB0 base
        for param in self.model.parameters():
            param.requires_grad = False
        
        # convert top layer to match our number of classes
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        
        # unfreeze the top/classifier layer for training
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    # method to pass data through model
    def forward(self, input_data):
        return self.model(input_data)

# build an instance of our model
def build_model(NUM_CLASSES, NUM_EPOCHS, BATCH_SIZE):
    # force model to use GPU or throw error
    device = 'cuda' if torch.cuda.is_available() else None
    if device is None:
        raise RuntimeError("GPU not available.")
    
    # wrap the model in skorch classifier for abstraction/ease of use
    model = NeuralNetClassifier(
        Model,
        module__num_classes=NUM_CLASSES, # number of classifiable classes 
        max_epochs=NUM_EPOCHS, # number of epochs
        optimizer=optim.Adamax, # adamax optimizer, dynamic LR updates
        criterion=nn.CrossEntropyLoss, # use cross entropy for categorical classificaton labels
        iterator_train__batch_size=BATCH_SIZE, # set batch sizes
        iterator_valid__batch_size=BATCH_SIZE, # set batch sizes
        device=device # run on gpu
    )

    return model

# function to train/fit the model
def train_model(model, train_loader, training_val_loader):
    # Early stopping (stop model training if val acc isnt improving)
    early_stop = EarlyStopping(
        monitor="valid_accuracy",  # what to monitor
        patience=4, # how many epochs to wait before stopping
        restore_best_weights=True  # roll back to the best weights when stopping
    )
    
    # LR scheduler (reduce LR if val acc doesnt increase)
    reduce_lr = ReduceLROnPlateau(
        monitor="valid_accuracy",  # what to monitor
        factor=0.5, # reduce learning rate by half
        patience=2, # how many epochs to wait before reducing LR
        min_lr=1e-7 # dont go below this LR
    )

    cleanup_epoch = EpochCleanupCallback()

    # fit the model
    model_history = model.fit(
        train_loader,  # training data loader
        y=None, # dont provide labels separately
        validation_data=training_val_loader,  # validation data loader
        callbacks=[early_stop, reduce_lr, cleanup_epoch],  # callbacks for early stopping, LR reduction, and custom resource optimization
        verbose=1,  # log training data
    )

    return model_history