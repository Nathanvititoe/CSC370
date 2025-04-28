import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from torchvision.models import EfficientNet_B0_Weights
from skorch.callbacks import EarlyStopping, LRScheduler, EpochScoring

from src.model_training.cleanup import EpochCleanupCallback

# create the model (base: EfficientNetB0, classification layer: custom)
class Model(nn.Module):
    # class constructor
    def __init__(self, num_classes):
        super(Model, self).__init__()
        # load EfficientNetB0 model as base
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # freeze all layers in the EfficientNetB0 base
        for param in self.model.parameters():
            param.requires_grad = False
        
        # convert top layer to match our number of classes
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        
        # unfreeze the top/classifier layer for training
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    # method to pass data through model
    def forward(self, input_data, **kwargs):
        return self.model(input_data)

# build an instance of our model
def build_model(train_ds, train_val_ds, NUM_CLASSES, NUM_EPOCHS, BATCH_SIZE):
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
        train_split=predefined_split(train_val_ds),
        iterator_train__batch_size=BATCH_SIZE, # set batch sizes
        iterator_valid__batch_size=BATCH_SIZE, # set batch sizes
        device=device, # run on gpu,
        classes=list(range(NUM_CLASSES)), # tell nn what classes there are
    )

    return model

# function to train/fit the model
def train_model(model, train_ds):
   
    # LR scheduler (reduce LR if val acc doesnt increase)
    lr_scheduler = LRScheduler(
            policy='ReduceLROnPlateau',
            monitor='valid_acc',  # monitor validation accuracy
            factor=0.5, # reduce LR by this factor
            patience=3,# wait this many epochs before reducing
            threshold=1e-6, # min change to qualify as improvement
            cooldown=2, # wait time before resuming normal operation
            min_lr=1e-7, # don't go lower than this
        )
    # quit early if no val_acc improvement
    early_stop = EarlyStopping(monitor='valid_acc', patience=5)  # stops if no improvement in 3 epochs
    # log training accuracy
    train_acc_log = EpochScoring(scoring='accuracy', lower_is_better=False, name='train_acc', on_train=True)

    cleanup_epoch = EpochCleanupCallback()

    # fit the model
    model.fit(
        train_ds,  # training data set
        y=None, # dont provide labels separately
        callbacks=[lr_scheduler, early_stop, train_acc_log, cleanup_epoch],  # callbacks for early stopping, LR reduction, logging train acc and custom resource optimization
        verbose=1,  # log training data
    )

    return model