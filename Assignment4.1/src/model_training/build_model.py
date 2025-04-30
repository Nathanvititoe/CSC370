import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping, LRScheduler, EpochScoring
from torch.optim.lr_scheduler import OneCycleLR
from src.model_training.cleanup import EpochCleanupCallback
from src.model_training.get_model_version import get_model

# create the model (base: ConvNeXt_Tiny, classification layer: custom)
class Model(nn.Module):
    # class constructor
    def __init__(self, num_classes):
        super(Model, self).__init__()
        # load pretrained model as base -- cn_t, cn_s, cn_b (tiny, small, base) 
        # final choice: ConvNeXt_Tiny, smallest, fastest but similar accuracy to larger, slower models
        self.model, in_features = get_model("cn_t") # get the pretrained model 

        # only unfreeze very top layers of pretrained model
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in ["stages.3", "stages.2", "head"]):  
                param.requires_grad = True
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # use gap to reduce dimensions
        self.flatten = nn.Flatten() # flatten pool to 1D Vector

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128), # convert flatten into 128 hidden layer
            nn.Dropout(0.1), # randomly drop 10% of neurons
            nn.BatchNorm1d(128), # normalize
            nn.ReLU(), # apply relu activation 
            nn.Linear(128, num_classes), # get output layer as num_classes features
        )

    # method to pass data through model
    def forward(self, x):
        x = self.model(x) # get features from pretrained model
        x = self.pool(x)  # use gap to reduce dimensions     
        x = self.flatten(x) # flatten to 1D vector
        x = self.classifier(x) # use custom classifier layer
        return x

# build an instance of the model
def build_model(DEVICE, dataset, NUM_CLASSES, NUM_EPOCHS, BATCH_SIZE):

    # steps = dataset_length / batch_size
    steps_per_epoch = len(dataset) // BATCH_SIZE

    # wrap the model in skorch classifier for abstraction/ease of use
    classifier = NeuralNetClassifier(
        module=Model,
        module__num_classes=NUM_CLASSES, # number of classifiable classes 
        max_epochs=NUM_EPOCHS, # number of epochs
        optimizer=optim.Adamax, # adamax optimizer, dynamic LR updates
        optimizer__weight_decay=1e-4,
        criterion=nn.CrossEntropyLoss(label_smoothing=0.1), # use cross entropy for categorical classification labels
        train_split=ValidSplit(0.5, stratified=True, random_state=42),
        iterator_train__batch_size=BATCH_SIZE, # set batch sizes
        iterator_valid__batch_size=BATCH_SIZE, # set batch sizes
        device=DEVICE, # force gpu usage,
        classes=list(range(NUM_CLASSES)), # convert classes to ints
        verbose=2, # output logs
        callbacks=[
        # log training accuracy
        EpochScoring(scoring='accuracy', lower_is_better=False, name='train_acc', on_train=True), 

        # optimizes LR 
        LRScheduler(
        policy=OneCycleLR, # use OneCycleLr
        max_lr=0.01, # max lr when increasing
        steps_per_epoch=steps_per_epoch, # steps to take each epoch
        epochs=NUM_EPOCHS, # num of epochs
        pct_start=0.3,  # % of steps to spend increasing LR
        anneal_strategy='cos',  # cosine annealing
        step_every='batch', # start stepping every batch
        ),

        # stops if no improvement in 5 epochs
        EarlyStopping(monitor='valid_acc', patience=4, threshold=1e-5, lower_is_better=False, load_best=True),

        # cleanup for resource optimization
        EpochCleanupCallback()
        ], 
    )

    return classifier

# function to train/fit the model
def train_model(classifier, dataset):
    y_labels = torch.tensor(dataset.labels) # get labels

    # train model, pass ds and labels
    classifier.fit(dataset, y=y_labels)

    return classifier


