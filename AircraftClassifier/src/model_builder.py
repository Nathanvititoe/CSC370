# pytorch
import torch   # type: ignore

# pytorch neural network methods
import torch.nn as nn  # type: ignore

# pretrained models import
import torchvision.models as models  # type: ignore
dropout = 0.349
lr = 0.000506
dense_units = 255
weight_decay = 2.3e-5

# TODO: 
# test different pretrained models
# see how pretrain classifier compares to custom
# test w full dataset using curr best params, and wider net params
# find model to test w lower res images EfficientNetB0 or MobileNetV2/V3
# LR reducer? LR Scheduler?

# Function to build pretrained base, custom head
def build_model(num_classes, dropout_rate=0.35, dense_units=256):
    # Load EfficientNetB0 trained on imageNet
        # EfficientNetB3, best valAcc was 90% on small_subset
        # EfficientNetB0, best valAcc was 92% on small_subset


    # - EfficientNet will identify which features are important
    base_model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    
    pretrained_feature_output = base_model.classifier[1].in_features # get num of features from base_model
    
    # Strip off the original top layer to add a custom one
    base_model.classifier = nn.Identity()

    # # create a custom top layer classifier
    # #   - custom classifier determines which features mean what
    # #   - highest val_acc of 92% on small_subset
    # classifier = nn.Sequential(
    #     nn.BatchNorm1d(pretrained_feature_output), # normalize
    #     nn.Linear(pretrained_feature_output, dense_units), # 
    #     nn.ReLU(), # apply activation function for non-linearity
    #     nn.Dropout(dropout_rate), # drop a % of neurons every pass
    #     nn.Linear(dense_units, num_classes) # converts raw output to output classes 
    # )

    # Test for full dataset
    classifier = nn.Sequential(
        # conv layer 1
        nn.BatchNorm1d(pretrained_feature_output),
        nn.Linear(pretrained_feature_output, dense_units),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        
        # conv layer 2
        nn.BatchNorm1d(dense_units),
        nn.Linear(dense_units, dense_units // 2),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate / 2),
        
        # output layer
        nn.Linear(dense_units // 2, num_classes)
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