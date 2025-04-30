from torchvision import models
import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
)

# define models for each keyword
MODEL_MAP = {
    # ConvNeXt models
    "cn_t": (models.convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1, {}), # fast, light
    "cn_s": (models.convnext_small, ConvNeXt_Small_Weights.IMAGENET1K_V1, {}), # middle
    "cn_b": (models.convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1, {}), # heavy
    "cn_l": (models.convnext_large, ConvNeXt_Large_Weights.IMAGENET1K_V1, {}), # very heavy
}

# function to toggle pretrained model base for testing
def get_model(MODEL_VERSION):
    if MODEL_VERSION not in MODEL_MAP:
        raise ValueError(f"Unknown MODEL_VERSION '{MODEL_VERSION}'")

    model_fn, weights = MODEL_MAP[MODEL_VERSION]
    model = model_fn(weights=weights)

    # freeze the pretrained model (dont let it train)
    for param in model.parameters():
        param.requires_grad = False

    # get number of output features
    in_features = get_in_features(model.classifier)

    # remove the output/classifier layer from pretrained model
    model.classifier = nn.Identity()

    return model, in_features

# get the output features from the final layer of pretrained model
def get_in_features(layer):
    for module in layer.modules():
        # if theres a linear layer
        if isinstance(module, nn.Linear):
            return module.in_features # return the features