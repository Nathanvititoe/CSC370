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
    # if invalid model version, throw err
    if MODEL_VERSION not in MODEL_MAP:
        raise ValueError(f"Unknown MODEL_VERSION '{MODEL_VERSION}'")

    # get model function and weights
    model_fn, weights, kwargs = MODEL_MAP[MODEL_VERSION]
    model = model_fn(weights=weights, **kwargs)

    # freeze the model
    for param in model.parameters():
        param.requires_grad = False

    # Extract output features before removing top layer
    if hasattr(model, 'classifier'):
        in_features = get_in_features(model.classifier)
        model.classifier = nn.Identity()
    elif hasattr(model, 'fc'):
        in_features = get_in_features(model.fc)
        model.fc = nn.Identity()
    elif hasattr(model, 'heads'):
        in_features = get_in_features(model.heads)
        model.heads = nn.Identity()
    else:
        raise AttributeError(f"Cannot strip top layer: unknown model structure for '{MODEL_VERSION}'")
        
    return model, in_features

# get the output features from the final layer of pretrained model
def get_in_features(layer):
    for m in layer.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    raise ValueError("No Linear layer found in classification head.")