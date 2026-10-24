
import torch.nn as nn
from torchvision import models


def create_model(num_classes: int = 41, pretrained: bool = True):
    """
    Create a MobileNetV3-Small classifier
    for cattle and buffalo breed recognition.
    """

    weights = (
        models.MobileNet_V3_Small_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.mobilenet_v3_small(weights=weights)

    # Replace the final classification layer
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model
