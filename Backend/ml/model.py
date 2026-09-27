from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_baseline_model(
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    """Build the GLAUCO-SENSE ResNet18 baseline classifier."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)
    return model
