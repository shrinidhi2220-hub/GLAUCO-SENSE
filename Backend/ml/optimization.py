from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def build_classification_loss(class_weights: Sequence[float] | torch.Tensor | None = None) -> nn.CrossEntropyLoss:
    """Build classification loss with optional positive class weights."""
    if class_weights is None:
        return nn.CrossEntropyLoss()
    if isinstance(class_weights, torch.Tensor):
        weights = class_weights.detach().clone().float()
    else:
        weights = torch.tensor(list(class_weights), dtype=torch.float32)
    if weights.ndim != 1:
        raise ValueError("class_weights must be a 1D sequence or tensor.")
    if len(weights) < 2:
        raise ValueError("class_weights must contain at least 2 class weights.")
    if not torch.isfinite(weights).all():
        raise ValueError("class_weights must contain only finite values.")
    if torch.any(weights <= 0):
        raise ValueError("class_weights must contain positive values.")
    return nn.CrossEntropyLoss(weight=weights)


def build_optimizer(model: nn.Module, learning_rate: float = 1e-4,
                     weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """Build the baseline AdamW optimizer."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero.")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
