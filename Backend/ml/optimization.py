from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def build_classification_loss(
    class_weights: Sequence[float] | torch.Tensor | None = None,
) -> nn.CrossEntropyLoss:
    """Build classification loss with optional class weights."""

    if class_weights is None:
        return nn.CrossEntropyLoss()

    if isinstance(class_weights, torch.Tensor):
        weights = class_weights.detach().clone().float()
    else:
        weights = torch.tensor(
            list(class_weights),
            dtype=torch.float32,
        )

    if weights.ndim != 1:
        raise ValueError(
            "class_weights must be a 1D sequence or tensor."
        )

    if len(weights) < 2:
        raise ValueError(
            "class_weights must contain at least 2 class weights."
        )

    if not torch.isfinite(weights).all():
        raise ValueError(
            "class_weights must contain only finite values."
        )

    if torch.any(weights <= 0):
        raise ValueError(
            "class_weights must contain positive values."
        )

    return nn.CrossEntropyLoss(weight=weights)


def build_inverse_frequency_weights(
    class_counts: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Build normalized inverse-frequency class weights.

    Weight for class i:

        total_samples / (num_classes * class_count_i)

    The resulting weights have mean approximately 1.
    """

    if isinstance(class_counts, torch.Tensor):
        counts = class_counts.detach().clone().float()
    else:
        counts = torch.tensor(
            list(class_counts),
            dtype=torch.float32,
        )

    if counts.ndim != 1:
        raise ValueError(
            "class_counts must be a 1D sequence or tensor."
        )

    if len(counts) < 2:
        raise ValueError(
            "At least two class counts are required."
        )

    if not torch.isfinite(counts).all():
        raise ValueError(
            "class_counts must contain only finite values."
        )

    if torch.any(counts <= 0):
        raise ValueError(
            "Every class count must be greater than zero."
        )

    total = counts.sum()
    num_classes = counts.numel()

    weights = total / (num_classes * counts)

    return weights


def build_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Build the baseline AdamW optimizer."""

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than zero."
        )

    if weight_decay < 0:
        raise ValueError(
            "weight_decay cannot be negative."
        )

    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )