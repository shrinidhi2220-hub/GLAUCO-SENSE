from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size
    if total == 0:
        raise ValueError("Training loader is empty.")
    return {"loss": running_loss / total, "accuracy": correct / total}


@torch.no_grad()
def validate_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                       device: torch.device) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size
    if total == 0:
        raise ValueError("Validation loader is empty.")
    return {"loss": running_loss / total, "accuracy": correct / total}


def save_checkpoint(path: str | Path, model: nn.Module,
                    optimizer: torch.optim.Optimizer, epoch: int,
                    train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }, checkpoint_path)


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device,
        epochs: int, checkpoint_path: str | Path, scheduler=None,
        on_epoch_end: Callable[[int, dict], None] | None = None) -> list[dict]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    history: list[dict] = []
    best_val_loss = float("inf")
    model.to(device)
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(record)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(checkpoint_path, model, optimizer, epoch, train_metrics, val_metrics)
        if on_epoch_end is not None:
            on_epoch_end(epoch, record)
    return history
