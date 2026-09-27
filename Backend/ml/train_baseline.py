from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.config import BATCH_SIZE, IMAGE_SIZE, LEARNING_RATE, SEED
from ml.model import build_baseline_model
from ml.training import fit


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_synthetic_loaders(train_samples: int = 16, val_samples: int = 8,
                            batch_size: int = BATCH_SIZE) -> tuple[DataLoader, DataLoader]:
    """Build deterministic synthetic image data for dependency-free pipeline testing."""
    generator = torch.Generator().manual_seed(SEED)
    train_images = torch.randn(train_samples, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    train_labels = torch.randint(0, 2, (train_samples,), generator=generator)
    val_images = torch.randn(val_samples, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    val_labels = torch.randint(0, 2, (val_samples,), generator=generator)
    return (
        DataLoader(TensorDataset(train_images, train_labels), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(val_images, val_labels), batch_size=batch_size, shuffle=False),
    )


def run_smoke_training(epochs: int, checkpoint_path: str | Path) -> list[dict]:
    """Run an end-to-end synthetic baseline training smoke test."""
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_synthetic_loaders()
    model = build_baseline_model(num_classes=2, pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    return fit(model=model, train_loader=train_loader, val_loader=val_loader,
               criterion=criterion, optimizer=optimizer, device=device,
               epochs=epochs, checkpoint_path=checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="GLAUCO-SENSE independent baseline smoke test using synthetic data.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of synthetic training epochs.")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("models/checkpoints/baseline_smoke_test.pt"),
                        help="Output checkpoint path.")
    args = parser.parse_args()
    history = run_smoke_training(epochs=args.epochs, checkpoint_path=args.checkpoint)
    final = history[-1]
    print(f"Completed {len(history)} epoch(s) on synthetic data.")
    print(f"Final train loss: {final['train_loss']:.4f}")
    print(f"Final validation loss: {final['val_loss']:.4f}")
    print(f"Checkpoint saved to: {args.checkpoint}")


if __name__ == "__main__":
    main()
