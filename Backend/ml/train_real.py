from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from ml.config import (
    LEARNING_RATE,
    SEED,
    WEIGHT_DECAY,
)
from ml.dataset import class_counts, load_manifest
from ml.model import build_baseline_model
from ml.optimization import (
    build_classification_loss,
    build_inverse_frequency_weights,
    build_optimizer,
)
from ml.real_data import build_real_dataloaders
from ml.training import fit


def set_seed(seed: int = SEED) -> None:
    """Set deterministic random seeds where supported."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_real_training(
    epochs: int,
    checkpoint_path: str | Path,
    pretrained: bool = False,
) -> list[dict]:
    """Train ResNet18 on the real HYGD dataset."""

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    (
        train_loader,
        val_loader,
        _test_loader,
        class_to_id,
    ) = build_real_dataloaders()

    train_manifest = load_manifest(
        "data/splits/train.csv"
    )

    counts = class_counts(
        train_manifest,
        class_to_id,
    )

    class_weights = build_inverse_frequency_weights(
        counts
    ).to(device)

    model = build_baseline_model(
        num_classes=len(class_to_id),
        pretrained=pretrained,
    )

    criterion = build_classification_loss(
        class_weights=class_weights
    )

    optimizer = build_optimizer(
        model=model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        checkpoint_path=checkpoint_path,
    )

    print()
    print("Real HYGD training completed.")
    print(f"Device: {device}")
    print(f"Classes: {class_to_id}")
    print(f"Class counts: {counts.tolist()}")
    print(
        "Class weights:",
        [round(value, 4) for value in class_weights.tolist()],
    )
    print(f"Epochs: {epochs}")
    print(f"Checkpoint: {Path(checkpoint_path).resolve()}")

    return history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GLAUCO-SENSE ResNet18 on real HYGD data."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "models/checkpoints/hygd_real_smoke.pt"
        ),
        help="Output checkpoint path.",
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained ImageNet ResNet18 weights.",
    )

    args = parser.parse_args()

    history = run_real_training(
        epochs=args.epochs,
        checkpoint_path=args.checkpoint,
        pretrained=args.pretrained,
    )

    final = history[-1]

    print()
    print(
        f"Final train loss: "
        f"{final['train_loss']:.4f}"
    )
    print(
        f"Final train accuracy: "
        f"{final['train_accuracy']:.4f}"
    )
    print(
        f"Final validation loss: "
        f"{final['val_loss']:.4f}"
    )
    print(
        f"Final validation accuracy: "
        f"{final['val_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()