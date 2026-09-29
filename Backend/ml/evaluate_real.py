from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.dataset import build_class_mapping, load_manifest
from ml.evaluation import build_prediction_table, save_evaluation_reports
from ml.model import build_baseline_model
from ml.real_data import build_real_dataloaders


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict:
    """Load a trained checkpoint into the model."""

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain model_state_dict: "
            f"{checkpoint_path}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return checkpoint


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[float]]:
    """Collect true labels, predicted labels and confidences."""

    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    confidence: list[float] = []

    for images, labels in loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        predicted = probabilities.argmax(
            dim=1
        )

        predicted_confidence = probabilities.max(
            dim=1
        ).values

        y_true.extend(
            labels.cpu().tolist()
        )

        y_pred.extend(
            predicted.cpu().tolist()
        )

        confidence.extend(
            predicted_confidence.cpu().tolist()
        )

    return (
        y_true,
        y_pred,
        confidence,
    )


def build_test_prediction_paths() -> list[str]:
    """Return test image paths in manifest order."""

    test_manifest = Path(
        "data/splits/test.csv"
    )

    df = load_manifest(test_manifest)

    if "processed_path" in df.columns:
        return (
            df["processed_path"]
            .astype(str)
            .tolist()
        )

    return (
        df["path"]
        .astype(str)
        .tolist()
    )


def run_real_evaluation(
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict:

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        _train_loader,
        _val_loader,
        test_loader,
        class_to_id,
    ) = build_real_dataloaders()

    train_manifest = load_manifest(
        "data/splits/train.csv"
    )

    class_to_id = build_class_mapping(
        train_manifest
    )

    classes = [
        label
        for label, _class_id in sorted(
            class_to_id.items(),
            key=lambda item: item[1],
        )
    ]

    model = build_baseline_model(
        num_classes=len(classes),
        pretrained=False,
    )

    checkpoint = load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    model.to(device)

    (
        y_true,
        y_pred,
        confidence,
    ) = collect_predictions(
        model=model,
        loader=test_loader,
        device=device,
    )

    paths = build_test_prediction_paths()

    if len(paths) != len(y_true):
        raise RuntimeError(
            "Prediction/path count mismatch: "
            f"{len(y_true)} predictions vs "
            f"{len(paths)} paths"
        )

    prediction_table = build_prediction_table(
        paths=paths,
        y_true=y_true,
        y_pred=y_pred,
        confidence=confidence,
        classes=classes,
    )

    metrics = save_evaluation_reports(
        y_true=y_true,
        y_pred=y_pred,
        classes=classes,
        output_dir=output_dir,
        prediction_table=prediction_table,
    )

    print()
    print("Real HYGD evaluation completed.")
    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Classes: {class_to_id}")
    print(f"Test samples: {metrics['test_samples']}")
    print(f"Incorrect samples: {metrics['incorrect_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(
        f"Sensitivity: "
        f"{metrics['sensitivity']:.4f}"
    )
    print(
        f"Specificity: "
        f"{metrics['specificity']:.4f}"
    )
    print(
        f"PPV: "
        f"{metrics['ppv']:.4f}"
    )
    print(
        f"NPV: "
        f"{metrics['npv']:.4f}"
    )
    print(
        f"Reports saved to: "
        f"{Path(output_dir).resolve()}"
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GLAUCO-SENSE ResNet18 "
            "on the held-out HYGD test set."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "models/checkpoints/"
            "hygd_resnet18_baseline.pt"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/evaluation/hygd_baseline"
        ),
    )

    args = parser.parse_args()

    run_real_evaluation(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()