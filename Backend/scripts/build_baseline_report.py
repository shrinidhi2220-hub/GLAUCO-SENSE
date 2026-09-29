from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


BASE_DIR = Path("reports/evaluation/hygd_baseline")
ERROR_DIR = BASE_DIR / "error_analysis"

CHECKPOINT = Path(
    "models/checkpoints/hygd_resnet18_baseline.pt"
)

OUTPUT = BASE_DIR / "BASELINE_REPORT.md"


def sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required report not found: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_error_summary(path: Path) -> dict:
    """Load error summary, handling the records-oriented JSON format."""

    payload = load_json(path)

    if isinstance(payload, dict):
        return payload

    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError(
                f"Expected one error-summary record, found {len(payload)}."
            )

        if not isinstance(payload[0], dict):
            raise ValueError(
                "Error summary record must be a JSON object."
            )

        return payload[0]

    raise ValueError(
        f"Unexpected error summary format: {type(payload).__name__}"
    )


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Required report not found: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    ).strip()


def load_checkpoint_metadata(
    checkpoint_path: Path,
) -> dict:
    """Load training metadata from the frozen checkpoint."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    return checkpoint


def main() -> None:
    metrics = load_json(
        BASE_DIR / "metrics.json"
    )

    error_summary = load_error_summary(
        ERROR_DIR / "error_summary.json"
    )

    error_pairs = read_text(
        ERROR_DIR / "error_pairs.csv"
    )

    confusion_matrix = read_text(
        BASE_DIR / "confusion_matrix.csv"
    )

    classification_report = read_text(
        BASE_DIR / "classification_report.txt"
    )

    checkpoint = load_checkpoint_metadata(
        CHECKPOINT
    )

    checkpoint_size_mb = (
        CHECKPOINT.stat().st_size
        / (1024 * 1024)
    )

    checkpoint_hash = sha256_file(
        CHECKPOINT
    )

    train_metrics = checkpoint.get(
        "train_metrics",
        {},
    )

    val_metrics = checkpoint.get(
        "val_metrics",
        {},
    )

    checkpoint_epoch = checkpoint.get(
        "epoch",
        "N/A",
    )

    lines = [
        "# GLAUCO-SENSE — HYGD ResNet18 Baseline Report",
        "",
        "## Experiment Status",
        "",
        "This report freezes the first real HYGD baseline experiment.",
        "",
        "The reported metrics are research/engineering results on the "
        "prepared HYGD dataset and should not be interpreted as clinical "
        "validation or diagnostic performance.",
        "",
        "## Dataset",
        "",
        "| Property | Value |",
        "|---|---:|",
        "| Dataset | Hillel-Yaffe Glaucoma Dataset (HYGD) |",
        "| Cleaned images | 737 |",
        "| Represented patients | 286 |",
        "| Train images | 499 |",
        "| Validation images | 123 |",
        "| Test images | 115 |",
        "| Train patients | 200 |",
        "| Validation patients | 43 |",
        "| Test patients | 43 |",
        "| Patient overlap | 0 |",
        "",
        "## Classes",
        "",
        "| Class | Dataset role | ID |",
        "|---|---|---:|",
        "| GON+ | Positive class | 0 |",
        "| GON- | Negative class | 1 |",
        "",
        "The evaluation metric layer treats **GON+** as the positive "
        "class for sensitivity/specificity calculations.",
        "",
        "## Preprocessing",
        "",
        "- EXIF orientation correction",
        "- RGB conversion",
        "- Resize to 224 × 224",
        "- JPEG output",
        "- ImageNet normalization is applied in the ML transform pipeline, "
        "not permanently written into the stored JPEG files.",
        "",
        "## Training Configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
        "| Architecture | ResNet18 |",
        "| Initialization | ImageNet pretrained weights |",
        "| Epochs requested | 10 |",
        f"| Best checkpoint epoch | {checkpoint_epoch} |",
        "| Batch size | 16 |",
        "| Learning rate | 1e-4 |",
        "| Weight decay | 1e-4 |",
        "| Optimizer | AdamW |",
        "| Training device | CPU |",
        "| Random seed | 42 |",
        "| Loss | Weighted CrossEntropyLoss |",
        "",
        "### Training class weights",
        "",
        "| Class | Training count | Weight |",
        "|---|---:|---:|",
        "| GON+ | 367 | 0.6798 |",
        "| GON- | 132 | 1.8902 |",
        "",
        "## Best Training/Validation Checkpoint Metrics",
        "",
        "| Metric | Value |",
        f"| Train loss | {train_metrics.get('loss', 'N/A')} |",
        f"| Train accuracy | {train_metrics.get('accuracy', 'N/A')} |",
        f"| Validation loss | {val_metrics.get('loss', 'N/A')} |",
        f"| Validation accuracy | {val_metrics.get('accuracy', 'N/A')} |",
        "",
        "## Held-Out Test Results",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Test samples | {metrics['test_samples']} |",
        f"| Incorrect samples | {metrics['incorrect_samples']} |",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall | {metrics['recall']:.4f} |",
        f"| F1 | {metrics['f1']:.4f} |",
        f"| Sensitivity | {metrics['sensitivity']:.4f} |",
        f"| Specificity | {metrics['specificity']:.4f} |",
        f"| PPV | {metrics['ppv']:.4f} |",
        f"| NPV | {metrics['npv']:.4f} |",
        "",
        "## Confusion Matrix",
        "",
        "```text",
        confusion_matrix,
        "```",
        "",
        "## Classification Report",
        "",
        "```text",
        classification_report,
        "```",
        "",
        "## Error Analysis",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total test samples | {error_summary['total_samples']} |",
        f"| Incorrect samples | {error_summary['incorrect_samples']} |",
        f"| Error rate | {error_summary['error_rate']:.6f} |",
        f"| Mean error confidence | {error_summary['mean_error_confidence']:.6f} |",
        f"| Lowest error confidence | {error_summary['lowest_confidence_error']:.6f} |",
        f"| Highest error confidence | {error_summary['highest_confidence_error']:.6f} |",
        "",
        "### Error Directions",
        "",
        "```text",
        error_pairs,
        "```",
        "",
        "### Incorrect Test Samples",
        "",
        "| Image | True | Predicted | Confidence |",
        "|---|---|---|---:|",
        "| 166_1.jpg | GON+ | GON- | 0.5192 |",
        "| 275_1.jpg | GON- | GON+ | 0.6085 |",
        "| 132_0.jpg | GON+ | GON- | 0.6765 |",
        "| 249_3.jpg | GON- | GON+ | 0.7983 |",
        "",
        "## Checkpoint Integrity",
        "",
        f"- Checkpoint: `{CHECKPOINT}`",
        f"- Size: `{checkpoint_size_mb:.2f} MB`",
        f"- SHA-256: `{checkpoint_hash}`",
        "",
        "## Evaluation Artifacts",
        "",
        "```text",
        "reports/evaluation/hygd_baseline/",
        "├── metrics.json",
        "├── classification_report.txt",
        "├── confusion_matrix.csv",
        "├── confusion_matrix.png",
        "├── all_predictions.csv",
        "├── incorrect_predictions.csv",
        "├── BASELINE_REPORT.md",
        "└── error_analysis/",
        "    ├── incorrect_predictions.csv",
        "    ├── errors_low_confidence_first.csv",
        "    ├── errors_high_confidence_first.csv",
        "    ├── error_pairs.csv",
        "    └── error_summary.json",
        "```",
        "",
        "## Reproducibility Note",
        "",
        "The held-out test set was evaluated after model selection. "
        "The test set was not used for training or checkpoint selection.",
        "",
        "The frozen checkpoint corresponds to the epoch at which "
        "validation loss was best during the 10-epoch training run.",
        "",
    ]

    OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("Baseline report generated.")
    print(f"Report: {OUTPUT.resolve()}")
    print(f"Checkpoint epoch: {checkpoint_epoch}")
    print(
        f"Checkpoint SHA-256: {checkpoint_hash}"
    )


if __name__ == "__main__":
    main()