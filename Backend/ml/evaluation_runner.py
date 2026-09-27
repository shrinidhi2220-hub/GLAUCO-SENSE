from __future__ import annotations

import argparse
from pathlib import Path

from ml.evaluation import (
    build_prediction_table,
    save_evaluation_reports,
)


CLASSES = ["normal", "glaucoma"]


def build_synthetic_predictions() -> tuple[
    list[str],
    list[int],
    list[int],
    list[float],
]:
    """
    Build deterministic synthetic predictions.

    This is only for development/testing and is NOT
    a glaucoma evaluation dataset.
    """

    paths = [
        "synthetic_001.jpg",
        "synthetic_002.jpg",
        "synthetic_003.jpg",
        "synthetic_004.jpg",
        "synthetic_005.jpg",
        "synthetic_006.jpg",
        "synthetic_007.jpg",
        "synthetic_008.jpg",
    ]

    y_true = [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    ]

    y_pred = [
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
    ]

    confidence = [
        0.96,
        0.91,
        0.72,
        0.88,
        0.95,
        0.90,
        0.67,
        0.93,
    ]

    return paths, y_true, y_pred, confidence


def run_synthetic_evaluation(
    output_dir: str | Path,
) -> dict:
    """
    Run the complete evaluation pipeline with synthetic data.
    """

    (
        paths,
        y_true,
        y_pred,
        confidence,
    ) = build_synthetic_predictions()

    prediction_table = build_prediction_table(
        paths=paths,
        y_true=y_true,
        y_pred=y_pred,
        confidence=confidence,
        classes=CLASSES,
    )

    metrics = save_evaluation_reports(
        y_true=y_true,
        y_pred=y_pred,
        classes=CLASSES,
        output_dir=output_dir,
        prediction_table=prediction_table,
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "GLAUCO-SENSE evaluation runner "
            "using synthetic predictions."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/evaluation/synthetic"
        ),
        help="Directory for evaluation reports.",
    )

    args = parser.parse_args()

    metrics = run_synthetic_evaluation(
        output_dir=args.output,
    )

    print("Synthetic evaluation completed.")

    print(
        f"Samples: {metrics['test_samples']}"
    )

    print(
        f"Accuracy: {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall: {metrics['recall']:.4f}"
    )

    print(
        f"F1: {metrics['f1']:.4f}"
    )

    print(
        f"Sensitivity: "
        f"{metrics['sensitivity']:.4f}"
    )

    print(
        f"Specificity: "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"Reports saved to: {args.output}"
    )


if __name__ == "__main__":
    main()