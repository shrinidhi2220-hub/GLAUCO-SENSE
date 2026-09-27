from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from ml.metrics import classification_metrics


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str],
) -> dict:
    """
    Evaluate predictions using the project's standard metrics.
    """

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Evaluation requires at least one prediction."
        )

    if len(classes) < 2:
        raise ValueError(
            "At least two classes are required."
        )

    labels = list(range(len(classes)))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    metrics = classification_metrics(
        y_true,
        y_pred,
        classes,
    )

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classes": list(classes),
        "samples": len(y_true),
        "incorrect_samples": int(
            sum(
                true != pred
                for true, pred in zip(y_true, y_pred)
            )
        ),
    }


def build_prediction_table(
    paths: Sequence[str],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    confidence: Sequence[float],
    classes: Sequence[str],
) -> pd.DataFrame:
    """
    Build a table containing every prediction.

    This table can later be used for incorrect-prediction analysis.
    """

    lengths = {
        len(paths),
        len(y_true),
        len(y_pred),
        len(confidence),
    }

    if len(lengths) != 1:
        raise ValueError(
            "paths, y_true, y_pred and confidence "
            "must have equal lengths."
        )

    if len(classes) < 2:
        raise ValueError(
            "At least two classes are required."
        )

    rows = []

    for path, true_id, pred_id, conf in zip(
        paths,
        y_true,
        y_pred,
        confidence,
    ):
        if not 0 <= int(true_id) < len(classes):
            raise ValueError(
                f"Invalid true label: {true_id}"
            )

        if not 0 <= int(pred_id) < len(classes):
            raise ValueError(
                f"Invalid predicted label: {pred_id}"
            )

        rows.append(
            {
                "path": str(path),
                "true_label": classes[int(true_id)],
                "predicted_label": classes[int(pred_id)],
                "confidence": float(conf),
                "correct": int(true_id) == int(pred_id),
            }
        )

    return pd.DataFrame(rows)


def save_confusion_matrix(
    cm: np.ndarray,
    classes: Sequence[str],
    output_path: str | Path,
) -> None:
    """
    Save a confusion-matrix visualization.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest",
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title="GLAUCO-SENSE Confusion Matrix",
    )

    threshold = (
        float(cm.max()) / 2.0
        if cm.size
        else 0.0
    )

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                int(cm[row, col]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[row, col] > threshold
                    else "black"
                ),
            )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)


def save_evaluation_reports(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str],
    output_dir: str | Path,
    prediction_table: pd.DataFrame | None = None,
) -> dict:
    """
    Save the complete evaluation package.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = evaluate_predictions(
        y_true,
        y_pred,
        classes,
    )

    metrics = result["metrics"]
    cm = result["confusion_matrix"]

    # Metrics JSON
    metrics_payload = {
        **metrics,
        "classes": list(classes),
        "test_samples": result["samples"],
        "incorrect_samples": result["incorrect_samples"],
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(
            metrics_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Classification report
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=list(classes),
        zero_division=0,
    )

    (output_dir / "classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    # Confusion matrix CSV
    pd.DataFrame(
        cm,
        index=classes,
        columns=classes,
    ).to_csv(
        output_dir / "confusion_matrix.csv"
    )

    # Confusion matrix image
    save_confusion_matrix(
        cm,
        classes,
        output_dir / "confusion_matrix.png",
    )

    # Prediction/error analysis
    if prediction_table is not None:
        prediction_table.to_csv(
            output_dir / "all_predictions.csv",
            index=False,
        )

        incorrect = prediction_table[
            ~prediction_table["correct"]
        ].copy()

        incorrect = incorrect.sort_values(
            "confidence",
        )

        incorrect.to_csv(
            output_dir / "incorrect_predictions.csv",
            index=False,
        )

    return metrics_payload