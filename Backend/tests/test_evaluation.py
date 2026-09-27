from pathlib import Path

import numpy as np

from ml.evaluation import (
    build_prediction_table,
    evaluate_predictions,
    save_confusion_matrix,
    save_evaluation_reports,
)


def test_evaluate_predictions():
    classes = ["normal", "glaucoma"]

    y_true = [0, 0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 1, 0, 0]

    result = evaluate_predictions(
        y_true,
        y_pred,
        classes,
    )

    assert result["samples"] == 6
    assert result["incorrect_samples"] == 2

    assert result["metrics"]["accuracy"] == 4 / 6

    expected_cm = np.array(
        [
            [2, 1],
            [1, 2],
        ]
    )

    assert np.array_equal(
        result["confusion_matrix"],
        expected_cm,
    )


def test_prediction_table():
    table = build_prediction_table(
        paths=[
            "a.jpg",
            "b.jpg",
            "c.jpg",
        ],
        y_true=[0, 1, 1],
        y_pred=[0, 0, 1],
        confidence=[
            0.95,
            0.70,
            0.88,
        ],
        classes=[
            "normal",
            "glaucoma",
        ],
    )

    assert len(table) == 3

    assert list(table.columns) == [
        "path",
        "true_label",
        "predicted_label",
        "confidence",
        "correct",
    ]

    assert bool(table.loc[0, "correct"]) is True
    assert bool(table.loc[1, "correct"]) is False
    assert bool(table.loc[2, "correct"]) is True


def test_save_confusion_matrix(tmp_path: Path):
    cm = np.array(
        [
            [3, 1],
            [2, 4],
        ]
    )

    output = tmp_path / "confusion_matrix.png"

    save_confusion_matrix(
        cm,
        ["normal", "glaucoma"],
        output,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_save_evaluation_reports(tmp_path: Path):
    classes = ["normal", "glaucoma"]

    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]

    table = build_prediction_table(
        paths=[
            "one.jpg",
            "two.jpg",
            "three.jpg",
            "four.jpg",
        ],
        y_true=y_true,
        y_pred=y_pred,
        confidence=[
            0.95,
            0.60,
            0.91,
            0.55,
        ],
        classes=classes,
    )

    output_dir = tmp_path / "evaluation"

    metrics = save_evaluation_reports(
        y_true=y_true,
        y_pred=y_pred,
        classes=classes,
        output_dir=output_dir,
        prediction_table=table,
    )

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "sensitivity" in metrics
    assert "specificity" in metrics

    expected_files = [
        "metrics.json",
        "classification_report.txt",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "all_predictions.csv",
        "incorrect_predictions.csv",
    ]

    for filename in expected_files:
        assert (output_dir / filename).exists()