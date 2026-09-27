from pathlib import Path

from ml.evaluation_runner import (
    build_synthetic_predictions,
    run_synthetic_evaluation,
)


def test_synthetic_predictions_are_valid():
    paths, y_true, y_pred, confidence = (
        build_synthetic_predictions()
    )

    assert len(paths) == 8
    assert len(y_true) == 8
    assert len(y_pred) == 8
    assert len(confidence) == 8

    assert all(
        0 <= value < 2
        for value in y_true
    )

    assert all(
        0 <= value < 2
        for value in y_pred
    )

    assert all(
        0.0 <= value <= 1.0
        for value in confidence
    )


def test_synthetic_evaluation_runner(tmp_path: Path):
    output_dir = (
        tmp_path / "synthetic_evaluation"
    )

    metrics = run_synthetic_evaluation(
        output_dir=output_dir,
    )

    assert metrics["test_samples"] == 8
    assert metrics["incorrect_samples"] == 2

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["sensitivity"] <= 1.0
    assert 0.0 <= metrics["specificity"] <= 1.0

    expected_files = [
        "metrics.json",
        "classification_report.txt",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "all_predictions.csv",
        "incorrect_predictions.csv",
    ]

    for filename in expected_files:
        assert (
            output_dir / filename
        ).exists()