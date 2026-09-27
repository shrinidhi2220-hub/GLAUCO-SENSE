from pathlib import Path

import pandas as pd

from ml.error_analysis import save_error_analysis
from ml.evaluation_runner import run_synthetic_evaluation


def test_complete_phase3_pipeline(tmp_path: Path):
    output_dir = tmp_path / "evaluation"

    # Run the complete synthetic evaluation runner.
    metrics = run_synthetic_evaluation(
        output_dir=output_dir,
    )

    # Verify the core evaluation result.
    assert metrics["test_samples"] == 8
    assert metrics["incorrect_samples"] == 2

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["sensitivity"] <= 1.0
    assert 0.0 <= metrics["specificity"] <= 1.0

    # Verify the evaluation runner created its outputs.
    required_evaluation_files = [
        "metrics.json",
        "classification_report.txt",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "all_predictions.csv",
        "incorrect_predictions.csv",
    ]

    for filename in required_evaluation_files:
        assert (output_dir / filename).exists()

    # Load the prediction table produced by the runner.
    prediction_table = pd.read_csv(
        output_dir / "all_predictions.csv"
    )

    assert len(prediction_table) == 8

    assert {
        "path",
        "true_label",
        "predicted_label",
        "confidence",
        "correct",
    }.issubset(
        prediction_table.columns
    )

    # Run the standalone error-analysis layer
    # on the evaluation output.
    error_dir = tmp_path / "error_analysis"

    summary = save_error_analysis(
        prediction_table,
        error_dir,
    )

    assert summary["total_samples"] == 8
    assert summary["incorrect_samples"] == 2

    required_error_files = [
        "incorrect_predictions.csv",
        "errors_low_confidence_first.csv",
        "errors_high_confidence_first.csv",
        "error_pairs.csv",
        "error_summary.json",
    ]

    for filename in required_error_files:
        assert (error_dir / filename).exists()