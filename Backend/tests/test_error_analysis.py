from pathlib import Path

import pandas as pd
import pytest

from ml.error_analysis import (
    get_incorrect_predictions,
    save_error_analysis,
    sort_errors_by_confidence,
    summarize_error_pairs,
    summarize_errors,
)


def make_prediction_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "path": "a.jpg",
                "true_label": "normal",
                "predicted_label": "normal",
                "confidence": 0.95,
                "correct": True,
            },
            {
                "path": "b.jpg",
                "true_label": "normal",
                "predicted_label": "glaucoma",
                "confidence": 0.62,
                "correct": False,
            },
            {
                "path": "c.jpg",
                "true_label": "glaucoma",
                "predicted_label": "glaucoma",
                "confidence": 0.91,
                "correct": True,
            },
            {
                "path": "d.jpg",
                "true_label": "glaucoma",
                "predicted_label": "normal",
                "confidence": 0.83,
                "correct": False,
            },
            {
                "path": "e.jpg",
                "true_label": "glaucoma",
                "predicted_label": "normal",
                "confidence": 0.55,
                "correct": False,
            },
        ]
    )


def test_get_incorrect_predictions():
    table = make_prediction_table()

    incorrect = get_incorrect_predictions(
        table
    )

    assert len(incorrect) == 3

    assert list(
        incorrect["path"]
    ) == [
        "b.jpg",
        "d.jpg",
        "e.jpg",
    ]


def test_sort_errors_by_confidence():
    table = make_prediction_table()

    sorted_errors = sort_errors_by_confidence(
        table,
        ascending=True,
    )

    assert list(
        sorted_errors["path"]
    ) == [
        "e.jpg",
        "b.jpg",
        "d.jpg",
    ]

    assert (
        sorted_errors.iloc[0]["confidence"]
        == 0.55
    )


def test_summarize_errors():
    table = make_prediction_table()

    summary = summarize_errors(table)

    assert summary["total_samples"] == 5
    assert summary["incorrect_samples"] == 3
    assert summary["error_rate"] == 0.6
    assert (
        summary["lowest_confidence_error"]
        == 0.55
    )
    assert (
        summary["highest_confidence_error"]
        == 0.83
    )


def test_summarize_error_pairs():
    table = make_prediction_table()

    summary = summarize_error_pairs(
        table
    )

    assert len(summary) == 2

    normal_to_glaucoma = summary[
        (
            summary["true_label"]
            == "normal"
        )
        & (
            summary["predicted_label"]
            == "glaucoma"
        )
    ]

    glaucoma_to_normal = summary[
        (
            summary["true_label"]
            == "glaucoma"
        )
        & (
            summary["predicted_label"]
            == "normal"
        )
    ]

    assert (
        normal_to_glaucoma.iloc[0]["count"]
        == 1
    )

    assert (
        glaucoma_to_normal.iloc[0]["count"]
        == 2
    )


def test_save_error_analysis(
    tmp_path: Path,
):
    table = make_prediction_table()

    output_dir = (
        tmp_path / "error_analysis"
    )

    summary = save_error_analysis(
        table,
        output_dir,
    )

    assert summary["incorrect_samples"] == 3

    expected_files = [
        "incorrect_predictions.csv",
        "errors_low_confidence_first.csv",
        "errors_high_confidence_first.csv",
        "error_pairs.csv",
        "error_summary.json",
    ]

    for filename in expected_files:
        assert (
            output_dir / filename
        ).exists()


def test_invalid_confidence():
    table = make_prediction_table()

    table.loc[
        0,
        "confidence",
    ] = 1.5

    with pytest.raises(
        ValueError,
        match="Confidence values",
    ):
        get_incorrect_predictions(table)