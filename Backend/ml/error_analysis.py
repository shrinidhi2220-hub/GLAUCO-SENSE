from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "path",
    "true_label",
    "predicted_label",
    "confidence",
    "correct",
}


def validate_prediction_table(
    table: pd.DataFrame,
) -> None:
    """Validate the schema required for error analysis."""

    missing = REQUIRED_COLUMNS - set(table.columns)

    if missing:
        raise ValueError(
            f"Prediction table is missing columns: "
            f"{sorted(missing)}"
        )

    if table.empty:
        raise ValueError(
            "Prediction table is empty."
        )

    if not table["confidence"].between(0.0, 1.0).all():
        raise ValueError(
            "Confidence values must be between 0 and 1."
        )


def get_incorrect_predictions(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Return only incorrectly classified samples."""

    validate_prediction_table(table)

    incorrect = table[
        ~table["correct"].astype(bool)
    ].copy()

    return incorrect.reset_index(drop=True)


def sort_errors_by_confidence(
    table: pd.DataFrame,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Sort incorrect predictions by confidence.

    ascending=True:
        lowest-confidence errors first.

    ascending=False:
        highest-confidence errors first.
    """

    incorrect = get_incorrect_predictions(table)

    return incorrect.sort_values(
        "confidence",
        ascending=ascending,
    ).reset_index(drop=True)


def summarize_errors(
    table: pd.DataFrame,
) -> dict:
    """Return a compact summary of model errors."""

    incorrect = get_incorrect_predictions(table)

    total = len(table)
    errors = len(incorrect)

    if total == 0:
        raise ValueError(
            "Cannot summarize an empty table."
        )

    error_rate = errors / total

    return {
        "total_samples": int(total),
        "incorrect_samples": int(errors),
        "error_rate": float(error_rate),
        "mean_error_confidence": (
            float(
                incorrect["confidence"].mean()
            )
            if errors
            else 0.0
        ),
        "highest_confidence_error": (
            float(
                incorrect["confidence"].max()
            )
            if errors
            else 0.0
        ),
        "lowest_confidence_error": (
            float(
                incorrect["confidence"].min()
            )
            if errors
            else 0.0
        ),
    }


def summarize_error_pairs(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count the direction of classification errors.

    Example:
        normal -> glaucoma
        glaucoma -> normal
    """

    incorrect = get_incorrect_predictions(table)

    if incorrect.empty:
        return pd.DataFrame(
            columns=[
                "true_label",
                "predicted_label",
                "count",
            ]
        )

    summary = (
        incorrect.groupby(
            [
                "true_label",
                "predicted_label",
            ]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return summary


def save_error_analysis(
    table: pd.DataFrame,
    output_dir: str | Path,
) -> dict:
    """
    Save detailed incorrect predictions and summaries.
    """

    validate_prediction_table(table)

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    incorrect = get_incorrect_predictions(
        table
    )

    low_confidence = sort_errors_by_confidence(
        table,
        ascending=True,
    )

    high_confidence = sort_errors_by_confidence(
        table,
        ascending=False,
    )

    summary = summarize_errors(table)

    error_pairs = summarize_error_pairs(table)

    incorrect.to_csv(
        output_dir / "incorrect_predictions.csv",
        index=False,
    )

    low_confidence.to_csv(
        output_dir / "errors_low_confidence_first.csv",
        index=False,
    )

    high_confidence.to_csv(
        output_dir / "errors_high_confidence_first.csv",
        index=False,
    )

    error_pairs.to_csv(
        output_dir / "error_pairs.csv",
        index=False,
    )

    pd.DataFrame(
        [summary]
    ).to_json(
        output_dir / "error_summary.json",
        orient="records",
        indent=2,
    )

    return summary