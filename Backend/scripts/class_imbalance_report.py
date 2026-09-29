from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def analyze_split(
    df: pd.DataFrame,
    split_name: str,
) -> list[dict]:
    """Calculate class distribution for one dataset split."""

    if "label" not in df.columns:
        raise ValueError(
            f"Manifest for '{split_name}' does not contain a 'label' column."
        )

    labels = df["label"].astype(str).str.strip()
    counts = labels.value_counts().sort_index()

    total = int(counts.sum())

    if total == 0:
        raise ValueError(
            f"Manifest for '{split_name}' contains no images."
        )

    majority_count = int(counts.max())
    minority_count = int(counts.min())

    majority_to_minority = (
        majority_count / minority_count
        if minority_count > 0
        else None
    )

    rows = []

    for label, count in counts.items():
        count = int(count)

        rows.append(
            {
                "split": split_name,
                "label": label,
                "count": count,
                "percentage": round((count / total) * 100, 4),
                "majority_to_class_ratio": round(
                    majority_count / count,
                    4,
                ),
                "total_samples": total,
                "majority_to_minority_ratio": round(
                    majority_to_minority,
                    4,
                )
                if majority_to_minority is not None
                else None,
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate class imbalance analysis for GLAUCO-SENSE."
    )

    parser.add_argument(
        "--splits-dir",
        default=r"data\splits",
        help="Directory containing train.csv, val.csv and test.csv.",
    )

    parser.add_argument(
        "--output",
        default=r"data\reports\class_imbalance.csv",
        help="CSV output path.",
    )

    parser.add_argument(
        "--json-output",
        default=r"data\reports\class_imbalance_report.json",
        help="JSON output path.",
    )

    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)

    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "test.csv"

    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required split manifest not found: {path}"
            )

    split_paths = {
        "train": train_path,
        "val": val_path,
        "test": test_path,
    }

    all_rows: list[dict] = []

    split_dataframes: dict[str, pd.DataFrame] = {}

    for split_name, path in split_paths.items():
        df = pd.read_csv(path)

        if df.empty:
            raise ValueError(
                f"{split_name}.csv is empty."
            )

        split_dataframes[split_name] = df

        all_rows.extend(
            analyze_split(
                df,
                split_name,
            )
        )

    # Combined dataset analysis.
    combined = pd.concat(
        split_dataframes.values(),
        ignore_index=True,
    )

    all_rows.extend(
        analyze_split(
            combined,
            "overall",
        )
    )

    report = pd.DataFrame(all_rows)

    split_order = {
        "overall": 0,
        "train": 1,
        "val": 2,
        "test": 3,
    }

    report["_order"] = report["split"].map(split_order)

    report = (
        report
        .sort_values(["_order", "label"])
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )

    # CSV report.
    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        output_path,
        index=False,
    )

    # JSON summary.
    json_output_path = Path(args.json_output)
    json_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_rows = report[
        report["split"] == "overall"
    ].to_dict(orient="records")

    split_summaries = {}

    for split_name in ["train", "val", "test"]:
        split_rows = report[
            report["split"] == split_name
        ]

        split_summaries[split_name] = {
            "total_samples": int(
                split_rows["total_samples"].iloc[0]
            ),
            "classes": split_rows[
                [
                    "label",
                    "count",
                    "percentage",
                ]
            ].to_dict(orient="records"),
            "majority_to_minority_ratio": float(
                split_rows[
                    "majority_to_minority_ratio"
                ].iloc[0]
            ),
        }

    summary = {
        "dataset": "HYGD cleaned and patient-level split dataset",
        "overall": {
            "total_samples": int(
                report[
                    report["split"] == "overall"
                ]["total_samples"].iloc[0]
            ),
            "classes": overall_rows,
            "majority_to_minority_ratio": float(
                report[
                    report["split"] == "overall"
                ]["majority_to_minority_ratio"].iloc[0]
            ),
        },
        "splits": split_summaries,
    }

    json_output_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Class imbalance analysis completed.")
    print()
    print(report.to_string(index=False))
    print()
    print(f"CSV report: {output_path.resolve()}")
    print(f"JSON report: {json_output_path.resolve()}")


if __name__ == "__main__":
    main()