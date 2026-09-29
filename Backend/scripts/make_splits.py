from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def build_patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build one row per patient and verify one label per patient."""

    required = {
        "image_name",
        "patient",
        "label",
        "quality_score",
        "processed_path",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"processed_manifest.csv is missing required columns: {sorted(missing)}"
        )

    df = df.copy()

    # Always treat patient IDs as strings.
    df["patient"] = df["patient"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    # Every patient must have exactly one class label.
    label_counts = df.groupby("patient")["label"].nunique()

    conflicting = label_counts[label_counts > 1]

    if not conflicting.empty:
        patients = conflicting.index.tolist()

        raise ValueError(
            "Some patients have multiple labels: "
            f"{patients[:20]}"
        )

    patients = (
        df[["patient", "label"]]
        .drop_duplicates()
        .sort_values("patient")
        .reset_index(drop=True)
    )

    return patients


def split_patients(
    patients: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split patients into train, validation and test sets."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")

    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1.")

    class_counts = patients["label"].value_counts()

    if class_counts.min() < 3:
        raise ValueError(
            "Each class needs at least 3 patients for stratified splitting. "
            f"Counts: {class_counts.to_dict()}"
        )

    # First create the test patient set.
    train_val, test = train_test_split(
        patients,
        test_size=test_size,
        random_state=seed,
        stratify=patients["label"],
    )

    # Calculate validation proportion relative to remaining patients.
    relative_val = val_size / (1.0 - test_size)

    # Split remaining patients into train and validation.
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=train_val["label"],
    )

    return train, val, test


def attach_split(
    images: pd.DataFrame,
    patients: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Attach split information to every image belonging to selected patients."""

    selected_patients = patients[["patient"]].copy()
    selected_patients["split"] = split_name

    return images.merge(
        selected_patients,
        on="patient",
        how="inner",
        validate="many_to_one",
    )


def verify_no_patient_overlap(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Fail if any patient appears in more than one split."""

    train_patients = set(train["patient"])
    val_patients = set(val["patient"])
    test_patients = set(test["patient"])

    train_val = train_patients & val_patients
    train_test = train_patients & test_patients
    val_test = val_patients & test_patients

    if train_val or train_test or val_test:
        raise RuntimeError(
            "Patient leakage detected: "
            f"train_val={sorted(train_val)}, "
            f"train_test={sorted(train_test)}, "
            f"val_test={sorted(val_test)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create reproducible patient-level train/val/test splits."
    )

    parser.add_argument(
        "--input",
        default=r"data\processed\processed_manifest.csv",
        help="Processed dataset manifest.",
    )

    parser.add_argument(
        "--output",
        default=r"data\splits",
        help="Directory for split manifests.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction of patients for test.",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Fraction of patients for validation.",
    )

    args = parser.parse_args()

    manifest_path = Path(args.input)
    output_dir = Path(args.output)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Processed manifest not found: {manifest_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load processed manifest.
    images = pd.read_csv(manifest_path)

    if images.empty:
        raise ValueError("Processed manifest is empty.")

    # IMPORTANT:
    # Force patient IDs and labels to strings before any merge/grouping.
    images["patient"] = images["patient"].astype(str).str.strip()
    images["label"] = images["label"].astype(str).str.strip()

    # Build one row per patient.
    patients = build_patient_table(images)

    # Patient-level split.
    train_patients, val_patients, test_patients = split_patients(
        patients,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    # Attach split to every image.
    train = attach_split(images, train_patients, "train")
    val = attach_split(images, val_patients, "val")
    test = attach_split(images, test_patients, "test")

    # Verify no patient leakage.
    verify_no_patient_overlap(train, val, test)

    # Create reproducible label mapping.
    label_map = (
        pd.DataFrame({"label": sorted(images["label"].unique())})
        .reset_index()
        .rename(columns={"index": "label_id"})
    )

    label_to_id = dict(
        zip(
            label_map["label"],
            label_map["label_id"],
        )
    )

    for dataframe in [train, val, test]:
        dataframe["label_id"] = dataframe["label"].map(label_to_id)

    columns = [
        "image_name",
        "patient",
        "label",
        "label_id",
        "quality_score",
        "processed_path",
        "original_width",
        "original_height",
        "processed_width",
        "processed_height",
        "channels",
        "split",
    ]

    train = train[columns].sort_values(
        ["patient", "image_name"]
    )

    val = val[columns].sort_values(
        ["patient", "image_name"]
    )

    test = test[columns].sort_values(
        ["patient", "image_name"]
    )

    # Save split manifests.
    train.to_csv(
        output_dir / "train.csv",
        index=False,
    )

    val.to_csv(
        output_dir / "val.csv",
        index=False,
    )

    test.to_csv(
        output_dir / "test.csv",
        index=False,
    )

    label_map.to_csv(
        output_dir / "label_map.csv",
        index=False,
    )

    # Class counts by split.
    split_class_counts = pd.concat(
        [
            train.groupby(["split", "label"])
            .size()
            .reset_index(name="image_count"),

            val.groupby(["split", "label"])
            .size()
            .reset_index(name="image_count"),

            test.groupby(["split", "label"])
            .size()
            .reset_index(name="image_count"),
        ],
        ignore_index=True,
    )

    split_class_counts.to_csv(
        output_dir / "split_class_counts.csv",
        index=False,
    )

    # Patient and image counts.
    patient_counts = pd.DataFrame(
        [
            {
                "split": "train",
                "patient_count": train["patient"].nunique(),
                "image_count": len(train),
            },
            {
                "split": "val",
                "patient_count": val["patient"].nunique(),
                "image_count": len(val),
            },
            {
                "split": "test",
                "patient_count": test["patient"].nunique(),
                "image_count": len(test),
            },
        ]
    )

    patient_counts.to_csv(
        output_dir / "split_patient_counts.csv",
        index=False,
    )

    # Save split configuration and verification information.
    info = [
        f"seed={args.seed}",
        "split_unit=patient",
        f"test_size={args.test_size}",
        f"val_size={args.val_size}",
        f"total_patients={patients['patient'].nunique()}",
        f"total_images={len(images)}",
        f"train_patients={train['patient'].nunique()}",
        f"val_patients={val['patient'].nunique()}",
        f"test_patients={test['patient'].nunique()}",
        f"train_images={len(train)}",
        f"val_images={len(val)}",
        f"test_images={len(test)}",
        "patient_overlap_train_val=0",
        "patient_overlap_train_test=0",
        "patient_overlap_val_test=0",
    ]

    (output_dir / "split_info.txt").write_text(
        "\n".join(info) + "\n",
        encoding="utf-8",
    )

    print("Patient-level splitting completed.")
    print(
        f"Patients: {patients['patient'].nunique()} | "
        f"Images: {len(images)}"
    )

    print(
        f"Train: {train['patient'].nunique()} patients | "
        f"{len(train)} images"
    )

    print(
        f"Val: {val['patient'].nunique()} patients | "
        f"{len(val)} images"
    )

    print(
        f"Test: {test['patient'].nunique()} patients | "
        f"{len(test)} images"
    )

    print("Patient overlap: 0 across all splits.")
    print(f"Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()