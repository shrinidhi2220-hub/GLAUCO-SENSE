from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


def _validate_manifest_columns(
    df: pd.DataFrame,
    manifest: str | Path,
) -> str:
    """Validate manifest columns and return the image-path column."""

    has_processed_path = "processed_path" in df.columns
    has_legacy_path = "path" in df.columns
    has_label = "label" in df.columns

    missing = []

    if not has_label:
        missing.append("label")

    if not has_processed_path and not has_legacy_path:
        missing.append("path or processed_path")

    if missing:
        raise ValueError(
            f"{manifest} is missing columns: {sorted(missing)}"
        )

    if has_processed_path:
        return "processed_path"

    return "path"


class ManifestDataset(Dataset):
    """Dataset reader for GLAUCO-SENSE split manifests.

    Current Person 1 handoff:
      - processed_path: path to the processed image
      - label: class label

    Legacy compatibility:
      - path: older image-path column
      - label: class label
    """

    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path,
        class_to_id: dict[str, int],
        transform: Callable | None = None,
    ) -> None:
        self.manifest = Path(manifest)
        self.data_root = Path(data_root)
        self.class_to_id = class_to_id
        self.transform = transform

        self.df = pd.read_csv(self.manifest)

        if self.df.empty:
            raise ValueError(f"{self.manifest} is empty.")

        self.path_column = _validate_manifest_columns(
            self.df,
            self.manifest,
        )

        unknown = sorted(
            set(
                self.df["label"]
                .astype(str)
                .str.strip()
            )
            - set(self.class_to_id)
        )

        if unknown:
            raise ValueError(
                f"Unknown labels in {self.manifest}: {unknown}. "
                "Class mapping must come from the training split/model."
            )

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, value: str) -> Path:
        """Resolve image paths from the manifest."""

        raw_path = Path(str(value))

        if raw_path.is_absolute():
            return raw_path

        return self.data_root / raw_path

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        image_path = self._resolve_image_path(
            str(row[self.path_column])
        )

        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(
                f"Unable to read image: {image_path}"
            ) from exc

        if self.transform is not None:
            image = self.transform(image)

        label = str(row["label"]).strip()
        target = self.class_to_id[label]

        return image, target


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load and validate a GLAUCO-SENSE manifest."""

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{path} is empty.")

    _validate_manifest_columns(
        df,
        path,
    )

    return df


def build_class_mapping(
    train_df: pd.DataFrame,
) -> dict[str, int]:
    """Create deterministic class mapping from training labels."""

    classes = sorted(
        train_df["label"]
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if len(classes) < 2:
        raise ValueError(
            f"At least two classes are required; found {classes}"
        )

    return {
        name: idx
        for idx, name in enumerate(classes)
    }


def class_counts(
    df: pd.DataFrame,
    class_to_id: dict[str, int],
) -> torch.Tensor:
    """Return class counts ordered by class ID."""

    counts = torch.zeros(
        len(class_to_id),
        dtype=torch.float32,
    )

    for label, count in (
        df["label"]
        .astype(str)
        .str.strip()
        .value_counts()
        .items()
    ):
        if label not in class_to_id:
            raise ValueError(
                f"Unknown label '{label}' encountered."
            )

        counts[class_to_id[label]] = float(count)

    return counts