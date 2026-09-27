from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    """Dataset reader for the split CSVs produced by Person 1.

    Required columns in the CSV:
      - path: path relative to --data-root
      - label: string class label
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
        required = {"path", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"{self.manifest} is missing required columns: {sorted(missing)}"
            )
        if self.df.empty:
            raise ValueError(f"{self.manifest} is empty.")

        unknown = sorted(
            set(self.df["label"].astype(str)) - set(self.class_to_id)
        )
        if unknown:
            raise ValueError(
                f"Unknown labels in {self.manifest}: {unknown}. "
                "Class mapping must come from the training split/model."
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        relative_path = Path(str(row["path"]))
        path = relative_path if relative_path.is_absolute() else self.data_root / relative_path

        try:
            image = Image.open(path).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Unable to read image: {path}") from exc

        if self.transform is not None:
            image = self.transform(image)

        target = self.class_to_id[str(row["label"])]
        return image, target


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df


def build_class_mapping(train_df: pd.DataFrame) -> dict[str, int]:
    classes = sorted(train_df["label"].astype(str).unique().tolist())
    if len(classes) < 2:
        raise ValueError(f"At least two classes are required; found {classes}")
    return {name: idx for idx, name in enumerate(classes)}


def class_counts(df: pd.DataFrame, class_to_id: dict[str, int]) -> torch.Tensor:
    counts = torch.zeros(len(class_to_id), dtype=torch.float32)
    for label, count in df["label"].astype(str).value_counts().items():
        counts[class_to_id[label]] = float(count)
    return counts
