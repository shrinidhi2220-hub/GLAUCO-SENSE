from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from ml.config import BATCH_SIZE
from ml.dataset import (
    ManifestDataset,
    build_class_mapping,
    load_manifest,
)
from ml.transforms import (
    build_eval_transform,
    build_train_transform,
)


def build_real_dataloaders(
    splits_dir: str | Path = "data/splits",
    data_root: str | Path = ".",
    batch_size: int = BATCH_SIZE,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    dict[str, int],
]:
    """Build train, validation and test DataLoaders from real HYGD manifests.

    Person 1's processed_path values are project-root-relative, for example:
        data/processed/Images/119_1.jpg

    Therefore the project root is used as data_root.
    """

    splits_dir = Path(splits_dir)
    data_root = Path(data_root)

    train_manifest = splits_dir / "train.csv"
    val_manifest = splits_dir / "val.csv"
    test_manifest = splits_dir / "test.csv"

    for manifest in (
        train_manifest,
        val_manifest,
        test_manifest,
    ):
        if not manifest.exists():
            raise FileNotFoundError(
                f"Required split manifest not found: {manifest}"
            )

    train_df = load_manifest(train_manifest)
    val_df = load_manifest(val_manifest)
    test_df = load_manifest(test_manifest)

    class_to_id = build_class_mapping(train_df)

    train_dataset = ManifestDataset(
        manifest=train_manifest,
        data_root=data_root,
        class_to_id=class_to_id,
        transform=build_train_transform(),
    )

    val_dataset = ManifestDataset(
        manifest=val_manifest,
        data_root=data_root,
        class_to_id=class_to_id,
        transform=build_eval_transform(),
    )

    test_dataset = ManifestDataset(
        manifest=test_manifest,
        data_root=data_root,
        class_to_id=class_to_id,
        transform=build_eval_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    if len(train_dataset) != len(train_df):
        raise RuntimeError(
            f"Train dataset mismatch: "
            f"{len(train_dataset)} != {len(train_df)}"
        )

    if len(val_dataset) != len(val_df):
        raise RuntimeError(
            f"Validation dataset mismatch: "
            f"{len(val_dataset)} != {len(val_df)}"
        )

    if len(test_dataset) != len(test_df):
        raise RuntimeError(
            f"Test dataset mismatch: "
            f"{len(test_dataset)} != {len(test_df)}"
        )

    return (
        train_loader,
        val_loader,
        test_loader,
        class_to_id,
    )