from pathlib import Path

import pandas as pd
from PIL import Image
import pytest

from ml.dataset import ManifestDataset, build_class_mapping, class_counts, load_manifest


def test_manifest_dataset_contract(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    image_path = data_root / "img1.jpg"
    Image.new("RGB", (32, 32), (128, 64, 32)).save(image_path)
    manifest = tmp_path / "train.csv"
    pd.DataFrame([{"path": "img1.jpg", "label": "glaucoma"}]).to_csv(manifest, index=False)
    train_df = load_manifest(manifest)
    combined_df = pd.concat([train_df, pd.DataFrame([{"path": "img2.jpg", "label": "normal"}])], ignore_index=True)
    mapping = build_class_mapping(combined_df)
    counts = class_counts(combined_df, mapping)
    dataset = ManifestDataset(manifest, data_root, mapping)
    image, label = dataset[0]
    assert image.mode == "RGB"
    assert label == mapping["glaucoma"]
    assert counts.tolist() == [1.0, 1.0]


def test_manifest_requires_path_and_label(tmp_path: Path):
    manifest = tmp_path / "bad.csv"
    pd.DataFrame([{"file": "img.jpg", "target": "glaucoma"}]).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_manifest(manifest)
