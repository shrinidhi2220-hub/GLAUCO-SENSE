from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def stratified_split(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    if df["label"].value_counts().min() < 3:
        raise SystemExit("Each class needs at least 3 images for stratified train/val/test splitting.")

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=train_val["label"],
    )
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reproducible stratified split manifests.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    args = parser.parse_args()

    root = Path(args.input)
    class_names = sorted(p.name for p in root.iterdir() if p.is_dir())
    label_map = {name: i for i, name in enumerate(class_names)}

    rows = []
    for label in class_names:
        for path in sorted((root / label).rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                rows.append({
                    "path": path.relative_to(root).as_posix(),
                    "label": label,
                    "label_id": label_map[label],
                })

    df = pd.DataFrame(rows)
    if df.empty or len(class_names) < 2:
        raise SystemExit("Need at least two class folders containing images.")

    train, val, test = stratified_split(df, args.test_size, args.val_size, args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for name, subset in [("train", train), ("val", val), ("test", test)]:
        subset.sort_values(["label_id", "path"]).to_csv(out / f"{name}.csv", index=False)

    pd.DataFrame([
        {"label_id": label_id, "label": label}
        for label, label_id in label_map.items()
    ]).sort_values("label_id").to_csv(out / "label_map.csv", index=False)

    summary = pd.concat([
        train.assign(split="train"),
        val.assign(split="val"),
        test.assign(split="test"),
    ]).groupby(["split", "label"], as_index=False).size()
    summary.rename(columns={"size": "count"}).to_csv(out / "split_class_counts.csv", index=False)

    (out / "split_info.txt").write_text(
        f"seed={args.seed}\ntrain={len(train)}\nval={len(val)}\ntest={len(test)}\nclasses={class_names}\n",
        encoding="utf-8",
    )
    print(f"Classes={class_names}")
    print(f"Train={len(train)} | Val={len(val)} | Test={len(test)}")


if __name__ == "__main__":
    main()
