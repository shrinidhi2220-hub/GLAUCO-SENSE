from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def load_labels(labels_csv: Path) -> dict[str, dict[str, str]]:
    """Load HYGD metadata indexed by image filename."""
    records: dict[str, dict[str, str]] = {}

    with labels_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        required = {"Image Name", "Patient", "Label", "Quality Score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Labels.csv is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            image_name = row["Image Name"].strip()
            if not image_name:
                continue

            records[image_name] = {
                "patient": row["Patient"].strip(),
                "label": row["Label"].strip(),
                "quality_score": row["Quality Score"].strip(),
            }

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess cleaned HYGD glaucoma images."
    )

    parser.add_argument(
        "--input",
        default=r"data\cleaned\Images",
        help="Cleaned image directory.",
    )

    parser.add_argument(
        "--labels",
        default=r"data\raw\HYGD\Labels.csv",
        help="HYGD Labels.csv file.",
    )

    parser.add_argument(
        "--output",
        default=r"data\processed",
        help="Processed dataset directory.",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Target image size.",
    )

    args = parser.parse_args()

    source = Path(args.input)
    labels_csv = Path(args.labels)
    destination = Path(args.output)
    output_images = destination / "Images"

    if not source.exists():
        raise FileNotFoundError(f"Input directory not found: {source}")

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")

    output_images.mkdir(parents=True, exist_ok=True)

    labels = load_labels(labels_csv)

    processed = 0
    skipped = 0
    missing_metadata = 0

    manifest_rows: list[dict[str, str]] = []

    image_paths = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )

    print(f"Input images found: {len(image_paths)}")
    print(f"Metadata records loaded: {len(labels)}")

    for path in image_paths:
        image_name = path.name

        metadata = labels.get(image_name)

        if metadata is None:
            missing_metadata += 1
            print(f"SKIP: No metadata found for {image_name}")
            continue

        output_name = f"{path.stem}.jpg"
        output_path = output_images / output_name

        try:
            with Image.open(path) as image:
                original_width, original_height = image.size

                image = ImageOps.exif_transpose(image).convert("RGB")
                image = image.resize(
                    (args.size, args.size),
                    Image.Resampling.LANCZOS,
                )

                image.save(
                    output_path,
                    format="JPEG",
                    quality=95,
                )

            manifest_rows.append(
                {
                    "image_name": image_name,
                    "patient": metadata["patient"],
                    "label": metadata["label"],
                    "quality_score": metadata["quality_score"],
                    "processed_path": str(
                        Path("data") / "processed" / "Images" / output_name
                    ),
                    "original_width": str(original_width),
                    "original_height": str(original_height),
                    "processed_width": str(args.size),
                    "processed_height": str(args.size),
                    "channels": "RGB",
                }
            )

            processed += 1

        except Exception as exc:
            skipped += 1
            print(f"SKIP: {image_name} :: {exc}")

    manifest_path = destination / "processed_manifest.csv"

    fieldnames = [
        "image_name",
        "patient",
        "label",
        "quality_score",
        "processed_path",
        "original_width",
        "original_height",
        "processed_width",
        "processed_height",
        "channels",
    ]

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print()
    print("Preprocessing completed.")
    print(f"Input images: {len(image_paths)}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Missing metadata: {missing_metadata}")
    print(f"Output directory: {output_images.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()