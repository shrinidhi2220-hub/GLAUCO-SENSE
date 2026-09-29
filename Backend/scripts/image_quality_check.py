from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]

LABELS_PATH = ROOT / "data" / "raw" / "HYGD" / "Labels.csv"
IMAGE_DIR = ROOT / "data" / "raw" / "HYGD" / "Images"
REPORT_DIR = ROOT / "data" / "reports"

CSV_REPORT = REPORT_DIR / "image_quality_report.csv"
JSON_REPORT = REPORT_DIR / "image_quality_report.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Labels file not found: {LABELS_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    with LABELS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            image_name = row["Image Name"].strip()

            image_path = IMAGE_DIR / image_name

            record = {
                "image_name": image_name,
                "exists": image_path.exists(),
                "readable": False,
                "width": None,
                "height": None,
                "mode": None,
                "channels": None,
                "sha256": None,
                "error": None,
            }

            if not image_path.exists():
                record["error"] = "missing_file"
                rows.append(record)
                continue

            try:
                with Image.open(image_path) as image:
                    image.verify()

                with Image.open(image_path) as image:
                    record["readable"] = True
                    record["width"] = image.width
                    record["height"] = image.height
                    record["mode"] = image.mode

                    channels = {
                        "1": 1,
                        "L": 1,
                        "LA": 2,
                        "RGB": 3,
                        "RGBA": 4,
                        "CMYK": 4,
                    }.get(image.mode)

                    record["channels"] = channels

                record["sha256"] = sha256_file(
                    image_path
                )

            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
            ) as exc:
                record["error"] = str(exc)

            rows.append(record)

    # Save detailed CSV report.
    fieldnames = list(rows[0].keys())

    with CSV_REPORT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    readable_rows = [
        row
        for row in rows
        if row["readable"]
    ]

    corrupt_rows = [
        row
        for row in rows
        if not row["readable"]
    ]

    non_rgb_rows = [
        row
        for row in readable_rows
        if row["channels"] != 3
    ]

    dimensions = Counter(
        (
            row["width"],
            row["height"],
        )
        for row in readable_rows
    )

    hashes = defaultdict(list)

    for row in readable_rows:
        hashes[row["sha256"]].append(
            row["image_name"]
        )

    duplicate_groups = [
        names
        for names in hashes.values()
        if len(names) > 1
    ]

    summary = {
        "total_manifest_images": len(rows),
        "readable_images": len(readable_rows),
        "corrupt_or_unreadable_images": len(
            corrupt_rows
        ),
        "non_rgb_images": len(non_rgb_rows),
        "unique_content_hashes": len(hashes),
        "duplicate_content_groups": len(
            duplicate_groups
        ),
        "duplicate_images": sum(
            len(group) - 1
            for group in duplicate_groups
        ),
        "dimensions": {
            f"{width}x{height}": count
            for (width, height), count
            in dimensions.items()
        },
        "corrupt_images": [
            row["image_name"]
            for row in corrupt_rows
        ],
        "non_rgb_image_names": [
            row["image_name"]
            for row in non_rgb_rows
        ],
        "duplicate_groups": duplicate_groups,
    }

    JSON_REPORT.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Image quality check completed.")
    print(
        f"Total manifest images: "
        f"{summary['total_manifest_images']}"
    )
    print(
        f"Readable images: "
        f"{summary['readable_images']}"
    )
    print(
        f"Corrupt/unreadable: "
        f"{summary['corrupt_or_unreadable_images']}"
    )
    print(
        f"Non-RGB images: "
        f"{summary['non_rgb_images']}"
    )
    print(
        f"Duplicate-content groups: "
        f"{summary['duplicate_content_groups']}"
    )
    print(
        f"Duplicate images: "
        f"{summary['duplicate_images']}"
    )
    print(
        f"CSV report: {CSV_REPORT}"
    )
    print(
        f"JSON report: {JSON_REPORT}"
    )


if __name__ == "__main__":
    main()