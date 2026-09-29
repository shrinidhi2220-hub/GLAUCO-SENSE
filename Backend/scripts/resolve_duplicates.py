from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RAW_IMAGE_DIR = (
    ROOT
    / "data"
    / "raw"
    / "HYGD"
    / "Images"
)

QUALITY_REPORT = (
    ROOT
    / "data"
    / "reports"
    / "image_quality_report.json"
)

CLEANED_IMAGE_DIR = (
    ROOT
    / "data"
    / "cleaned"
    / "Images"
)

REPORT_DIR = (
    ROOT
    / "data"
    / "reports"
)

RESOLUTION_REPORT = (
    REPORT_DIR
    / "duplicate_resolution.csv"
)


def main() -> None:
    if not RAW_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Raw image directory not found: {RAW_IMAGE_DIR}"
        )

    if not QUALITY_REPORT.exists():
        raise FileNotFoundError(
            f"Quality report not found: {QUALITY_REPORT}"
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CLEANED_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = json.loads(
        QUALITY_REPORT.read_text(
            encoding="utf-8"
        )
    )

    duplicate_groups = report.get(
        "duplicate_groups",
        [],
    )

    duplicate_lookup = {}

    for group in duplicate_groups:
        if not group:
            continue

        keep = group[0]

        for name in group[1:]:
            duplicate_lookup[name] = {
                "keep": keep,
                "group_size": len(group),
            }

    raw_images = sorted(
        RAW_IMAGE_DIR.glob("*.jpg")
    )

    resolution_rows = []

    kept_count = 0
    excluded_count = 0

    for source_path in raw_images:
        name = source_path.name

        if name in duplicate_lookup:
            info = duplicate_lookup[name]

            resolution_rows.append(
                {
                    "image_name": name,
                    "action": "exclude_duplicate",
                    "kept_image": info["keep"],
                    "reason": "exact_content_duplicate",
                }
            )

            excluded_count += 1
            continue

        destination = (
            CLEANED_IMAGE_DIR
            / name
        )

        shutil.copy2(
            source_path,
            destination,
        )

        resolution_rows.append(
            {
                "image_name": name,
                "action": "keep",
                "kept_image": name,
                "reason": "unique_content",
            }
        )

        kept_count += 1

    with RESOLUTION_REPORT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_name",
                "action",
                "kept_image",
                "reason",
            ],
        )

        writer.writeheader()
        writer.writerows(
            resolution_rows
        )

    print("Duplicate resolution completed.")
    print(
        f"Raw images: {len(raw_images)}"
    )
    print(
        f"Unique images copied: {kept_count}"
    )
    print(
        f"Duplicate copies excluded: {excluded_count}"
    )
    print(
        f"Cleaned directory: {CLEANED_IMAGE_DIR}"
    )
    print(
        f"Resolution report: {RESOLUTION_REPORT}"
    )


if __name__ == "__main__":
    main()