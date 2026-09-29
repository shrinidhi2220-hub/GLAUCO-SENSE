from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

LABELS_PATH = (
    ROOT
    / "data"
    / "raw"
    / "HYGD"
    / "Labels.csv"
)

QUALITY_REPORT = (
    ROOT
    / "data"
    / "reports"
    / "image_quality_report.json"
)

OUTPUT = (
    ROOT
    / "data"
    / "reports"
    / "duplicate_analysis.json"
)


def main() -> None:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Labels file not found: {LABELS_PATH}"
        )

    if not QUALITY_REPORT.exists():
        raise FileNotFoundError(
            f"Quality report not found: {QUALITY_REPORT}"
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

    with LABELS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        labels = list(csv.DictReader(file))

    label_map = {
        row["Image Name"].strip(): {
            "patient": row["Patient"].strip(),
            "label": row["Label"].strip(),
            "quality_score": row["Quality Score"].strip(),
        }
        for row in labels
    }

    results = []

    for group_index, group in enumerate(
        duplicate_groups,
        start=1,
    ):
        items = []

        for image_name in group:
            metadata = label_map.get(
                image_name
            )

            image_path = (
                ROOT
                / "data"
                / "raw"
                / "HYGD"
                / "Images"
                / image_name
            )

            width = None
            height = None

            if image_path.exists():
                with Image.open(image_path) as image:
                    width = image.width
                    height = image.height

            items.append(
                {
                    "image_name": image_name,
                    "patient": (
                        metadata["patient"]
                        if metadata
                        else None
                    ),
                    "label": (
                        metadata["label"]
                        if metadata
                        else None
                    ),
                    "quality_score": (
                        metadata["quality_score"]
                        if metadata
                        else None
                    ),
                    "width": width,
                    "height": height,
                }
            )

        results.append(
            {
                "group_id": group_index,
                "images": items,
                "same_patient": len(
                    {
                        item["patient"]
                        for item in items
                    }
                ) == 1,
                "same_label": len(
                    {
                        item["label"]
                        for item in items
                    }
                ) == 1,
            }
        )

    OUTPUT.write_text(
        json.dumps(
            {
                "duplicate_groups": results,
                "total_groups": len(results),
                "groups_with_label_conflict": sum(
                    not group["same_label"]
                    for group in results
                ),
                "groups_across_multiple_patients": sum(
                    not group["same_patient"]
                    for group in results
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Duplicate groups analyzed: {len(results)}"
    )

    print(
        "Groups with label conflict: "
        f"{sum(not group['same_label'] for group in results)}"
    )

    print(
        "Groups spanning multiple patients: "
        f"{sum(not group['same_patient'] for group in results)}"
    )

    print(
        f"Report saved to: {OUTPUT}"
    )


if __name__ == "__main__":
    main()