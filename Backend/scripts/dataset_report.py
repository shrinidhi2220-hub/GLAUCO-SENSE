from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_dataset(root: Path):
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    label_counts = Counter()
    size_counts = Counter()
    format_counts = Counter()
    unreadable = []
    hashes = defaultdict(list)

    for path in files:
        label = path.parent.name
        label_counts[label] += 1
        format_counts[path.suffix.lower()] += 1
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                size_counts[(image.width, image.height)] += 1
            hashes[sha256_file(path)].append(path.as_posix())
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            unreadable.append({"path": path.as_posix(), "error": str(exc)})

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    duplicate_files = sum(len(group) - 1 for group in duplicate_groups)

    return {
        "dataset_root": str(root.resolve()),
        "image_count": len(files),
        "label_counts": dict(sorted(label_counts.items())),
        "format_counts": dict(sorted(format_counts.items())),
        "common_image_sizes": [
            {"width": w, "height": h, "count": count}
            for (w, h), count in size_counts.most_common(15)
        ],
        "unreadable_count": len(unreadable),
        "unreadable": unreadable[:500],
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_file_count_after_first": duplicate_files,
        "duplicate_groups": duplicate_groups[:200],
    }


def write_outputs(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = [
        "GLAUCO-SENSE DATASET REPORT",
        "=" * 34,
        f"Dataset root: {report['dataset_root']}",
        f"Total image files: {report['image_count']}",
        f"Unreadable/corrupt: {report['unreadable_count']}",
        f"Duplicate groups: {report['duplicate_group_count']}",
        f"Duplicate files after first copy: {report['duplicate_file_count_after_first']}",
        "",
        "CLASS COUNTS",
    ]
    for label, count in report["label_counts"].items():
        lines.append(f"  {label}: {count}")
    lines += ["", "COMMON IMAGE SIZES"]
    for item in report["common_image_sizes"]:
        lines.append(f"  {item['width']}x{item['height']}: {item['count']}")
    if report["unreadable"]:
        lines += ["", "UNREADABLE FILES (first 100)"]
        lines.extend(f"  {x['path']} :: {x['error']}" for x in report["unreadable"][:100])
    (output_dir / "dataset_report.txt").write_text("\n".join(lines), encoding="utf-8")

    with (output_dir / "class_balance.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "count", "percentage"])
        total = max(report["image_count"], 1)
        for label, count in report["label_counts"].items():
            writer.writerow([label, count, round(100 * count / total, 4)])

    with (output_dir / "duplicate_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["duplicate_group", "path"])
        for group_id, group in enumerate(report["duplicate_groups"], start=1):
            for path in group:
                writer.writerow([group_id, path])


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan and document a class-folder glaucoma dataset.")
    parser.add_argument("--input", required=True, help="Dataset root containing one folder per class.")
    parser.add_argument("--output", default="data/reports")
    args = parser.parse_args()

    report = scan_dataset(Path(args.input))
    write_outputs(report, Path(args.output))
    print(json.dumps({k: report[k] for k in ["image_count", "label_counts", "unreadable_count", "duplicate_group_count"]}, indent=2))


if __name__ == "__main__":
    main()
