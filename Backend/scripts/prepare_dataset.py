from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and resize class-folder images.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    source = Path(args.input)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)

    seen = set()
    processed = skipped = duplicates = 0

    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        label = path.parent.name
        try:
            digest = file_hash(path)
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image = image.resize((args.size, args.size), Image.Resampling.LANCZOS)
                out_dir = destination / label
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{path.stem}_{digest[:10]}.jpg"
                image.save(out_path, format="JPEG", quality=95)
            processed += 1
        except Exception as exc:
            skipped += 1
            print(f"SKIP: {path} :: {exc}")

    print(f"Processed={processed} | skipped={skipped} | duplicate_files={duplicates}")


if __name__ == "__main__":
    main()
