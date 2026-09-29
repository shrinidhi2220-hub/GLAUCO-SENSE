from __future__ import annotations

from pathlib import Path

import torch


CHECKPOINT = Path(
    "models/checkpoints/hygd_resnet18_baseline.pt"
)


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint["classes"] = [
        "GON+",
        "GON-",
    ]

    temporary = CHECKPOINT.with_suffix(
        ".metadata_tmp.pt"
    )

    torch.save(
        checkpoint,
        temporary,
    )

    temporary.replace(CHECKPOINT)

    print("Checkpoint metadata updated.")
    print(f"Checkpoint: {CHECKPOINT.resolve()}")
    print(f"Classes: {checkpoint['classes']}")


if __name__ == "__main__":
    main()