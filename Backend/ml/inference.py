from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from ml.model import build_baseline_model
from ml.transforms import build_eval_transform


DEFAULT_CLASSES = [
    "glaucoma",
    "normal",
]


class GlaucomaInference:
    """
    Reusable single-image inference engine.

    This class is independent of Person 1's dataset.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        classes: Sequence[str] | None = None,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(
            checkpoint_path
        )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        self.classes = list(
            classes or DEFAULT_CLASSES
        )

        if len(self.classes) != 2:
            raise ValueError(
                "Inference currently requires exactly "
                "two classes."
            )

        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model = build_baseline_model(
            num_classes=len(self.classes),
            pretrained=False,
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if "model_state_dict" not in checkpoint:
            raise ValueError(
                "Checkpoint does not contain "
                "'model_state_dict'."
            )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = build_eval_transform()

    @torch.no_grad()
    def predict(
        self,
        image_path: str | Path,
    ) -> dict:
        """
        Predict a single image.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        try:
            image = Image.open(
                image_path
            ).convert("RGB")
        except Exception as exc:
            raise ValueError(
                f"Unable to read image: "
                f"{image_path}"
            ) from exc

        tensor = self.transform(
            image
        ).unsqueeze(0)

        tensor = tensor.to(self.device)

        logits = self.model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence, predicted_index = (
            probabilities.max(dim=1)
        )

        index = int(
            predicted_index.item()
        )

        return {
            "prediction": self.classes[index],
            "class_index": index,
            "confidence": float(
                confidence.item()
            ),
            "probabilities": {
                class_name: float(
                    probabilities[0, class_id].item()
                )
                for class_id, class_name
                in enumerate(self.classes)
            },
        }