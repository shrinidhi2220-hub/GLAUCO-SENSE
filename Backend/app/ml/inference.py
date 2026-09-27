from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import models, transforms

DEFAULT_CLASSES = ["glaucoma", "normal"]
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def load_model(model_path: str | Path) -> tuple[Any | None, list[str] | None]:
    path = Path(model_path)
    if not path.exists():
        return None, None

    checkpoint = torch.load(path, map_location="cpu")
    classes = [str(x) for x in checkpoint.get("classes", DEFAULT_CLASSES)]
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model, classes


def predict_image(image: Image.Image, model: Any, class_names: list[str] | None = None) -> dict:
    classes = class_names or DEFAULT_CLASSES
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ]
    )
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(torch.argmax(probabilities).item())
    return {
        "class": classes[index],
        "confidence": float(probabilities[index].item()),
        "probabilities": {
            classes[i]: float(probabilities[i].item()) for i in range(len(classes))
        },
    }
