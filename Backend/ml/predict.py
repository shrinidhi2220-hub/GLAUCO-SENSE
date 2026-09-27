from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

try:
    from .evaluate import load_model, make_transform
except ImportError:
    from evaluate import load_model, make_transform


def predict_image(image_path: str | Path, model_path: str | Path) -> dict:
    model, classes = load_model(Path(model_path))
    transform = make_transform()
    path = Path(image_path)
    try:
        image = Image.open(path).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Unable to read image: {path}") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    pred = int(torch.argmax(probs).item())
    return {
        "image": str(path),
        "prediction": classes[pred],
        "confidence": float(probs[pred].item()),
        "probabilities": {
            classes[i]: float(probs[i].item()) for i in range(len(classes))
        },
        "research_only": True,
        "note": "This output is a research-model prediction and is not a medical diagnosis.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the saved glaucoma model on one image.")
    parser.add_argument("image")
    parser.add_argument("--model", default="models/glaucoma_resnet18.pt")
    args = parser.parse_args()
    print(json.dumps(predict_image(args.image, args.model), indent=2))


if __name__ == "__main__":
    main()
