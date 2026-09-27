from pathlib import Path
import sys

import torch
from PIL import Image

# Add Backend to Python's import path so this script
# can be run directly with: python tests\create_inference_demo.py
BACKEND_ROOT = Path(__file__).resolve().parents[1]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.model import build_baseline_model


ROOT = BACKEND_ROOT / "tests" / "demo"

ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

checkpoint = ROOT / "demo_model.pt"
image = ROOT / "demo_image.jpg"

model = build_baseline_model(
    num_classes=2,
    pretrained=False,
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
    },
    checkpoint,
)

Image.new(
    "RGB",
    (640, 480),
    (100, 120, 140),
).save(image)

print(f"Checkpoint: {checkpoint}")
print(f"Image: {image}")