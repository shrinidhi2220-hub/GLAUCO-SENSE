from __future__ import annotations

from pathlib import Path

SEED = 42
IMAGE_SIZE = 224
ARCHITECTURE = "resnet18"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 4

DATA_ROOT = Path("data")
SPLITS_DIR = DATA_ROOT / "splits"
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
