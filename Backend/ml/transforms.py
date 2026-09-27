from __future__ import annotations

from torchvision import transforms

from ml.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def build_train_transform(image_size: int = IMAGE_SIZE):
    """Training preprocessing with light augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform(image_size: int = IMAGE_SIZE):
    """Validation/test preprocessing without random augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
