import torch
from PIL import Image

from ml.model import build_baseline_model
from ml.transforms import build_eval_transform, build_train_transform


def test_resnet18_binary_output():
    model = build_baseline_model(num_classes=2, pretrained=False)
    output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 2)


def test_resnet18_has_binary_classifier():
    model = build_baseline_model(num_classes=2, pretrained=False)
    assert model.fc.out_features == 2


def test_train_transform_shape():
    tensor = build_train_transform()(Image.new("RGB", (640, 480)))
    assert tensor.shape == (3, 224, 224)


def test_eval_transform_shape():
    tensor = build_eval_transform()(Image.new("RGB", (640, 480)))
    assert tensor.shape == (3, 224, 224)
