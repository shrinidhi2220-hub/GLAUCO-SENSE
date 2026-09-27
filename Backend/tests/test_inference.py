from pathlib import Path

import torch
from PIL import Image

from ml.inference import GlaucomaInference
from ml.model import build_baseline_model


def create_test_checkpoint(
    path: Path,
) -> None:
    model = build_baseline_model(
        num_classes=2,
        pretrained=False,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def create_test_image(
    path: Path,
) -> None:
    image = Image.new(
        "RGB",
        (640, 480),
        (128, 96, 64),
    )

    image.save(path)


def test_inference_engine_predicts_image(
    tmp_path: Path,
):
    checkpoint = (
        tmp_path / "test_model.pt"
    )

    image = (
        tmp_path / "test_image.jpg"
    )

    create_test_checkpoint(
        checkpoint
    )

    create_test_image(
        image
    )

    engine = GlaucomaInference(
        checkpoint_path=checkpoint,
        device="cpu",
    )

    result = engine.predict(
        image
    )

    assert result["prediction"] in {
        "glaucoma",
        "normal",
    }

    assert result["class_index"] in {
        0,
        1,
    }

    assert 0.0 <= result[
        "confidence"
    ] <= 1.0

    assert set(
        result["probabilities"].keys()
    ) == {
        "glaucoma",
        "normal",
    }

    total_probability = sum(
        result["probabilities"].values()
    )

    assert abs(
        total_probability - 1.0
    ) < 1e-6


def test_missing_checkpoint_fails(
    tmp_path: Path,
):
    missing = (
        tmp_path / "missing.pt"
    )

    try:
        GlaucomaInference(
            checkpoint_path=missing,
            device="cpu",
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError(
            "Expected FileNotFoundError."
        )


def test_missing_image_fails(
    tmp_path: Path,
):
    checkpoint = (
        tmp_path / "test_model.pt"
    )

    create_test_checkpoint(
        checkpoint
    )

    engine = GlaucomaInference(
        checkpoint_path=checkpoint,
        device="cpu",
    )

    missing_image = (
        tmp_path / "missing.jpg"
    )

    try:
        engine.predict(
            missing_image
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError(
            "Expected FileNotFoundError."
        )