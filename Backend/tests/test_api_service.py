from pathlib import Path

import torch
from PIL import Image

from app.ml.api_service import predict_image
from ml.model import build_baseline_model


def create_checkpoint(
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


def create_image(
    path: Path,
) -> None:
    Image.new(
        "RGB",
        (640, 480),
        (110, 90, 70),
    ).save(path)


def test_predict_image_service(
    tmp_path: Path,
):
    checkpoint = (
        tmp_path / "model.pt"
    )

    image = (
        tmp_path / "image.jpg"
    )

    create_checkpoint(
        checkpoint,
    )

    create_image(
        image,
    )

    result = predict_image(
        checkpoint_path=checkpoint,
        image_path=image,
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

    assert abs(
        sum(result["probabilities"].values())
        - 1.0
    ) < 1e-6