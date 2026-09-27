from pathlib import Path

import torch
from PIL import Image

from ml.inference_runner import run_prediction
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
        (100, 120, 140),
    )

    image.save(path)


def test_run_prediction(
    tmp_path: Path,
):
    checkpoint = (
        tmp_path / "model.pt"
    )

    image = (
        tmp_path / "image.jpg"
    )

    output = (
        tmp_path / "result.json"
    )

    create_test_checkpoint(
        checkpoint
    )

    create_test_image(
        image
    )

    result = run_prediction(
        checkpoint=checkpoint,
        image=image,
        output=output,
    )

    assert result["prediction"] in {
        "glaucoma",
        "normal",
    }

    assert 0.0 <= result[
        "confidence"
    ] <= 1.0

    assert output.exists()

    saved = output.read_text(
        encoding="utf-8"
    )

    assert '"prediction"' in saved
    assert '"confidence"' in saved
    assert '"probabilities"' in saved


def test_run_prediction_without_output(
    tmp_path: Path,
):
    checkpoint = (
        tmp_path / "model.pt"
    )

    image = (
        tmp_path / "image.jpg"
    )

    create_test_checkpoint(
        checkpoint
    )

    create_test_image(
        image
    )

    result = run_prediction(
        checkpoint=checkpoint,
        image=image,
    )

    assert "prediction" in result
    assert "confidence" in result
    assert "probabilities" in result