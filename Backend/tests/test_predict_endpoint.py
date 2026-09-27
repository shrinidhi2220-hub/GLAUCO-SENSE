from pathlib import Path

import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from ml.model import build_baseline_model


def create_checkpoint(path: Path) -> None:
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


def create_image_bytes() -> bytes:
    image_path = Path("tests") / "temporary_api_image.jpg"

    Image.new(
        "RGB",
        (640, 480),
        (100, 120, 140),
    ).save(image_path)

    data = image_path.read_bytes()

    image_path.unlink(
        missing_ok=True
    )

    return data


def test_predict_endpoint(tmp_path: Path):
    checkpoint = (
        tmp_path / "model.pt"
    )

    create_checkpoint(
        checkpoint
    )

    original_checkpoint = (
        app.state.checkpoint_path
    )

    app.state.checkpoint_path = checkpoint

    try:
        client = TestClient(app)

        image_bytes = (
            create_image_bytes()
        )

        response = client.post(
            "/predict",
            files={
                "file": (
                    "fundus.jpg",
                    image_bytes,
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert body["prediction"] in {
            "glaucoma",
            "normal",
        }

        assert body["class_index"] in {
            0,
            1,
        }

        assert 0.0 <= body[
            "confidence"
        ] <= 1.0

        assert set(
            body["probabilities"].keys()
        ) == {
            "glaucoma",
            "normal",
        }

        assert abs(
            sum(body["probabilities"].values())
            - 1.0
        ) < 1e-6

    finally:
        app.state.checkpoint_path = (
            original_checkpoint
        )