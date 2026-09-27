from __future__ import annotations

import io
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
    buffer = io.BytesIO()

    Image.new(
        "RGB",
        (640, 480),
        (120, 100, 80),
    ).save(
        buffer,
        format="JPEG",
    )

    return buffer.getvalue()


def test_complete_phase5_api_pipeline(
    tmp_path: Path,
):
    checkpoint = tmp_path / "model.pt"

    create_checkpoint(checkpoint)

    original_checkpoint = app.state.checkpoint_path
    app.state.checkpoint_path = checkpoint

    try:
        client = TestClient(app)

        # 1. Health check
        health = client.get("/health")

        assert health.status_code == 200

        health_body = health.json()

        assert health_body["status"] == "ok"
        assert health_body["model_available"] is True
        assert health_body["research_only"] is True

        # 2. Valid prediction request
        response = client.post(
            "/predict",
            files={
                "file": (
                    "fundus.jpg",
                    create_image_bytes(),
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert body["filename"] == "fundus.jpg"

        assert body["prediction"] in {
            "glaucoma",
            "normal",
        }

        assert body["class_index"] in {
            0,
            1,
        }

        assert 0.0 <= body["confidence"] <= 1.0

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

        assert body["research_only"] is True

        assert (
            "not a medical diagnosis"
            in body["note"]
        )

        # 3. Verify the alias endpoint also works
        alias_response = client.post(
            "/api/predict",
            files={
                "file": (
                    "fundus.jpg",
                    create_image_bytes(),
                    "image/jpeg",
                )
            },
        )

        assert alias_response.status_code == 200

        alias_body = alias_response.json()

        assert alias_body["prediction"] in {
            "glaucoma",
            "normal",
        }

    finally:
        app.state.checkpoint_path = (
            original_checkpoint
        )