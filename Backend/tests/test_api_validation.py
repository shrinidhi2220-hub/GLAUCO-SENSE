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


def create_valid_image_bytes() -> bytes:
    buffer = io.BytesIO()

    Image.new(
        "RGB",
        (640, 480),
        (100, 120, 140),
    ).save(
        buffer,
        format="JPEG",
    )

    return buffer.getvalue()


def test_valid_image_returns_200(
    tmp_path: Path,
):
    checkpoint = tmp_path / "model.pt"

    create_checkpoint(checkpoint)

    original_checkpoint = (
        app.state.checkpoint_path
    )

    app.state.checkpoint_path = checkpoint

    try:
        client = TestClient(app)

        response = client.post(
            "/predict",
            files={
                "file": (
                    "fundus.jpg",
                    create_valid_image_bytes(),
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

    finally:
        app.state.checkpoint_path = (
            original_checkpoint
        )


def test_non_image_file_returns_400(
    tmp_path: Path,
):
    checkpoint = tmp_path / "model.pt"

    create_checkpoint(checkpoint)

    original_checkpoint = (
        app.state.checkpoint_path
    )

    app.state.checkpoint_path = checkpoint

    try:
        client = TestClient(app)

        response = client.post(
            "/predict",
            files={
                "file": (
                    "notes.txt",
                    b"this is not an image",
                    "text/plain",
                )
            },
        )

        assert response.status_code == 400
        assert (
            response.json()["detail"]
            == "Please upload an image file."
        )

    finally:
        app.state.checkpoint_path = (
            original_checkpoint
        )


def test_empty_upload_returns_400(
    tmp_path: Path,
):
    checkpoint = tmp_path / "model.pt"

    create_checkpoint(checkpoint)

    original_checkpoint = (
        app.state.checkpoint_path
    )

    app.state.checkpoint_path = checkpoint

    try:
        client = TestClient(app)

        response = client.post(
            "/predict",
            files={
                "file": (
                    "empty.jpg",
                    b"",
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 400
        assert (
            response.json()["detail"]
            == "The uploaded image is empty."
        )

    finally:
        app.state.checkpoint_path = (
            original_checkpoint
        )


def test_corrupt_image_returns_400(
    tmp_path: Path,
):
    checkpoint = tmp_path / "model.pt"

    create_checkpoint(checkpoint)

    original_checkpoint = (
        app.state.checkpoint_path
    )

    app.state.checkpoint_path = checkpoint

    try:
        client = TestClient(app)

        response = client.post(
            "/predict",
            files={
                "file": (
                    "broken.jpg",
                    b"not-a-real-jpeg",
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 400

        assert (
            response.json()["detail"]
            == "The uploaded file is not a valid image."
        )

    finally:
        app.state.checkpoint_path = (
            original_checkpoint
        )


def test_missing_model_returns_503(
    tmp_path: Path,
):
    missing_checkpoint = (
        tmp_path / "does_not_exist.pt"
    )

    original_checkpoint = (
        app.state.checkpoint_path
    )

    app.state.checkpoint_path = (
        missing_checkpoint
    )

    try:
        client = TestClient(app)

        response = client.post(
            "/predict",
            files={
                "file": (
                    "fundus.jpg",
                    create_valid_image_bytes(),
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 503

        assert (
            "No trained model checkpoint"
            in response.json()["detail"]
        )

    finally:
        app.state.checkpoint_path = (
            original_checkpoint
        )