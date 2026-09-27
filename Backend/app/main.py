from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.ml.api_service import predict_image


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = Path(
    os.getenv(
        "GLAUCO_MODEL_PATH",
        BASE_DIR
        / "models"
        / "checkpoints"
        / "baseline_smoke_test.pt",
    )
)


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="GLAUCO-SENSE Backend",
    version="0.1.0",
    description=(
        "GLAUCO-SENSE research backend for "
        "glaucoma image classification."
    ),
)

# IMPORTANT:
# Store the active checkpoint on FastAPI state.
# This allows tests and later configuration to replace it.
app.state.checkpoint_path = MODEL_PATH


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/health")
def health():
    checkpoint_path = Path(
        app.state.checkpoint_path
    )

    return {
        "status": "ok",
        "model_available": checkpoint_path.exists(),
        "model_path": str(checkpoint_path),
        "research_only": True,
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
):
    """
    Upload an image and run the current ML inference pipeline.

    Supported endpoints:
        POST /predict
        POST /api/predict
    """

    # Validate content type
    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    checkpoint_path = Path(
        app.state.checkpoint_path
    )

    # Check model availability
    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model checkpoint was found. "
                f"Expected: {checkpoint_path}"
            ),
        )

    # Read uploaded file
    raw = await file.read()

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    # Validate image
    try:
        image = Image.open(
            io.BytesIO(raw)
        )

        # Force complete image verification.
        image.verify()

    except (
        UnidentifiedImageError,
        OSError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file is not "
                "a valid image."
            ),
        ) from exc

    temporary_path: Path | None = None

    try:
        suffix = (
            Path(file.filename).suffix
            if file.filename
            else ".jpg"
        )

        if not suffix:
            suffix = ".jpg"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temporary_file:

            temporary_path = Path(
                temporary_file.name
            )

            temporary_file.write(raw)

        # Use the active checkpoint from app state.
        result = predict_image(
            checkpoint_path=checkpoint_path,
            image_path=temporary_path,
        )

        return {
    "filename": file.filename,
    **result,
    "research_only": True,
    "note": (
        "This output is a research-model "
        "prediction and is not a medical diagnosis."
    ),
}

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "An error occurred while running "
                f"the inference pipeline: {exc}"
            ),
        ) from exc

    finally:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True
            )