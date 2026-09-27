from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.ml.api_service import predict_image


router = APIRouter()


class PredictionResponse(BaseModel):
    prediction: str
    class_index: int
    confidence: float
    probabilities: dict[str, float]


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    request: Request,
    file: UploadFile = File(...),
) -> PredictionResponse:
    """Run single-image ML prediction."""

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="File content type is missing.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are accepted.",
        )

    checkpoint_path = Path(
        request.app.state.checkpoint_path
    )

    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Inference model checkpoint is not available.",
        )

    suffix = Path(
        file.filename or "upload.jpg"
    ).suffix or ".jpg"

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name
            )

            content = await file.read()

            if not content:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded image is empty.",
                )

            temporary_file.write(content)

        result = predict_image(
            checkpoint_path=checkpoint_path,
            image_path=temporary_path,
        )

        return PredictionResponse(
            **result
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Unable to process image: {exc}",
        ) from exc

    finally:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True
            )