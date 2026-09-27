from __future__ import annotations

from pathlib import Path

from ml.inference import GlaucomaInference


def predict_image(
    checkpoint_path: str | Path,
    image_path: str | Path,
) -> dict:
    """
    Application-level prediction service.

    This is independent of Person 1's dataset.
    """

    engine = GlaucomaInference(
        checkpoint_path=checkpoint_path,
    )

    result = engine.predict(
        image_path,
    )

    return {
        "prediction": result["prediction"],
        "class_index": result["class_index"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
    }