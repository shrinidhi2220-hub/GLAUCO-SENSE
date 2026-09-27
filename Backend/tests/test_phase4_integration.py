from pathlib import Path

import torch
from PIL import Image

from ml.inference import GlaucomaInference
from ml.inference_runner import run_prediction
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


def create_image(path: Path) -> None:
    Image.new(
        "RGB",
        (640, 480),
        (120, 100, 80),
    ).save(path)


def test_complete_phase4_pipeline(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    image = tmp_path / "fundus.jpg"
    output = tmp_path / "result.json"

    create_checkpoint(checkpoint)
    create_image(image)

    # Step 1: direct inference engine
    engine = GlaucomaInference(
        checkpoint_path=checkpoint,
        device="cpu",
    )

    direct_result = engine.predict(image)

    assert direct_result["prediction"] in {
        "glaucoma",
        "normal",
    }

    assert 0.0 <= direct_result["confidence"] <= 1.0

    assert set(
        direct_result["probabilities"].keys()
    ) == {
        "glaucoma",
        "normal",
    }

    assert abs(
        sum(direct_result["probabilities"].values()) - 1.0
    ) < 1e-6

    # Step 2: runner
    runner_result = run_prediction(
        checkpoint=checkpoint,
        image=image,
        output=output,
    )

    assert runner_result["prediction"] == (
        direct_result["prediction"]
    )

    assert abs(
        runner_result["confidence"]
        - direct_result["confidence"]
    ) < 1e-6

    assert output.exists()

    saved = output.read_text(
        encoding="utf-8"
    )

    assert '"prediction"' in saved
    assert '"confidence"' in saved
    assert '"probabilities"' in saved