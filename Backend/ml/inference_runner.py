from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.inference import GlaucomaInference


def run_prediction(
    checkpoint: str | Path,
    image: str | Path,
    output: str | Path | None = None,
) -> dict:
    """
    Run single-image inference and optionally save the result.
    """

    engine = GlaucomaInference(
        checkpoint_path=checkpoint,
    )

    result = engine.predict(image)

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                {
                    "image": str(image),
                    **result,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GLAUCO-SENSE single-image inference runner."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a trained model checkpoint.",
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the image to classify.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )

    args = parser.parse_args()

    result = run_prediction(
        checkpoint=args.checkpoint,
        image=args.image,
        output=args.output,
    )

    print("Inference completed.")
    print(
        f"Prediction: {result['prediction']}"
    )
    print(
        f"Confidence: {result['confidence']:.4f}"
    )

    print("Probabilities:")

    for class_name, probability in (
        result["probabilities"].items()
    ):
        print(
            f"  {class_name}: {probability:.4f}"
        )

    if args.output is not None:
        print(
            f"Result saved to: {args.output}"
        )


if __name__ == "__main__":
    main()