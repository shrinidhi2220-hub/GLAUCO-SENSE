from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torchvision import models, transforms

try:
    from .metrics import classification_metrics
except ImportError:
    from metrics import classification_metrics

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    classes = [str(x) for x in checkpoint["classes"]]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes


def make_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def save_confusion_matrix(cm: np.ndarray, classes: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="GLAUCO-SENSE — Confusion Matrix",
    )
    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved baseline model on the held-out test split.")
    parser.add_argument("--model", default="models/glaucoma_resnet18.pt")
    parser.add_argument("--manifest", default="data/splits/test.csv")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--report-dir", default="reports/evaluation")
    args = parser.parse_args()

    model, classes = load_model(Path(args.model))
    class_to_id = {label: i for i, label in enumerate(classes)}
    transform = make_transform()
    data_root = Path(args.data_root)
    df = pd.read_csv(args.manifest)
    required = {"path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Test manifest missing columns: {sorted(missing)}")
    report = Path(args.report_dir)
    report.mkdir(parents=True, exist_ok=True)

    if not set(df["label"].astype(str)).issubset(classes):
        raise SystemExit("Test manifest contains labels not present in the saved model.")

    y_true: list[int] = []
    y_pred: list[int] = []
    confidence: list[float] = []
    paths: list[str] = []
    probabilities: list[list[float]] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for _, row in df.iterrows():
            raw_path = Path(str(row["path"]))
            path = raw_path if raw_path.is_absolute() else data_root / raw_path
            try:
                image = Image.open(path).convert("RGB")
            except (OSError, UnidentifiedImageError) as exc:
                raise RuntimeError(f"Unable to read test image: {path}") from exc
            tensor = transform(image).unsqueeze(0).to(device)
            probs = torch.softmax(model(tensor), dim=1)[0]
            pred = int(torch.argmax(probs).item())
            y_pred.append(pred)
            y_true.append(class_to_id[str(row["label"])])
            confidence.append(float(probs[pred].item()))
            probabilities.append([float(x) for x in probs.detach().cpu().tolist()])
            paths.append(str(row["path"]))

    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = classification_metrics(y_true, y_pred, classes)
    metrics.update(
        {
            "classes": classes,
            "test_samples": len(df),
            "incorrect_samples": int(sum(a != b for a, b in zip(y_true, y_pred))),
            "model": str(Path(args.model)),
        }
    )

    error_df = pd.DataFrame(
        {
            "path": paths,
            "true_label": [classes[i] for i in y_true],
            "predicted_label": [classes[i] for i in y_pred],
            "confidence": confidence,
        }
    )
    for idx, class_name in enumerate(classes):
        error_df[f"probability_{class_name}"] = [p[idx] for p in probabilities]

    incorrect = error_df[error_df["true_label"] != error_df["predicted_label"]].sort_values("confidence")
    incorrect.to_csv(report / "incorrect_predictions.csv", index=False)
    error_df.to_csv(report / "all_predictions.csv", index=False)

    (report / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (report / "classification_report.txt").write_text(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=classes,
            zero_division=0,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(report / "confusion_matrix.csv")
    save_confusion_matrix(cm, classes, report / "confusion_matrix.png")

    print(json.dumps(metrics, indent=2))
    print(f"Reports saved to: {report.resolve()}")


if __name__ == "__main__":
    main()
