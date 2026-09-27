# Person 2 — ML + Evaluation (Locked Scope)

## Responsibility
Person 2 owns the baseline glaucoma ML model, training, evaluation, error analysis, model saving, and single-image inference.

## What can be built before Person 1 finishes
- ResNet18 baseline architecture
- Data loader contract for `train.csv`, `val.csv`, `test.csv`
- Class-weighted loss implementation
- Training loop, validation loop, scheduler, early stopping
- Accuracy / precision / recall / F1 calculations
- Sensitivity / specificity / PPV / NPV for binary classification
- Confusion-matrix generation
- Incorrect-prediction analysis
- Model checkpoint + metadata saving
- Single-image prediction CLI
- FastAPI prediction endpoint

## Dependency on Person 1
The final real training run starts only after Person 1 supplies the accepted dataset splits. The expected manifest format is:

```csv
path,label
relative/path/to/image1.jpg,normal
relative/path/to/image2.jpg,glaucoma
```

Required files:
- `Backend/data/splits/train.csv`
- `Backend/data/splits/val.csv`
- `Backend/data/splits/test.csv`

`path` is resolved relative to `--data-root` unless it is absolute.

## Baseline model
ResNet18 with ImageNet initialization and a task-specific final classifier layer. Training uses class-weighted cross-entropy based only on the training split.

## Outputs
Training creates:
- `models/glaucoma_resnet18.pt`
- `models/glaucoma_resnet18.json`
- `reports/training_history.csv`

Evaluation creates:
- `reports/evaluation/metrics.json`
- `reports/evaluation/classification_report.txt`
- `reports/evaluation/confusion_matrix.csv`
- `reports/evaluation/confusion_matrix.png`
- `reports/evaluation/all_predictions.csv`
- `reports/evaluation/incorrect_predictions.csv`

## Research boundary
The model output is a research-project prediction only and must not be presented as a medical diagnosis.
