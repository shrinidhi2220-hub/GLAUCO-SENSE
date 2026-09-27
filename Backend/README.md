# GLAUCO-SENSE Backend

This backend is intentionally split into two team responsibilities.

## Person 1 — Data + preprocessing
Use `scripts/` and `data/` to:
- organize and document the dataset
- validate/clean images
- preprocess images when needed
- create reproducible train/validation/test manifests
- measure class imbalance

## Person 2 — ML + evaluation
Use `ml/` to:
- train the ResNet18 baseline
- handle class imbalance with training-set class weights
- save the best model
- evaluate on the held-out test split
- generate metrics, confusion matrix and incorrect-prediction analysis
- provide inference for the API

## Typical workflow
From `Backend/` with the virtual environment active:

```powershell
# Person 1
python scripts/dataset_report.py --input data/raw/glaucoma_dataset --output data/reports
python scripts/prepare_dataset.py --input data/raw/glaucoma_dataset --output data/processed
python scripts/make_splits.py --input data/processed --output data/splits
python scripts/class_imbalance_report.py --manifest data/splits/train.csv --output data/reports/class_imbalance.csv

# Person 2
python ml/train_baseline.py --splits data/splits --data-root data/processed --output models/glaucoma_resnet18.pt --history reports/training_history.csv
python ml/evaluate.py --model models/glaucoma_resnet18.pt --manifest data/splits/test.csv --data-root data/processed --report-dir reports/evaluation

# API
python -m uvicorn app.main:app --reload
```

## Output files from Person 2
- `models/glaucoma_resnet18.pt`
- `models/glaucoma_resnet18.json`
- `reports/training_history.csv`
- `reports/evaluation/metrics.json`
- `reports/evaluation/classification_report.txt`
- `reports/evaluation/confusion_matrix.csv`
- `reports/evaluation/confusion_matrix.png`
- `reports/evaluation/all_predictions.csv`
- `reports/evaluation/incorrect_predictions.csv`

## Medical use disclaimer
This is a research prototype. Model output must not be presented as a medical diagnosis.
