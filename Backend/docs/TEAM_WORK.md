# GLAUCO-SENSE Team Work — Locked

## Person 1 — Data + Preprocessing
- Download/organize the glaucoma dataset
- Understand labels
- Clean/check images
- Resize/normalize images
- Create train/validation/test splits
- Handle class imbalance at the dataset-analysis/preprocessing stage
- Document the dataset

## Person 2 — ML + Evaluation
- Build the baseline model
- Train it
- Evaluate accuracy, precision, recall, F1, sensitivity/specificity
- Generate confusion matrix
- Analyze incorrect predictions
- Save the trained model

## Shared — Later
- Multimodal model
- Final integration

## Fixed handoff contract
Person 1 supplies `train.csv`, `val.csv`, and `test.csv` under `Backend/data/splits/`, each with `path,label` columns. Person 2 consumes those manifests and does not relabel or replace the dataset.
