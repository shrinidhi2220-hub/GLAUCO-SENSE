# GLAUCO-SENSE

Glaucoma research prototype with a separated frontend and backend.

## Project structure

```text
GLAUCO-SENSE-main/
├── Frontend/                  # UI
├── Backend/
│   ├── data/                  # Person 1: dataset + preprocessing artifacts
│   ├── scripts/               # Person 1: dataset/report/split scripts
│   ├── ml/                    # Person 2: training + evaluation
│   ├── models/                # saved model checkpoints
│   ├── reports/               # experiment/evaluation reports
│   ├── app/                   # FastAPI + inference
│   ├── multimodal/            # shared final integration
│   └── docs/                  # team documentation
├── .gitignore
└── README.md
```

## Team split

### Person 1
Data + preprocessing: dataset organization, label verification, cleaning, resizing/normalization, train/validation/test split, class-balance documentation.

### Person 2
ML + evaluation: baseline model, training, class-imbalance mitigation, metrics, sensitivity/specificity, confusion matrix, incorrect-prediction analysis, model saving and inference.

### Both
Multimodal model and final integration.

## Run
See `Backend/README.md`.
