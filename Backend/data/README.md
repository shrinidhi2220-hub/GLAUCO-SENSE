# Person 1 — Data + Preprocessing

This folder is the working area for Person 1.

## Responsibilities
- Download/organize the selected glaucoma dataset.
- Record source, license/usage terms and label definitions.
- Validate and clean images.
- Detect unreadable/corrupt files and exact duplicates.
- Resize images consistently.
- Create train/validation/test manifests.
- Measure class imbalance and document the chosen mitigation.

## Folder layout
- `raw/` — original dataset files; do not modify.
- `cleaned/` — validated/cleaned copies when needed.
- `processed/` — optional processed images produced by preprocessing.
- `splits/` — CSV manifests used by Person 2.
- `reports/` — dataset reports and class-balance summaries.

## Expected class-folder input
The preprocessing scripts accept a class-folder dataset such as:

```text
raw/glaucoma_dataset/
├── glaucoma/
│   ├── image001.jpg
│   └── ...
└── normal/
    ├── image002.jpg
    └── ...
```

The actual class names must match the selected dataset documentation. Do not assume a label means glaucoma until the dataset documentation is verified.
