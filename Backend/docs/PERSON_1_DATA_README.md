# Person 1 — Data + Preprocessing Handoff

## Scope

Person 1 owns the complete dataset preparation pipeline for GLAUCO-SENSE:

1. Dataset acquisition and organization
2. Label and metadata verification
3. Image quality validation
4. Duplicate detection and resolution
5. Image preprocessing
6. Patient-level train/validation/test splitting
7. Class-imbalance analysis
8. Dataset documentation and handoff

---

# Dataset

## Source

Dataset: Hillel-Yaffe Glaucoma Dataset (HYGD)

Raw dataset location:

```text
Backend/data/raw/HYGD/
├── Images/
└── Labels.csv