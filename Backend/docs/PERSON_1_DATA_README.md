# Person 1 — Data + Preprocessing (Locked Scope)

Person 1 owns dataset acquisition/organization, label understanding, image cleaning/quality checks, resize/normalization, train/validation/test splitting, class-distribution analysis, and dataset documentation.

## Handoff to Person 2
Create these files under `Backend/data/splits/`:
- `train.csv`
- `val.csv`
- `test.csv`

Each CSV must contain:
- `path` — path to an image, relative to the agreed `data-root`
- `label` — the accepted class name

Keep labels consistent across all three splits. Do not change labels inside Person 2's code.
