from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure class imbalance from a split manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="data/reports/class_imbalance.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    counts = df["label"].astype(str).value_counts().rename_axis("label").reset_index(name="count")
    counts["percentage"] = counts["count"] / counts["count"].sum() * 100
    counts["max_to_class_ratio"] = counts["count"].max() / counts["count"]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out, index=False)
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
