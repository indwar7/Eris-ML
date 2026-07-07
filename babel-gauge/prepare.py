#!/usr/bin/env python3
"""prepare.py - split the raw telemetry lines into public/private data for
the "Babel Gauge: Legacy Telemetry Log Canonicalization" task.

Public  = 60,000 training pairs (raw_line -> canonical), 8,000 unlabeled
          test lines, the site/UTC-offset table, and a sample submission.
Private = the canonical target string for every test line.

The split is fixed in the raw file (the generator draws the two pools with
disjoint device-id ranges and different format mixes). This script only
selects columns; it is trivially deterministic - byte-identical on every
run. Only pandas is used (Kaggle Python Docker image).

Usage:
    python prepare.py [--raw dataset/raw] [--public prepared/public]
                      [--private prepared/private]
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw, public, private = Path(raw), Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    lines = pd.read_csv(raw / "telemetry_lines.csv")
    sites = pd.read_csv(raw / "sites.csv")

    train = lines.loc[lines["split"] == "train",
                      ["line_id", "raw_line", "canonical"]]
    test = lines.loc[lines["split"] == "test", ["line_id", "raw_line"]]
    answers = lines.loc[lines["split"] == "test", ["line_id", "canonical"]]

    train.to_csv(public / "train.csv", index=False)
    test.to_csv(public / "test.csv", index=False)
    sites.to_csv(public / "sites.csv", index=False)

    sample = test[["line_id"]].copy()
    sample["output"] = "REJECT"
    sample.to_csv(public / "sample_submission.csv", index=False)

    answers.to_csv(private / "answers.csv", index=False)

    print(f"public/train.csv          {len(train):,} rows")
    print(f"public/test.csv           {len(test):,} rows")
    print(f"public/sites.csv          {len(sites):,} rows")
    print(f"private/answers.csv       {len(answers):,} rows "
          f"({(answers['canonical'] == 'REJECT').mean():.2%} REJECT)")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=os.path.join(here, "dataset", "raw"))
    parser.add_argument("--public",
                        default=os.path.join(here, "prepared", "public"))
    parser.add_argument("--private",
                        default=os.path.join(here, "prepared", "private"))
    args = parser.parse_args()
    prepare(Path(args.raw), Path(args.public), Path(args.private))


if __name__ == "__main__":
    main()
