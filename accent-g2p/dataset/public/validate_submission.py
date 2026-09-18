"""Offline structural checks for submission.csv. Needs no answers file.

Usage: python validate_submission.py submission.csv test.csv
"""
import sys

import pandas as pd


def validate(submission_path, test_path):
    sub = pd.read_csv(submission_path, keep_default_na=False)
    test = pd.read_csv(test_path, keep_default_na=False)
    problems = []
    for col in ("row_id", "phonemes"):
        if col not in sub.columns:
            problems.append(f"MISSING COLUMN '{col}' -- submission will be rejected")
    if problems:
        return problems
    if sub["row_id"].duplicated().any():
        problems.append(f"{int(sub['row_id'].duplicated().sum())} duplicate row_id (first kept)")
    missing = set(test["row_id"]) - set(sub["row_id"])
    if missing:
        problems.append(f"{len(missing)} test row_ids absent (each scores 0)")
    extra = set(sub["row_id"]) - set(test["row_id"])
    if extra:
        problems.append(f"{len(extra)} row_ids not in test.csv (ignored)")
    blank = sub["phonemes"].isna() | (sub["phonemes"].astype(str).str.strip() == "")
    if blank.any():
        problems.append(f"{int(blank.sum())} blank predictions (each scores 0)")
    m = test.merge(sub, on="row_id", how="inner")
    same = (m["phonemes"].astype(str) == m["source_pronunciation"].astype(str)).mean()
    if same > 0.9:
        problems.append("predictions are the source transcription copied through -- the chance floor; "
                        "88% of rows need a real change")
    per_src = m.groupby(["word", "src_accent"])["phonemes"].nunique()
    if (per_src == 1).mean() > 0.9:
        problems.append("same prediction regardless of target accent -- the tgt_accent input is unused")
    return problems


if __name__ == "__main__":
    issues = validate(sys.argv[1], sys.argv[2])
    print("OK -- submission is structurally valid" if not issues
          else "\n".join("WARN: " + i for i in issues))
