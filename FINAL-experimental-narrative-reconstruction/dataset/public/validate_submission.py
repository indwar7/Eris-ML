"""Offline structural checks for submission.csv. Needs no answers file.

Usage: python validate_submission.py submission.csv test.csv
"""
import sys

import pandas as pd


def validate(submission_path, test_path):
    sub = pd.read_csv(submission_path)
    test = pd.read_csv(test_path)
    problems = []

    for col in ("paper_id", "predicted_order"):
        if col not in sub.columns:
            problems.append(f"MISSING COLUMN '{col}' -- submission will be rejected")
    if problems:
        return problems

    if sub["paper_id"].duplicated().any():
        problems.append(f"{int(sub['paper_id'].duplicated().sum())} duplicate paper_id "
                        "(only the first is kept)")

    slots = {pid: "ABCDE"[:int(n)] for pid, n in zip(test["paper_id"], test["n_slots"])}
    missing = set(slots) - set(sub["paper_id"])
    if missing:
        problems.append(f"{len(missing)} test papers absent (each scores 0.0)")

    extra = set(sub["paper_id"]) - set(slots)
    if extra:
        problems.append(f"{len(extra)} paper_ids not in test.csv (ignored)")

    bad = []
    for pid, po in zip(sub["paper_id"], sub["predicted_order"]):
        if pid not in slots:
            continue
        p = "" if pd.isna(po) else str(po).strip().upper()
        if "".join(sorted(p)) != slots[pid]:
            bad.append((pid, po))
    if bad:
        problems.append(f"{len(bad)} predictions are not a permutation of that paper's "
                        f"slots (e.g. {bad[0]}) -- each scores 0.0")

    ident = sum(1 for pid, po in zip(sub["paper_id"], sub["predicted_order"])
                if pid in slots and str(po).strip().upper() == slots[pid])
    if ident == len(sub):
        problems.append("every prediction is the shipped (shuffled) order -- scores chance (~0.22)")

    return problems


if __name__ == "__main__":
    issues = validate(sys.argv[1], sys.argv[2])
    print("OK -- submission is structurally valid" if not issues
          else "\n".join("WARN: " + i for i in issues))
