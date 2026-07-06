#!/usr/bin/env python3
"""grade.py - score a submission for the "Silent Degradation" task.

Metric: Average Precision (area under the precision-recall curve, computed
with sklearn.metrics.average_precision_score) of the submitted risk scores
against the private snapshot labels. Higher is better; a constant (or random)
submission scores approximately the positive rate of the evaluation set.

The metric is ranking-based: only the ordering of the scores matters, so any
finite real-valued scores are gradeable (values outside [0, 1] trigger a
warning but are scored). A submission is rejected with InvalidSubmissionError
only when it genuinely cannot be scored: unreadable file, missing columns,
missing/unknown/duplicated snapshot_ids, or non-numeric / NaN / infinite
predictions.

Usage:
    python grade.py <submission.csv> [--labels prepared/private/answers.csv]

Prints a JSON object: {"average_precision": <float>} and exits 0 on success.
Only pandas / numpy / scikit-learn are used (Kaggle Python Docker image).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

LABEL_COL = "failure_within_48h"
ID_COL = "snapshot_id"
PRED_COL = "prediction"


class InvalidSubmissionError(Exception):
    """Raised when a submission is malformed beyond scoring."""


def grade(submission, answers):
    """Return the Average Precision of a submission. Raises
    InvalidSubmissionError for truly invalid submissions.

    `submission` and `answers` may each be a pandas DataFrame (as passed by
    the platform's grade entrypoint) or a path to a CSV file (CLI use).
    """
    if isinstance(answers, pd.DataFrame):
        labels = answers.copy()
    else:
        labels = pd.read_csv(answers)

    if isinstance(submission, pd.DataFrame):
        sub = submission.copy()
    else:
        try:
            sub = pd.read_csv(submission)
        except Exception as exc:
            raise InvalidSubmissionError(
                f"Could not read submission as CSV: {exc}") from exc

    missing_cols = [c for c in (ID_COL, PRED_COL) if c not in sub.columns]
    if missing_cols:
        raise InvalidSubmissionError(
            f"Submission is missing required column(s) {missing_cols}; "
            f"expected columns: {ID_COL},{PRED_COL} (found: {list(sub.columns)})")

    sub = sub[[ID_COL, PRED_COL]].copy()
    sub[ID_COL] = sub[ID_COL].astype(str)

    dup = sub[ID_COL].duplicated()
    if dup.any():
        dupes = sorted(sub.loc[dup, ID_COL].unique())
        raise InvalidSubmissionError(
            f"Submission contains {dup.sum()} duplicated snapshot_id row(s), "
            f"e.g. {dupes[:5]}. Provide exactly one row per snapshot_id.")

    expected = set(labels[ID_COL].astype(str))
    got = set(sub[ID_COL])
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing or extra:
        raise InvalidSubmissionError(
            f"Submission snapshot_ids do not match the evaluation set: "
            f"{len(missing)} missing (e.g. {missing[:5]}), "
            f"{len(extra)} unknown (e.g. {extra[:5]}). "
            f"Provide exactly one row for each of the {len(expected)} "
            f"snapshot_ids in test_features.csv.")

    preds = pd.to_numeric(sub[PRED_COL], errors="coerce")
    bad = ~np.isfinite(preds.to_numpy(dtype=float, na_value=np.nan))
    if bad.any():
        bad_ids = sorted(sub.loc[bad, ID_COL])
        raise InvalidSubmissionError(
            f"Submission contains {int(bad.sum())} non-numeric, NaN or "
            f"infinite prediction(s), e.g. for {bad_ids[:5]}. "
            f"All predictions must be finite numbers.")

    if (preds < 0).any() or (preds > 1).any():
        print(f"WARNING: {int(((preds < 0) | (preds > 1)).sum())} prediction(s) "
              f"outside [0, 1]; the metric is ranking-based so they are scored "
              f"as-is.", file=sys.stderr)

    merged = labels.merge(sub, on=ID_COL, how="left")
    y_true = merged[LABEL_COL].to_numpy(dtype=int)
    y_score = pd.to_numeric(merged[PRED_COL]).to_numpy(dtype=float)

    return float(average_precision_score(y_true, y_score))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", help="path to submission CSV")
    parser.add_argument("--labels", default=os.path.join(
        here, "prepared", "private", "answers.csv"))
    args = parser.parse_args()

    try:
        score = grade(args.submission, args.labels)
    except InvalidSubmissionError as exc:
        print(json.dumps({"error": "invalid_submission", "message": str(exc)}))
        sys.exit(1)

    print(json.dumps({"average_precision": round(score, 6)}))


if __name__ == "__main__":
    main()
