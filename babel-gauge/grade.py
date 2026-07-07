#!/usr/bin/env python3
"""grade.py - score a submission for the "Babel Gauge: Legacy Telemetry Log
Canonicalization" task.

Metric: mean per-line field accuracy over the 8,000 test lines.

For a line whose ground truth is the literal string REJECT, the line scores
1.0 iff the submitted output (stripped of surrounding whitespace) is exactly
"REJECT", else 0.0.

For every other line the truth is a canonical record with exactly six
`key=value` fields joined by `|`:

    ts=...|dev=...|site=...|temp_c=...|pres_kpa=...|status=...

The submitted output is split on `|`; each piece is parsed as `key=value`
(first `=` only). The line's score is

    (# of the 6 truth fields whose value matches exactly)
    ------------------------------------------------------
            max(6, # of pieces in the submitted output)

so missing, wrong, malformed, and extra fields all cost. Submitting
"REJECT" for a clean line scores 0. The final score is the unweighted mean
over all lines; range [0, 1], higher is better.

Empty/missing outputs score 0 for their line. A submission is rejected with
InvalidSubmissionError only when it genuinely cannot be scored: unreadable
file, missing columns, or missing/unknown/duplicated line_ids.

Usage:
    python grade.py <submission.csv> [--answers prepared/private/answers.csv]

Prints a JSON object: {"mean_field_accuracy": <float>} and exits 0.
Only pandas / numpy are used (Kaggle Python Docker image).
"""

import argparse
import json
import os
import sys

import pandas as pd

ID_COL = "line_id"
OUT_COL = "output"
FIELDS = ["ts", "dev", "site", "temp_c", "pres_kpa", "status"]


class InvalidSubmissionError(Exception):
    """Raised when a submission is malformed beyond scoring."""


def _parse_canonical(s):
    """Split a canonical-format string into a key->value dict + piece count."""
    pieces = s.split("|")
    kv = {}
    for p in pieces:
        if "=" in p:
            k, v = p.split("=", 1)
            if k not in kv:            # first occurrence wins
                kv[k] = v
    return kv, len(pieces)


def _line_score(truth, out):
    out = "" if out is None else str(out).strip()
    if truth == "REJECT":
        return 1.0 if out == "REJECT" else 0.0
    if not out or out == "REJECT":
        return 0.0
    truth_kv, _ = _parse_canonical(truth)
    got_kv, n_pieces = _parse_canonical(out)
    correct = sum(1 for k in FIELDS if got_kv.get(k) == truth_kv[k])
    return correct / max(6, n_pieces)


def grade(submission, answers):
    """Return mean per-line field accuracy. Raises InvalidSubmissionError
    for truly invalid submissions.

    `submission` and `answers` may each be a pandas DataFrame (as passed by
    the platform's grade entrypoint) or a path to a CSV file (CLI use).
    """
    if isinstance(answers, pd.DataFrame):
        ans = answers.copy()
    else:
        ans = pd.read_csv(answers)

    if isinstance(submission, pd.DataFrame):
        sub = submission.copy()
    else:
        try:
            sub = pd.read_csv(submission, keep_default_na=False)
        except Exception as exc:
            raise InvalidSubmissionError(
                f"Could not read submission as CSV: {exc}") from exc

    missing_cols = [c for c in (ID_COL, OUT_COL) if c not in sub.columns]
    if missing_cols:
        raise InvalidSubmissionError(
            f"Submission is missing required column(s) {missing_cols}; "
            f"expected columns: {ID_COL},{OUT_COL} "
            f"(found: {list(sub.columns)})")

    sub = sub[[ID_COL, OUT_COL]].copy()
    sub[ID_COL] = sub[ID_COL].astype(str)

    dup = sub[ID_COL].duplicated()
    if dup.any():
        dupes = sorted(sub.loc[dup, ID_COL].unique())
        raise InvalidSubmissionError(
            f"Submission contains {int(dup.sum())} duplicated line_id "
            f"row(s), e.g. {dupes[:5]}. Provide exactly one row per line_id.")

    expected = set(ans[ID_COL].astype(str))
    got = set(sub[ID_COL])
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing or extra:
        raise InvalidSubmissionError(
            f"Submission line_ids do not match the test set: "
            f"{len(missing)} missing (e.g. {missing[:5]}), "
            f"{len(extra)} unknown (e.g. {extra[:5]}). Provide exactly one "
            f"row for each of the {len(expected)} line_ids in test.csv.")

    truth = dict(zip(ans[ID_COL].astype(str), ans["canonical"].astype(str)))
    scores = [_line_score(truth[lid], out)
              for lid, out in zip(sub[ID_COL], sub[OUT_COL])]
    return float(sum(scores) / len(scores))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", help="path to submission CSV")
    parser.add_argument("--answers", default=os.path.join(
        here, "prepared", "private", "answers.csv"))
    args = parser.parse_args()

    try:
        score = grade(args.submission, args.answers)
    except InvalidSubmissionError as exc:
        print(json.dumps({"error": "invalid_submission", "message": str(exc)}))
        sys.exit(1)

    print(json.dumps({"mean_field_accuracy": round(score, 6)}))


if __name__ == "__main__":
    main()
