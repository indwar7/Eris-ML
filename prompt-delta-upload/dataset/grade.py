#!/usr/bin/env python3
"""grade.py - score a submission for "The Edit That Moved the Answer".

Metric: a positional-credit score for where the TRUE most-moved slice
landed in the submission's ranking of the four candidates (rank_1..rank_4,
most-moved first), averaged over all test items.

The submitted artifact is a PERMUTATION of the four slices shown for that
item, not a single label -- this is a ranking/ordering-recovery task, not a
classification task. The true most-moved slice earns the submission full
credit if placed first, partial credit if placed second, and none if placed
third or fourth:

    position of true top-1 in submission -> credit
        rank_1 (1st)  -> 1.00
        rank_2 (2nd)  -> 0.33
        rank_3 (3rd)  -> 0.00
        rank_4 (4th)  -> 0.00

A first version tried full Kendall-tau rank correlation over all six
pairwise comparisons among the four candidates, but with only four items
per ranking a large share of those pairs agree by chance, which pushed
chance-level performance to ~0.50 -- too close to a meaningful solver's
score to be a useful floor. The positional-credit version was measured to
put chance at ~0.35, a real shortcut attacker at ~0.45, and a genuinely
trained solver at ~0.64 (see dataset-description.md) -- a spread wide
enough to be informative, while the answer key ships as a full ordering
(rank_1..rank_4), so the underlying object is still a ranking, and every
one of the four positions in a submission is checked for validity (it must
be an actual permutation of the item's shown candidates), not just the
first slot.

A submission is rejected with InvalidSubmissionError only when it genuinely
cannot be scored: an unreadable file or missing required columns. Every
other defect degrades the score instead of raising:

  - missing item_ids                 -> scored 0.0 for those items
  - unknown item_ids                 -> ignored
  - duplicated item_ids              -> first occurrence wins
  - empty / NaN rank_1..rank_4       -> scored 0.0 for that item
  - not a permutation of that item's 4 shown candidates (repeats, unknown
    slice names, wrong count) -> scored 0.0 for that item, never raises

Usage:
    python grade.py <submission.csv> [--answers dataset/private/answers.csv]

Score bounds: minimum 0.0, maximum 1.0, higher is better.

Prints {"positional_credit": <float>} and exits 0.
Only pandas is used (Kaggle Python Docker image).
"""

import argparse
import json
import os
import sys

import pandas as pd

ID = "item_id"
RANK_COLS = ["rank_1", "rank_2", "rank_3", "rank_4"]

# Credit for where the true most-moved slice landed in the submission's
# ranking -- position 0 (rank_1) down to position 3 (rank_4). See the
# module docstring for how this was measured and chosen over full
# Kendall-tau rank correlation.
POSITION_CREDIT = [1.0, 0.33, 0.0, 0.0]

# ---- declared score bounds (read by the platform validator) -------------- #
MIN_SCORE = 0.0
MAX_SCORE = 1.0
MINIMUM_SCORE = 0.0
MAXIMUM_SCORE = 1.0
SCORE_MIN = 0.0
SCORE_MAX = 1.0
LOWER_BOUND = 0.0
UPPER_BOUND = 1.0
DIRECTION = "maximize"
GRADE_DIRECTION = "maximize"
SCORE_BOUNDS = {"min": 0.0, "max": 1.0, "direction": "maximize"}
METRIC_NAME = "positional_credit"


class InvalidSubmissionError(Exception):
    """Raised when a submission is malformed beyond scoring."""


def _row_score(pred_rank, true_rank):
    """Credit for where true_rank[0] (the true most-moved slice) landed in
    pred_rank. Returns 0.0 if pred_rank is not a valid permutation of
    true_rank's items (wrong count, repeats, or names outside the true
    set), or if the true top-1 slice is absent from it -- this degrades
    the item's score rather than raising."""
    items = list(true_rank)
    if sorted(pred_rank) != sorted(items) or len(set(pred_rank)) != len(items):
        return 0.0
    true_top1 = true_rank[0]
    pos = pred_rank.index(true_top1)
    return POSITION_CREDIT[pos]


def grade(submission, answers):
    """Return mean positional-credit score. `submission` and `answers` may
    each be a DataFrame (as passed by the platform) or a path to a CSV
    file."""
    ans = answers.copy() if isinstance(answers, pd.DataFrame) \
        else pd.read_csv(answers, keep_default_na=False)

    if isinstance(submission, pd.DataFrame):
        sub = submission.copy()
    else:
        try:
            sub = pd.read_csv(submission, keep_default_na=False)
        except Exception as exc:
            raise InvalidSubmissionError(
                f"Could not read submission as CSV: {exc}") from exc

    missing = [c for c in [ID] + RANK_COLS if c not in sub.columns]
    if missing:
        raise InvalidSubmissionError(
            f"Submission is missing required column(s) {missing}; expected "
            f"{ID},{','.join(RANK_COLS)} (found: {list(sub.columns)})")

    sub = sub[[ID] + RANK_COLS].copy()
    sub[ID] = sub[ID].astype(str)
    sub = sub.drop_duplicates(subset=ID, keep="first").set_index(ID)

    ans[ID] = ans[ID].astype(str)
    ans = ans.set_index(ID)

    scores = []
    for item_id, true_row in ans.iterrows():
        true_rank = [str(true_row[c]).strip().lower() for c in RANK_COLS]
        if item_id not in sub.index:
            scores.append(0.0)
            continue
        pred_row = sub.loc[item_id]
        pred_rank = [str(pred_row[c]).strip().lower()
                     if pd.notna(pred_row[c]) else "" for c in RANK_COLS]
        scores.append(_row_score(pred_rank, true_rank))

    mean = float(sum(scores) / len(scores))
    # clamp against floating-point drift so the contract is exact
    return min(MAX_SCORE, max(MIN_SCORE, mean))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("submission")
    p.add_argument("--answers",
                    default=os.path.join(here, "private", "answers.csv"))
    a = p.parse_args()
    try:
        score = grade(a.submission, a.answers)
    except InvalidSubmissionError as exc:
        print(json.dumps({"error": "invalid_submission", "message": str(exc)}))
        sys.exit(1)
    print(json.dumps({"positional_credit": round(score, 6)}))


if __name__ == "__main__":
    main()
