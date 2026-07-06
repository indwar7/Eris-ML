#!/usr/bin/env python3
"""grade.py - score a submission for the "Restock Radar: Next-Week Grocery
Purchase Ranking" task.

Metric: mean NDCG@20 over the 1,600 target users. For each user the
submission provides a ranked list of 20 item_ids (rank 1 = strongest
recommendation); relevance is binary - 1 if the user purchased that item
during the evaluation week, else 0:

    DCG@20  = sum_{r=1..20} rel_r / log2(r + 1)
    IDCG@20 = sum_{r=1..min(20, P)} 1 / log2(r + 1)   (P = user's positives)
    NDCG@20 = DCG@20 / IDCG@20,  averaged over all target users.

Higher is better; range [0, 1]. Item ids not in the catalog simply score
relevance 0. A submission is rejected with InvalidSubmissionError only when
it genuinely cannot be scored: unreadable file, missing columns, wrong user
set, missing/duplicate ranks, or duplicate items within a user.

Usage:
    python grade.py <submission.csv> [--answers prepared/private/answers.csv]

Prints a JSON object: {"ndcg_at_20": <float>} and exits 0 on success.
Only pandas / numpy are used (Kaggle Python Docker image).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

USER_COL = "user_id"
RANK_COL = "rank"
ITEM_COL = "item_id"
TOP_K = 20


class InvalidSubmissionError(Exception):
    """Raised when a submission is malformed beyond scoring."""


def grade(submission, answers):
    """Return mean NDCG@20 of a submission. Raises InvalidSubmissionError
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
            sub = pd.read_csv(submission)
        except Exception as exc:
            raise InvalidSubmissionError(
                f"Could not read submission as CSV: {exc}") from exc

    missing_cols = [c for c in (USER_COL, RANK_COL, ITEM_COL)
                    if c not in sub.columns]
    if missing_cols:
        raise InvalidSubmissionError(
            f"Submission is missing required column(s) {missing_cols}; "
            f"expected columns: {USER_COL},{RANK_COL},{ITEM_COL} "
            f"(found: {list(sub.columns)})")

    sub = sub[[USER_COL, RANK_COL, ITEM_COL]].copy()
    sub[USER_COL] = sub[USER_COL].astype(str)
    sub[ITEM_COL] = sub[ITEM_COL].astype(str)

    ranks = pd.to_numeric(sub[RANK_COL], errors="coerce")
    bad_rank = ranks.isna() | (ranks != ranks.round()) | \
        (ranks < 1) | (ranks > TOP_K)
    if bad_rank.any():
        bad = sub.loc[bad_rank, RANK_COL].astype(str).unique()[:5].tolist()
        raise InvalidSubmissionError(
            f"Submission contains {int(bad_rank.sum())} invalid rank "
            f"value(s) (e.g. {bad}); ranks must be integers 1..{TOP_K}.")
    sub[RANK_COL] = ranks.astype(int)

    expected_users = set(ans[USER_COL].astype(str))
    got_users = set(sub[USER_COL])
    missing_u = sorted(expected_users - got_users)
    extra_u = sorted(got_users - expected_users)
    if missing_u or extra_u:
        raise InvalidSubmissionError(
            f"Submission users do not match the evaluation set: "
            f"{len(missing_u)} missing (e.g. {missing_u[:5]}), "
            f"{len(extra_u)} unknown (e.g. {extra_u[:5]}). Provide ranked "
            f"lists for exactly the {len(expected_users)} user_ids in "
            f"test.csv.")

    counts = sub.groupby(USER_COL).size()
    wrong_n = counts[counts != TOP_K]
    if len(wrong_n):
        raise InvalidSubmissionError(
            f"{len(wrong_n)} user(s) do not have exactly {TOP_K} rows "
            f"(e.g. {dict(wrong_n.head(3))}). Provide exactly one row per "
            f"rank 1..{TOP_K} for every user.")

    dup_rank = sub.duplicated([USER_COL, RANK_COL])
    if dup_rank.any():
        ex = sub.loc[dup_rank, [USER_COL, RANK_COL]].head(3)
        raise InvalidSubmissionError(
            f"Submission contains {int(dup_rank.sum())} duplicated "
            f"(user_id, rank) pair(s), e.g.\n{ex.to_string(index=False)}. "
            f"Each user needs each rank 1..{TOP_K} exactly once.")

    dup_item = sub.duplicated([USER_COL, ITEM_COL])
    if dup_item.any():
        ex = sub.loc[dup_item, [USER_COL, ITEM_COL]].head(3)
        raise InvalidSubmissionError(
            f"Submission contains {int(dup_item.sum())} duplicated "
            f"(user_id, item_id) pair(s), e.g.\n{ex.to_string(index=False)}."
            f" An item may appear at most once per user.")

    ans = ans[[USER_COL, ITEM_COL]].astype(str).drop_duplicates()
    ans["rel"] = 1
    merged = sub.merge(ans, on=[USER_COL, ITEM_COL], how="left")
    merged["rel"] = merged["rel"].fillna(0).astype(int)

    merged["gain"] = merged["rel"] / np.log2(merged[RANK_COL] + 1.0)
    dcg = merged.groupby(USER_COL)["gain"].sum()

    pos = ans.groupby(USER_COL).size().clip(upper=TOP_K)
    disc = 1.0 / np.log2(np.arange(1, TOP_K + 1) + 1.0)
    cum = np.concatenate([[0.0], np.cumsum(disc)])
    idcg = pd.Series(cum[pos.to_numpy()], index=pos.index)

    ndcg = (dcg / idcg).reindex(sorted(expected_users)).fillna(0.0)
    return float(ndcg.mean())


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

    print(json.dumps({"ndcg_at_20": round(score, 6)}))


if __name__ == "__main__":
    main()
