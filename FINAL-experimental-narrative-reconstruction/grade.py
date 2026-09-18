"""
grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float

Metric: mean per-paper POSITION ACCURACY -- for each paper, the fraction of
its subsections placed at exactly their true position; averaged over papers.
1.0 = every subsection of every paper in its authors' position.

Why position accuracy (v2). The first version scored rescaled Kendall's tau.
Tau is a fine ordering metric, but its chance level is 0.5 by construction,
which compresses the usable range: a solver that has recovered nothing scores
0.5, and platform agent runs on the tau grader landed at 0.56 / 0.82 / 0.90
-- the top of the scale, with the bottom half of [0, 1] unreachable by any
submission. Position accuracy keeps the task and data identical and spreads
the same submissions over the scale: its chance level is 1/n per paper (0.22
averaged over the shipped test split, where n is 4 or 5), the TF-IDF
reference that scored 0.658 tau scores 0.358 here, and a near-perfect
ordering still scores near 1.0. It remains graded credit -- getting three of
five experiments in place earns 0.6 -- rather than the all-or-nothing of
exact match (1-in-120 by chance on five items), and it is the standard "Acc"
of the sentence-ordering literature alongside PMR. What it deliberately does
NOT do is credit a globally shifted order: if every experiment is one step
late, the arc is right but the positions are all wrong, and the paper scores
0. That is stricter than tau, and it is the point.

Degradation, not rejection: a paper missing from the submission, or one whose
predicted_order is not a permutation of that paper's slots, scores 0.0 for
that paper (no position can be verified) rather than voiding the run. Only a
submission missing required columns is rejected outright.

Slice-safety: only paper_ids present in `answers` are scored; extras ignored.
"""
import pandas as pd

REQUIRED_SUBMISSION_COLUMNS = {"paper_id", "predicted_order"}
REQUIRED_ANSWER_COLUMNS = {"paper_id", "true_order"}


class InvalidSubmission(ValueError):
    pass


def _position_accuracy(pred: str, true: str) -> float:
    """Fraction of slots at exactly their true position. pred and true are
    permutations of the same slot letters."""
    n = len(true)
    if n == 0:
        return 0.0
    return sum(1 for a, b in zip(pred, true) if a == b) / n


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    if not REQUIRED_ANSWER_COLUMNS.issubset(answers.columns):
        raise InvalidSubmission(f"answers missing required columns: {REQUIRED_ANSWER_COLUMNS}")
    if submission is None or len(submission) == 0:
        submission = pd.DataFrame(columns=sorted(REQUIRED_SUBMISSION_COLUMNS))
    missing = REQUIRED_SUBMISSION_COLUMNS - set(submission.columns)
    if missing:
        raise InvalidSubmission(f"submission missing required columns: {missing}")

    submission = submission.drop_duplicates(subset=["paper_id"], keep="first")
    pred = dict(zip(submission["paper_id"].astype(str),
                    submission["predicted_order"].astype(str).str.strip().str.upper()))

    scores = []
    for row in answers.itertuples(index=False):
        true = str(row.true_order).strip().upper()
        p = pred.get(str(row.paper_id))
        # must be a permutation of exactly the paper's slots; otherwise nothing
        # can be verified and the paper scores 0
        if p is None or p in ("", "NAN") or sorted(p) != sorted(true):
            scores.append(0.0)
            continue
        scores.append(_position_accuracy(p, true))

    return float(sum(scores) / len(scores)) if scores else 0.0


if __name__ == "__main__":
    import sys
    print(grade(pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])))
