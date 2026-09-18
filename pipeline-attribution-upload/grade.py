"""
grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float

Metric: mean per-bag Adjusted Rand Index (ARI), clipped to [0, 1].

SHAPE (v3): ONE ROW PER BAG in both submission.csv and answers.csv --
    bag_id, batch_id
where `batch_id` is a whitespace-separated list of group labels, one per
snippet of that bag, in the snippet order given by test.csv (the integer
after "::" in each test row_id). Labels are arbitrary tokens; only
co-membership within a bag is scored.

Why one row per bag (review round 2): the leaderboard splits answers.csv
into public and private slices BY ROW. With one row per snippet, every bag
was cut in two and each fragment was scored as a bag of its own; 24 public
fragments had three rows or fewer, where ARI hands out 1.0 almost for free
(an all-singleton guess on a 3-row fragment whose rows happen to come from
three batches is a perfect partition). The all-singleton sample scored 0.048
on the public board and 0.000 on private, and one solver run was exactly
that. With one row per bag, any row-wise slice keeps every bag whole, so a
bag is always scored against its full 19-37 snippets and its full 3-4 true
groups, and the degenerate partitions score exactly 0 on every slice.

Why ARI, not raw pairwise F1: an earlier version of this grader used raw
pairwise co-membership F1. Measurement caught a problem with that choice:
raw pairwise F1 structurally rewards "lump everything into one group" -- it
gets perfect recall on every true same-batch pair for free, and only loses
on precision, which for bags with a handful of groups isn't enough of a
penalty. Measured: "everyone in one group" scored 0.376 under raw pairwise
F1, comfortably beating a properly trained embedding-based reference's
0.289. Adjusted Rand Index specifically corrects for this: it is
chance-normalized, so both degenerate strategies (one giant group, or every
snippet its own singleton) score ~0 by construction, and only genuine
above-chance agreement with the true partition earns credit. No group-label
alignment step is needed (ARI is computed directly from the two label
assignments).

Rejected outright (raise InvalidSubmission, a ValueError): a required column
missing. Nothing else raises.

Degradation, not rejection, for everything else -- a bag scores 0 when its
prediction cannot be trusted or matched, and the rest is still graded:
  - a bag missing from the submission;
  - a bag_id that appears MORE THAN ONCE (review round 2: the previous grader
    silently kept the first duplicate, which let a repeated id score as if
    correct -- now the conflicting rows are all discarded and the bag scores
    0; the grader will not choose between them);
  - a `batch_id` value that is empty, NaN, or whose token count does not match
    the bag's snippet count (no snippet can be matched to a prediction).
A bag_id that matches no bag in the answers (malformed, or from another
slice) is ignored; it cannot crash the grader.

Slice-safety: only bag_ids present in `answers` are scored; extra bag_ids in
the submission are ignored.
"""
import re

import pandas as pd
from sklearn.metrics import adjusted_rand_score

REQUIRED_SUBMISSION_COLUMNS = {"bag_id", "batch_id"}
REQUIRED_ANSWER_COLUMNS = {"bag_id", "batch_id"}

_SPLIT = re.compile(r"[\s,;]+")


class InvalidSubmission(ValueError):
    pass


def _tokens(value) -> list:
    """Split a `batch_id` cell into labels. Empty / NaN -> []."""
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return [t for t in _SPLIT.split(s) if t] if s else []


def _bag_ari(true_labels: list, pred_labels: list) -> float:
    if len(set(true_labels)) < 2:
        # a bag with a single true group can't be scored discriminatively;
        # shouldn't occur by construction (min 3 groups/bag) but degrade
        # gracefully rather than raise if it ever does
        return 1.0 if len(set(pred_labels)) == 1 else 0.0
    return max(0.0, float(adjusted_rand_score(true_labels, pred_labels)))


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    if answers is None or not REQUIRED_ANSWER_COLUMNS.issubset(answers.columns):
        raise InvalidSubmission(f"answers missing required columns: {REQUIRED_ANSWER_COLUMNS}")
    if submission is None or len(submission) == 0:
        submission = pd.DataFrame(columns=sorted(REQUIRED_SUBMISSION_COLUMNS))
    missing_cols = REQUIRED_SUBMISSION_COLUMNS - set(submission.columns)
    if missing_cols:
        raise InvalidSubmission(f"submission missing required columns: {sorted(missing_cols)}")

    sub_ids = submission["bag_id"].astype(str).str.strip()
    dup_ids = set(sub_ids[sub_ids.duplicated()])
    # a repeated bag_id is a conflict, not a choice: every row for it is
    # dropped, so that bag scores 0 below rather than "first one wins"
    pred = {b: g for b, g in zip(sub_ids, submission["batch_id"]) if b not in dup_ids}

    true_by_bag = {}
    for row in answers.itertuples(index=False):
        true_by_bag[str(row.bag_id).strip()] = _tokens(row.batch_id)

    scores = []
    for bag_id in sorted(true_by_bag):
        true_labels = true_by_bag[bag_id]
        if len(true_labels) < 2:
            continue
        pred_labels = _tokens(pred.get(bag_id))
        if len(pred_labels) != len(true_labels):
            # missing bag, empty cell, or wrong token count: nothing can be
            # matched to a snippet, so this bag scores 0
            scores.append(0.0)
            continue
        scores.append(_bag_ari(true_labels, pred_labels))

    return float(sum(scores) / len(scores)) if scores else 0.0


if __name__ == "__main__":
    import sys

    sub = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
    ans = pd.read_csv(sys.argv[2], dtype=str, keep_default_na=False)
    print(grade(sub, ans))
