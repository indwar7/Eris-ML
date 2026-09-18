"""Offline structural validation for submission.csv -- run this before
submitting. Mirrors the grader's own behaviour: it checks structure only
(columns, one row per bag, label counts), never correctness, since ground
truth is private.

Shape: ONE ROW PER BAG --
    bag_id, batch_id
where `batch_id` is a whitespace-separated list of group labels, one per
snippet of that bag, in the snippet order of test.csv (the integer after
"::" in each test row_id).

Usage: python validate_submission.py submission.csv test.csv
"""
import re
import sys

import pandas as pd

REQUIRED_COLUMNS = {"bag_id", "batch_id"}
_SPLIT = re.compile(r"[\s,;]+")


def _tokens(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    return [t for t in _SPLIT.split(s) if t] if s else []


def validate(submission_path: str, test_path: str) -> None:
    errors = []
    warnings = []

    try:
        sub = pd.read_csv(submission_path, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"ERROR: could not read {submission_path}: {e}")
        sys.exit(1)

    test = pd.read_csv(test_path, dtype=str, keep_default_na=False)
    test_bag = test["row_id"].str.split("::", n=1).str[0]
    n_snippets = test_bag.value_counts().to_dict()

    missing_cols = REQUIRED_COLUMNS - set(sub.columns)
    if missing_cols:
        errors.append(f"missing required columns: {sorted(missing_cols)} -- the grader rejects this")

    dup = []
    if not errors:
        ids = sub["bag_id"].str.strip()
        dup = ids[ids.duplicated()].unique().tolist()
        if dup:
            warnings.append(
                f"{len(dup)} bag_id(s) appear more than once (e.g. {dup[:3]}) -- "
                "each such bag scores 0 (the grader does not pick one of the conflicting rows)"
            )

    if not errors:
        got = {b: g for b, g in zip(sub["bag_id"].str.strip(), sub["batch_id"]) if b not in set(dup)}
        missing_bags = [b for b in n_snippets if b not in got]
        extra_bags = [b for b in got if b not in n_snippets]
        wrong_len = []
        all_singleton = True
        for b, n in n_snippets.items():
            toks = _tokens(got.get(b))
            if b in got and len(toks) != n:
                wrong_len.append((b, len(toks), n))
            if b in got and len(toks) == n and len(set(toks)) != n:
                all_singleton = False
        if missing_bags:
            warnings.append(
                f"{len(missing_bags)} bags from test.csv are missing from your submission "
                f"(e.g. {missing_bags[:3]}) -- each scores 0 for that bag, not a rejection"
            )
        if wrong_len:
            b, k, n = wrong_len[0]
            warnings.append(
                f"{len(wrong_len)} bags have the wrong number of labels (e.g. {b}: {k} labels "
                f"for {n} snippets) -- each scores 0 for that bag"
            )
        if extra_bags:
            warnings.append(f"{len(extra_bags)} bag_ids are not in test.csv -- these are ignored")
        if all_singleton and not missing_bags and not wrong_len:
            warnings.append(
                "every bag is all-singletons (every snippet its own group) -- valid, "
                "but scores exactly 0 under ARI"
            )

    if errors:
        print("FAILED structural validation:")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)

    if warnings:
        print("Structural validation passed, with warnings (these cost points, they do not block submission):")
        for w in warnings:
            print(f"  WARNING: {w}")
    else:
        print("Structural validation passed: one row per bag, every bag present, label counts match.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python validate_submission.py submission.csv test.csv")
        sys.exit(1)
    validate(sys.argv[1], sys.argv[2])
