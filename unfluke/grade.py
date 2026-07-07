#!/usr/bin/env python3
"""grade.py - score a submission for "Unfluke: Skill-vs-Luck Forensics
for Systematic Trading Records".

A submission provides, for each of the 2,432 test strategies:

    p_skill     in [0, 1]   probability the strategy has a genuine edge
    oos_sharpe  in [-10,10] predicted out-of-sample Sharpe ratio
    select      in {0, 1}   portfolio pick; exactly 240 rows must be 1

Composite score (higher is better, bounds [0, 1]):

    D  (weight 0.40)  skill detection, stratified by strategy family:
        AUC_f = Mann-Whitney ROC-AUC of p_skill vs the hidden skill flag
                within family f (average ranks for ties)
        D     = mean over the 4 families of max(0, 2 * AUC_f - 1)

    R  (weight 0.25)  within-market forward ranking:
        rho_a = Spearman rank correlation between predicted and true
                out-of-sample Sharpe within test arena a
                (rho_a = 0 when either side is constant)
        R     = max(0, mean over the 64 test arenas of rho_a)

    P  (weight 0.35)  portfolio precision:
        prec = (# selected strategies whose hidden skill flag is 1) / 240
        q    = base rate of skilled strategies in the test set
        P    = clip((prec - q) / (1 - q), 0, 1)

    penalty:  0.001 per logically inconsistent row, capped at 0.10.
        A row is inconsistent when select == 1 while p_skill < 0.2, or
        select == 1 while predicted oos_sharpe < 0 (each condition
        counts separately).

    score = clip(0.40 * D + 0.25 * R + 0.35 * P - penalty, 0, 1)

A submission is rejected with InvalidSubmissionError only when it cannot
be scored: unreadable file, missing columns, wrong row count, wrong or
duplicate IDs, missing/non-finite values, out-of-range values, or a
portfolio that does not contain exactly 240 picks.

Usage:
    python grade.py <submission.csv> [--answers dataset/private/answers.csv]

Prints a JSON report and exits 0 on success. Uses only pandas / numpy
(Kaggle Python Docker image).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID_COL = "strategy_id"
PSKILL_COL = "p_skill"
SHARPE_COL = "oos_sharpe"
SELECT_COL = "select"

ANSWER_SKILL = "skill"
ANSWER_SHARPE = "oos_sharpe"
ANSWER_FAMILY = "family"
ANSWER_ARENA = "arena_id"
ANSWER_TERCILE = "is_sharpe_tercile"

K_SELECT = 240
SHARPE_RANGE = 10.0
W_DETECTION = 0.40
W_RANKING = 0.25
W_PORTFOLIO = 0.35
PENALTY_PER_ROW = 0.001
PENALTY_CAP = 0.10
PSKILL_CONTRADICTION = 0.2


class InvalidSubmissionError(Exception):
    """Raised when a submission is malformed beyond scoring."""


def _load(obj, what: str) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    try:
        return pd.read_csv(obj)
    except Exception as exc:  # noqa: BLE001
        raise InvalidSubmissionError(
            f"Could not read {what} as CSV: {exc}") from exc


def _validate(sub: pd.DataFrame, ans: pd.DataFrame) -> pd.DataFrame:
    required = [ID_COL, PSKILL_COL, SHARPE_COL, SELECT_COL]
    missing = [c for c in required if c not in sub.columns]
    if missing:
        raise InvalidSubmissionError(
            f"Submission is missing required column(s) {missing}; "
            f"expected columns: {required} (found: {list(sub.columns)})")

    sub = sub[required].copy()
    sub[ID_COL] = sub[ID_COL].astype(str)

    if len(sub) != len(ans):
        raise InvalidSubmissionError(
            f"Expected exactly {len(ans)} rows (one per test strategy), "
            f"got {len(sub)}")

    if sub[ID_COL].duplicated().any():
        dupes = sub.loc[sub[ID_COL].duplicated(), ID_COL].head(3).tolist()
        raise InvalidSubmissionError(
            f"Submission contains duplicate strategy_id values, e.g. {dupes}")

    expected = set(ans[ID_COL].astype(str))
    got = set(sub[ID_COL])
    if got != expected:
        extra = sorted(got - expected)[:3]
        absent = sorted(expected - got)[:3]
        raise InvalidSubmissionError(
            "Submission strategy_id values do not match the test set "
            f"(unknown ids e.g. {extra}; missing ids e.g. {absent})")

    for col in (PSKILL_COL, SHARPE_COL, SELECT_COL):
        vals = pd.to_numeric(sub[col], errors="coerce")
        if vals.isna().any():
            raise InvalidSubmissionError(
                f"Column '{col}' contains missing or non-numeric values")
        if not np.isfinite(vals).all():
            raise InvalidSubmissionError(
                f"Column '{col}' contains non-finite values")
        sub[col] = vals

    if (sub[PSKILL_COL].lt(0) | sub[PSKILL_COL].gt(1)).any():
        raise InvalidSubmissionError("p_skill values must lie in [0, 1]")

    if sub[SHARPE_COL].abs().gt(SHARPE_RANGE).any():
        raise InvalidSubmissionError(
            f"oos_sharpe predictions must lie in [-{SHARPE_RANGE}, "
            f"{SHARPE_RANGE}]")

    bad_select = ~sub[SELECT_COL].isin([0, 1])
    if bad_select.any():
        raise InvalidSubmissionError("select values must be 0 or 1")
    sub[SELECT_COL] = sub[SELECT_COL].astype(int)

    n_sel = int(sub[SELECT_COL].sum())
    if n_sel != K_SELECT:
        raise InvalidSubmissionError(
            f"Exactly {K_SELECT} rows must have select == 1 (got {n_sel})")

    merged = ans.merge(sub, on=ID_COL, how="left", validate="one_to_one",
                       suffixes=("_true", "_pred"))
    return merged


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC with average ranks for ties."""
    pos = labels == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def grade(submission, answers) -> float:
    """Score a submission against the private answer key.

    `submission` and `answers` may each be a pandas DataFrame (platform
    entrypoint) or a path to a CSV file (CLI use). Returns the composite
    score in [0, 1]; higher is better. Raises InvalidSubmissionError for
    truly invalid submissions.
    """
    ans = _load(answers, "answers")
    ans[ID_COL] = ans[ID_COL].astype(str)
    sub = _load(submission, "submission")
    m = _validate(sub, ans)

    skill = m[ANSWER_SKILL].astype(int).to_numpy()
    true_sharpe = m[f"{ANSWER_SHARPE}_true"].astype(float).to_numpy()
    p_skill = m[PSKILL_COL].astype(float).to_numpy()
    pred_sharpe = m[f"{SHARPE_COL}_pred"].astype(float).to_numpy()
    select = m[SELECT_COL].to_numpy()

    # ---- D: family-stratified skill detection ----
    family_auc = {}
    for fam, idx in m.groupby(ANSWER_FAMILY).groups.items():
        loc = m.index.get_indexer(idx)
        family_auc[str(fam)] = _auc(skill[loc], p_skill[loc])
    detection = float(np.mean(
        [max(0.0, 2.0 * a - 1.0) for a in family_auc.values()]))

    # ---- R: within-arena forward ranking ----
    rhos = []
    for _, grp in m.groupby(ANSWER_ARENA):
        pred = grp[f"{SHARPE_COL}_pred"].astype(float)
        true = grp[f"{ANSWER_SHARPE}_true"].astype(float)
        if pred.nunique() < 2 or true.nunique() < 2:
            rhos.append(0.0)
            continue
        rho = pred.corr(true, method="spearman")
        rhos.append(0.0 if pd.isna(rho) else float(rho))
    ranking = max(0.0, float(np.mean(rhos)))

    # ---- P: portfolio precision over the base rate ----
    base_rate = float(skill.mean())
    precision = float(skill[select == 1].mean())
    portfolio = float(np.clip(
        (precision - base_rate) / (1.0 - base_rate), 0.0, 1.0))

    # ---- logical consistency penalty ----
    n_bad = int(((select == 1) & (p_skill < PSKILL_CONTRADICTION)).sum())
    n_bad += int(((select == 1) & (pred_sharpe < 0.0)).sum())
    penalty = min(PENALTY_PER_ROW * n_bad, PENALTY_CAP)

    score = float(np.clip(
        W_DETECTION * detection + W_RANKING * ranking
        + W_PORTFOLIO * portfolio - penalty, 0.0, 1.0))

    # diagnostics for humans; grade() return value is `score`
    tercile_auc = {}
    for terc, idx in m.groupby(ANSWER_TERCILE).groups.items():
        loc = m.index.get_indexer(idx)
        tercile_auc[str(terc)] = round(_auc(skill[loc], p_skill[loc]), 4)
    grade.last_report = {
        "score": round(score, 6),
        "detection_D": round(detection, 4),
        "family_auc": {k: round(v, 4) for k, v in sorted(family_auc.items())},
        "ranking_R": round(ranking, 4),
        "portfolio_P": round(portfolio, 4),
        "portfolio_precision": round(precision, 4),
        "skilled_base_rate": round(base_rate, 4),
        "inconsistent_rows": n_bad,
        "penalty": round(penalty, 4),
        "is_sharpe_tercile_auc": tercile_auc,
    }
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", help="path to submission.csv")
    parser.add_argument(
        "--answers",
        default=str(Path(__file__).resolve().parent
                    / "dataset" / "private" / "answers.csv"),
        help="path to the private answer key",
    )
    args = parser.parse_args()
    try:
        grade(args.submission, args.answers)
    except InvalidSubmissionError as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    print(json.dumps(grade.last_report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
