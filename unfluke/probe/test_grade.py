#!/usr/bin/env python3
"""test_grade.py - behavior tests for grade.py.

Covers: perfect submission, sample submission, malformed inputs
(missing columns, wrong row count, duplicate/unknown ids, NaN,
out-of-range values, wrong portfolio size), the consistency penalty,
component edge cases (constant predictions), and determinism.

Run either way:
    python probe/test_grade.py
    pytest probe/test_grade.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grade import (  # noqa: E402
    K_SELECT, InvalidSubmissionError, grade,
)

ANSWERS = ROOT / "dataset" / "private" / "answers.csv"


def _answers() -> pd.DataFrame:
    return pd.read_csv(ANSWERS)


def _perfect() -> pd.DataFrame:
    ans = _answers()
    sub = ans[["strategy_id"]].copy()
    sub["p_skill"] = ans["skill"].astype(float)
    sub["oos_sharpe"] = ans["oos_sharpe"].clip(-10, 10)
    # pick K truly skilled strategies, predicted non-negative
    sub["select"] = 0
    skilled = ans.index[ans["skill"] == 1][:K_SELECT]
    sub.loc[skilled, "select"] = 1
    # consistency: selected rows must not predict a negative sharpe
    sel = sub["select"] == 1
    sub.loc[sel, "oos_sharpe"] = sub.loc[sel, "oos_sharpe"].clip(lower=0.0)
    return sub


def test_perfect_submission_scores_near_one():
    score = grade(_perfect(), _answers())
    rep = grade.last_report
    assert rep["detection_D"] > 0.99, rep
    assert rep["portfolio_P"] == 1.0, rep
    assert rep["penalty"] == 0.0, rep
    # R is < 1 because true oos_sharpe was clipped upward on selected rows
    assert score > 0.95, rep
    print(f"  perfect submission: {score:.4f}")


def test_sample_submission_scores_near_floor():
    score = grade(str(ROOT / "dataset" / "public" / "sample_submission.csv"),
                  _answers())
    assert 0.0 <= score < 0.15, grade.last_report
    print(f"  sample submission: {score:.4f}")


def test_determinism():
    s1 = grade(_perfect(), _answers())
    s2 = grade(_perfect(), _answers())
    assert s1 == s2
    print(f"  deterministic: {s1:.6f} == {s2:.6f}")


def _expect_invalid(sub, why: str):
    try:
        grade(sub, _answers())
    except InvalidSubmissionError:
        print(f"  rejected as expected: {why}")
        return
    raise AssertionError(f"submission should have been rejected: {why}")


def test_missing_column():
    _expect_invalid(_perfect().drop(columns=["p_skill"]), "missing column")


def test_wrong_row_count():
    _expect_invalid(_perfect().iloc[:-3], "wrong row count")


def test_duplicate_ids():
    sub = _perfect()
    sub.loc[1, "strategy_id"] = sub.loc[0, "strategy_id"]
    _expect_invalid(sub, "duplicate ids")


def test_unknown_ids():
    sub = _perfect()
    sub.loc[0, "strategy_id"] = "S99999"
    _expect_invalid(sub, "unknown id")


def test_nan_values():
    sub = _perfect()
    sub.loc[0, "p_skill"] = np.nan
    _expect_invalid(sub, "NaN p_skill")


def test_out_of_range_p_skill():
    sub = _perfect()
    sub.loc[0, "p_skill"] = 1.5
    _expect_invalid(sub, "p_skill > 1")


def test_out_of_range_sharpe():
    sub = _perfect()
    sub.loc[0, "oos_sharpe"] = 25.0
    _expect_invalid(sub, "oos_sharpe out of range")


def test_nonbinary_select():
    sub = _perfect()
    sub.loc[sub.index[0], "select"] = 2
    _expect_invalid(sub, "select not in {0,1}")


def test_wrong_portfolio_size():
    sub = _perfect()
    first_zero = sub.index[sub["select"] == 0][0]
    sub.loc[first_zero, "select"] = 1  # K+1 picks
    _expect_invalid(sub, f"{K_SELECT + 1} picks instead of {K_SELECT}")


def test_consistency_penalty():
    clean = _perfect()
    base = grade(clean, _answers())
    base_rep = dict(grade.last_report)

    dirty = clean.copy()
    sel_idx = dirty.index[dirty["select"] == 1][:30]
    dirty.loc[sel_idx, "oos_sharpe"] = -0.5   # picked but predicted to lose
    dirty_score = grade(dirty, _answers())
    rep = grade.last_report
    assert rep["inconsistent_rows"] == 30, rep
    assert rep["penalty"] == 0.03, rep
    assert dirty_score < base, (dirty_score, base)
    print(f"  penalty applied: {base:.4f} -> {dirty_score:.4f} "
          f"(30 contradictions, penalty {rep['penalty']}; "
          f"clean penalty {base_rep['penalty']})")


def test_constant_predictions_zero_components():
    ans = _answers()
    sub = _perfect()
    sub["p_skill"] = 0.5
    sub["oos_sharpe"] = 1.0
    grade(sub, ans)
    rep = grade.last_report
    assert rep["detection_D"] == 0.0, rep
    assert rep["ranking_R"] == 0.0, rep
    print("  constant p_skill/oos_sharpe -> D = R = 0")


def main() -> None:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"running {len(tests)} grader tests against {ANSWERS}")
    for name, fn in tests:
        print(f"[{name}]")
        fn()
    print("ALL GRADER TESTS PASSED")


if __name__ == "__main__":
    main()
