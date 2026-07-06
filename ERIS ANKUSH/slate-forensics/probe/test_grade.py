#!/usr/bin/env python3
"""probe/test_grade.py — validation tests for grade.py.

Covers: valid submissions, every malformed-submission branch, consistency
penalties, edge cases (string flags, mixed-case modes, extra columns,
shuffled rows) and determinism of the returned score.

Run either way:
    python probe/test_grade.py
    python -m pytest probe/test_grade.py -q
Uses only pandas / numpy (Kaggle Python Docker image).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grade import InvalidSubmissionError, grade  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANSWERS = ROOT / "dataset" / "private" / "answers.csv"
SAMPLE = ROOT / "dataset" / "public" / "sample_submission.csv"


# --------------------------------------------------------------------------
# synthetic micro-fixture: 4 slates, one per condition
# --------------------------------------------------------------------------
def toy_answers() -> pd.DataFrame:
    conds = [("T1", 0, "none"), ("T2", 1, "popularity_fallback"),
             ("T3", 1, "price_band_shift"), ("T4", 1, "stale_index")]
    rows = []
    for n, (sid, flag, mode) in enumerate(conds):
        for pos in range(1, 11):
            healthy = f"H{n}{pos:02d}"
            emitted = healthy if flag == 0 else f"E{n}{pos:02d}"
            rows.append({"slate_id": sid, "corrupted": flag, "mode": mode,
                         "position": pos, "healthy_code": healthy,
                         "emitted_code": emitted})
    return pd.DataFrame(rows)


def perfect_sub(ans: pd.DataFrame) -> pd.DataFrame:
    out = ans.rename(columns={"healthy_code": "stock_code"})
    return out[["slate_id", "corrupted", "mode", "position", "stock_code"]].copy()


def copy_emitted_sub(ans: pd.DataFrame) -> pd.DataFrame:
    out = ans.rename(columns={"emitted_code": "stock_code"})
    out = out[["slate_id", "position", "stock_code"]].copy()
    out.insert(1, "corrupted", 0)
    out.insert(2, "mode", "none")
    return out


def expect_invalid(sub, ans, needle: str, label: str) -> None:
    try:
        grade(sub, ans)
    except InvalidSubmissionError as exc:
        assert needle.lower() in str(exc).lower(), (
            f"{label}: wrong message: {exc}")
        print(f"  ok  {label}: rejected ({str(exc)[:60]}...)")
        return
    raise AssertionError(f"{label}: malformed submission was NOT rejected")


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_perfect_scores_one():
    ans = toy_answers()
    s, comp = grade(perfect_sub(ans), ans, verbose=True)
    assert abs(s - 1.0) < 1e-9, s
    assert comp["S_consistency"] == 1.0
    print(f"  ok  perfect toy submission -> {s}")


def test_copy_emitted_is_consistent_but_low():
    ans = toy_answers()
    s, comp = grade(copy_emitted_sub(ans), ans, verbose=True)
    # repair: only the clean bucket matches -> S_repair = 0.25;
    # audit balanced 0.5; mode 0; consistency perfect.
    assert abs(comp["S_repair"] - 0.25) < 1e-9
    assert abs(comp["S_audit"] - 0.5) < 1e-9
    assert comp["S_mode"] == 0.0
    assert comp["S_consistency"] == 1.0
    assert abs(s - (0.5 * 0.25 + 0.3 * 0.5)) < 1e-9
    print(f"  ok  copy-emitted toy submission -> {s:.4f}")


def test_consistency_penalty_applies():
    ans = toy_answers()
    # v1: claim clean but tamper with one item of T1
    sub = copy_emitted_sub(ans)
    sub.loc[(sub.slate_id == "T1") & (sub.position == 1), "stock_code"] = "ZZZZZ"
    _, comp = grade(sub, ans, verbose=True)
    assert comp["S_consistency"] == 0.75
    assert comp["consistency_violations"]["flag_clean_but_slate_differs"] == 1

    # isolate the multiplier: flag T2 corrupted in both submissions; the only
    # difference is whether its slate was actually changed (v2 or not). Repair
    # (zero overlap either way), audit and mode components are identical.
    base = copy_emitted_sub(ans)
    m = base.slate_id == "T2"
    base.loc[m, "corrupted"] = 1
    base.loc[m, "mode"] = "popularity_fallback"
    sub_bad = base.copy()                                     # slate unchanged -> v2
    sub_ok = base.copy()                                      # reverse the slate
    sub_ok.loc[m, "stock_code"] = sub_ok.loc[m, "stock_code"].to_numpy()[::-1]
    s_bad, c_bad = grade(sub_bad, ans, verbose=True)
    s_ok, c_ok = grade(sub_ok, ans, verbose=True)
    assert c_bad["consistency_violations"]["flag_corrupted_but_slate_unchanged"] == 1
    assert c_ok["S_consistency"] == 1.0 and c_bad["S_consistency"] == 0.75
    for key in ("S_repair", "S_audit", "S_mode"):
        assert c_bad[key] == c_ok[key], key
    assert s_ok > s_bad  # only the consistency multiplier differs
    print(f"  ok  consistency penalty: inconsistent {s_bad:.4f} < consistent {s_ok:.4f}")


def test_flag_corrupted_but_unchanged_penalised():
    ans = toy_answers()
    sub = copy_emitted_sub(ans)
    m = sub.slate_id == "T2"
    sub.loc[m, "corrupted"] = 1          # claims corrupted...
    sub.loc[m, "mode"] = "popularity_fallback"
    # ...but leaves the emitted slate untouched -> v2
    _, comp = grade(sub, ans, verbose=True)
    assert comp["consistency_violations"]["flag_corrupted_but_slate_unchanged"] == 1
    print("  ok  corrupted-but-unchanged violation detected")


def test_malformed_battery():
    ans = toy_answers()
    good = perfect_sub(ans)

    expect_invalid(good.drop(columns=["mode"]), ans, "missing required",
                   "missing column")
    expect_invalid(good.iloc[:-1], ans, "exactly 10 rows", "9 rows in a slate")
    dup_pos = good.copy()
    dup_pos.loc[dup_pos.index[-1], "position"] = 1
    expect_invalid(dup_pos, ans, "position", "duplicate position")
    dup_item = good.copy()
    dup_item.loc[dup_item.index[1], "stock_code"] = dup_item.iloc[0]["stock_code"]
    expect_invalid(dup_item, ans, "duplicated item", "duplicate item in slate")
    wrong_ids = good.copy()
    wrong_ids["slate_id"] = wrong_ids["slate_id"].replace({"T1": "TX"})
    expect_invalid(wrong_ids, ans, "missing", "missing required slate_id")
    bad_flag = good.copy()
    bad_flag.loc[bad_flag.index[0], "corrupted"] = 2
    expect_invalid(bad_flag, ans, "0 or 1", "flag out of range")
    bad_mode = good.copy()
    bad_mode.loc[bad_mode.index[0], "mode"] = "gremlins"
    expect_invalid(bad_mode, ans, "unknown mode", "unknown mode")
    empty_code = good.copy()
    empty_code.loc[empty_code.index[0], "stock_code"] = ""
    expect_invalid(empty_code, ans, "empty/missing stock_code", "empty code")
    split_flag = good.copy()
    split_flag.loc[split_flag.index[0], "corrupted"] = 1 - int(split_flag.iloc[0]["corrupted"])
    expect_invalid(split_flag, ans, "not constant", "flag varies within slate")
    split_mode = good.copy()
    split_mode.loc[(split_mode.slate_id == "T2").idxmax(), "mode"] = "stale_index"
    expect_invalid(split_mode, ans, "not constant", "mode varies within slate")
    bad_pos = good.copy()
    bad_pos.loc[bad_pos.index[0], "position"] = 0
    expect_invalid(bad_pos, ans, "invalid position", "position 0")
    expect_invalid("no/such/file.csv", ans, "could not read", "unreadable path")


def test_edge_cases_tolerated():
    ans = toy_answers()
    sub = perfect_sub(ans)
    sub["corrupted"] = sub["corrupted"].astype(str)          # string flags
    sub["mode"] = sub["mode"].str.upper()                    # case-insensitive
    sub["extra_column"] = "ignored"
    shuffled = sub.sample(frac=1.0, random_state=13)         # any row order
    s = grade(shuffled, ans)
    assert abs(s - 1.0) < 1e-9
    print("  ok  string flags, uppercase modes, extra cols, shuffled rows")


def test_deterministic_and_stable():
    ans = toy_answers()
    sub = copy_emitted_sub(ans)
    scores = {grade(sub, ans) for _ in range(5)}
    assert len(scores) == 1
    print(f"  ok  5 repeat gradings identical -> {scores.pop():.6f}")


def test_partial_answers_tolerated():
    """The platform may split answers.csv into public/private portions —
    row-wise (some positions of each slate) or slate-wise (some slates).
    The grader must score whatever portion it is given, and a perfect
    submission must still score 1.0 on any portion."""
    ans = toy_answers()
    sub = perfect_sub(ans)

    # row-wise split: keep positions 2, 5, 7 of every slate
    part_rows = ans[ans.position.isin([2, 5, 7])].reset_index(drop=True)
    s, comp = grade(sub, part_rows, verbose=True)
    assert abs(s - 1.0) < 1e-9, s
    assert comp["S_consistency"] == 1.0

    # slate-wise split: only two of the four slates
    part_slates = ans[ans.slate_id.isin(["T1", "T3"])].reset_index(drop=True)
    s2 = grade(sub, part_slates)
    assert abs(s2 - 1.0) < 1e-9, s2

    # copy-emitted on the row-wise portion stays consistent and low
    s3, comp3 = grade(copy_emitted_sub(ans), part_rows, verbose=True)
    assert comp3["S_consistency"] == 1.0 and s3 < 0.5
    print(f"  ok  partial answers: perfect=1.0 (rows+slates), "
          f"copy-emitted={s3:.4f}")


def test_real_dataset_if_present():
    if not (ANSWERS.exists() and SAMPLE.exists()):
        print("  --  real dataset not prepared; skipping integration checks")
        return
    ans = pd.read_csv(ANSWERS, dtype=str)
    s_sample, comp = grade(pd.read_csv(SAMPLE, dtype=str), ans, verbose=True)
    assert 0.0 < s_sample < 0.45, f"sample floor out of band: {s_sample}"
    assert comp["S_consistency"] == 1.0
    perfect = ans.rename(columns={"healthy_code": "stock_code"})[
        ["slate_id", "corrupted", "mode", "position", "stock_code"]]
    assert abs(grade(perfect, ans) - 1.0) < 1e-9
    r1 = grade(pd.read_csv(SAMPLE, dtype=str), ans)
    r2 = grade(pd.read_csv(SAMPLE, dtype=str), ans)
    assert r1 == r2 == s_sample
    print(f"  ok  real data: sample={s_sample:.4f}, oracle=1.0, stable reruns")


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    os.chdir(ROOT)
    for fn in ALL_TESTS:
        print(f"[{fn.__name__}]")
        fn()
    print(f"\nAll {len(ALL_TESTS)} test groups passed.")
