#!/usr/bin/env python3
"""grade.py — composite scorer for "Slate Forensics: Auditing and Repairing a
Broken Product Recommender".

A submission carries, for every test slate, three coupled outputs:
  * an audit flag        corrupted ∈ {0, 1}
  * a failure diagnosis  mode ∈ {none, popularity_fallback,
                                 price_band_shift, stale_index}
  * a repaired slate     positions 1..10, each a stock_code (the solver's
                         reconstruction of what the HEALTHY policy served;
                         for a clean slate that is the emitted slate itself)

Composite score (higher is better, bounds [0, 1]):

    S_repair : truncated rank-biased overlap RBO@10 (p = 0.9) between the
               submitted slate and the hidden healthy slate, averaged within
               each true-condition bucket {none, popularity_fallback,
               price_band_shift, stale_index}, then averaged across the four
               buckets with EQUAL weight (a submission cannot farm the
               majority clean bucket).
    S_audit  : 0.5 * mean per-mode detection recall  +  0.5 * clean
               specificity (balanced so all-clean and all-corrupted both
               land at 0.5).
    S_mode   : macro recall of the predicted failure mode over the three
               corrupted conditions (predicting 'none' on a corrupted slate
               scores zero for that slate).
    S_cons   : fraction of slates free of logical-consistency violations:
                 v1 flag=0 but submitted slate differs from the emitted one
                 v2 flag=1 but submitted slate equals the emitted one
                 v3 flag=0 but mode != none
                 v4 flag=1 but mode == none

    final = (0.50*S_repair + 0.30*S_audit + 0.20*S_mode) * (0.7 + 0.3*S_cons)

Fabricated stock codes earn no overlap and displace real matches, so
ungrounded repairs are strictly dominated; when dataset/public/catalog.csv is
reachable the CLI additionally reports (not scores) the ungrounded-item rate.

Malformed submissions raise InvalidSubmissionError with a precise reason:
missing columns, missing slates, missing/duplicate positions, duplicate
items inside a slate, empty codes, non-{0,1} flags, unknown modes, or a flag/
mode that is not constant within its slate.

Platform note: the grader tolerates PARTIAL answer files — the platform may
split answers.csv row-wise or slate-wise into public/private portions. The
submission must always be complete (10 rows per evaluation slate); scoring
restricts itself to the slates/positions present in the answers it is given,
and submitted slates absent from the answers are ignored. With the full
answers file the scores are identical to the original definition.

Usage:
    python grade.py <submission.csv> [--answers dataset/private/answers.csv]

Prints a JSON object with the component scores and exits 0 on success.
Only pandas / numpy are used (Kaggle Python Docker image).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ID_COL, FLAG_COL, MODE_COL = "slate_id", "corrupted", "mode"
POS_COL, CODE_COL = "position", "stock_code"
TOP_K = 10
RBO_P = 0.9
MODES = ["popularity_fallback", "price_band_shift", "stale_index"]
ALL_CONDITIONS = ["none"] + MODES
W_REPAIR, W_AUDIT, W_MODE = 0.50, 0.30, 0.20
CONS_FLOOR = 0.7  # final = base * (CONS_FLOOR + (1-CONS_FLOOR) * S_cons)


class InvalidSubmissionError(Exception):
    """Raised when a submission is malformed beyond scoring."""


# --------------------------------------------------------------------------
# loading / validation
# --------------------------------------------------------------------------
def _load(obj, what: str) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    try:
        return pd.read_csv(obj, dtype=str)
    except Exception as exc:  # noqa: BLE001
        raise InvalidSubmissionError(
            f"Could not read {what} as CSV: {exc}") from exc


def _validate(sub: pd.DataFrame, ans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [ID_COL, FLAG_COL, MODE_COL, POS_COL, CODE_COL]
    missing = [c for c in required if c not in sub.columns]
    if missing:
        raise InvalidSubmissionError(
            f"Submission is missing required column(s) {missing}; expected "
            f"columns {required} (found {list(sub.columns)}).")

    sub = sub[required].copy()
    sub[ID_COL] = sub[ID_COL].astype(str).str.strip()
    sub[CODE_COL] = sub[CODE_COL].astype(str).str.strip().str.upper()
    sub[MODE_COL] = sub[MODE_COL].astype(str).str.strip().str.lower()

    if (sub[CODE_COL].isin(["", "NAN", "NONE"])).any():
        n = int(sub[CODE_COL].isin(["", "NAN", "NONE"]).sum())
        raise InvalidSubmissionError(
            f"Submission contains {n} empty/missing stock_code value(s); "
            f"every position 1..{TOP_K} needs a real item code.")

    pos = pd.to_numeric(sub[POS_COL], errors="coerce")
    bad = pos.isna() | (pos != pos.round()) | (pos < 1) | (pos > TOP_K)
    if bad.any():
        ex = sub.loc[bad, POS_COL].astype(str).unique()[:5].tolist()
        raise InvalidSubmissionError(
            f"Submission contains {int(bad.sum())} invalid position value(s) "
            f"(e.g. {ex}); positions must be integers 1..{TOP_K}.")
    sub[POS_COL] = pos.astype(int)

    flag = pd.to_numeric(sub[FLAG_COL], errors="coerce")
    bad = flag.isna() | ~flag.isin([0, 1])
    if bad.any():
        ex = sub.loc[bad, FLAG_COL].astype(str).unique()[:5].tolist()
        raise InvalidSubmissionError(
            f"Submission contains invalid corrupted value(s) (e.g. {ex}); "
            f"the flag must be 0 or 1.")
    sub[FLAG_COL] = flag.astype(int)

    unknown = set(sub[MODE_COL].unique()) - set(ALL_CONDITIONS)
    if unknown:
        raise InvalidSubmissionError(
            f"Submission contains unknown mode value(s) {sorted(unknown)[:5]}; "
            f"allowed: {ALL_CONDITIONS}.")

    expected = set(ans[ID_COL].astype(str))
    got = set(sub[ID_COL])
    miss = sorted(expected - got)
    if miss:
        raise InvalidSubmissionError(
            f"Submission slate_ids do not match the evaluation set: "
            f"{len(miss)} required slate_id(s) are missing (e.g. {miss[:5]}). "
            f"Provide every slate_id from slates_test.csv.")
    # slates not present in the answers (e.g. the platform is scoring a
    # public/private portion) are simply not scored
    sub = sub[sub[ID_COL].isin(expected)]

    counts = sub.groupby(ID_COL).size()
    wrong = counts[counts != TOP_K]
    if len(wrong):
        raise InvalidSubmissionError(
            f"{len(wrong)} slate(s) do not have exactly {TOP_K} rows "
            f"(e.g. {dict(wrong.head(3))}).")

    if sub.duplicated([ID_COL, POS_COL]).any():
        ex = sub[sub.duplicated([ID_COL, POS_COL])].head(3)
        raise InvalidSubmissionError(
            "Submission contains duplicated (slate_id, position) pair(s), "
            f"e.g.\n{ex[[ID_COL, POS_COL]].to_string(index=False)}. Each "
            f"slate needs each position 1..{TOP_K} exactly once.")

    if sub.duplicated([ID_COL, CODE_COL]).any():
        ex = sub[sub.duplicated([ID_COL, CODE_COL])].head(3)
        raise InvalidSubmissionError(
            "Submission contains duplicated item(s) within a slate, e.g.\n"
            f"{ex[[ID_COL, CODE_COL]].to_string(index=False)}. An item may "
            f"appear at most once per slate.")

    per_flag = sub.groupby(ID_COL)[FLAG_COL].nunique()
    if (per_flag > 1).any():
        ex = per_flag[per_flag > 1].index[:3].tolist()
        raise InvalidSubmissionError(
            f"corrupted flag is not constant within slate(s) {ex}; give one "
            f"flag per slate, repeated on its {TOP_K} rows.")
    per_mode = sub.groupby(ID_COL)[MODE_COL].nunique()
    if (per_mode > 1).any():
        ex = per_mode[per_mode > 1].index[:3].tolist()
        raise InvalidSubmissionError(
            f"mode is not constant within slate(s) {ex}; give one mode per "
            f"slate, repeated on its {TOP_K} rows.")

    ans = ans.copy()
    ans[ID_COL] = ans[ID_COL].astype(str)
    ans[POS_COL] = ans[POS_COL].astype(int)
    for c in ("healthy_code", "emitted_code"):
        ans[c] = ans[c].astype(str).str.strip().str.upper()
    ans["corrupted"] = ans["corrupted"].astype(int)
    return sub, ans


# --------------------------------------------------------------------------
# components
# --------------------------------------------------------------------------
def _rbo_at_k(pred: list[str], truth_by_pos: dict[int, str]) -> float:
    """Truncated rank-biased overlap (p = RBO_P) between the submitted
    ranking and the healthy slate, which may be OBSERVED ONLY PARTIALLY
    (truth_by_pos maps observed positions 1..10 to codes). Agreement at
    depth d is |pred[:d] ∩ observed_truth[:d]| / |observed_truth[:d]|; depths
    with no observed truth are skipped and the weights renormalised. With all
    10 positions observed this is exactly the standard truncated RBO."""
    weights = RBO_P ** np.arange(TOP_K)
    pred_seen: set[str] = set()
    truth_seen: set[str] = set()
    acc = wsum = 0.0
    for d in range(1, TOP_K + 1):
        if d <= len(pred):
            pred_seen.add(pred[d - 1])
        if d in truth_by_pos:
            truth_seen.add(truth_by_pos[d])
        if truth_seen:
            acc += weights[d - 1] * len(pred_seen & truth_seen) / len(truth_seen)
            wsum += weights[d - 1]
    return float(acc / wsum) if wsum else 0.0


def grade(submission, answers, verbose: bool = False):
    """Score a submission; returns the composite float in [0, 1].

    `submission` / `answers` may be DataFrames (platform entrypoint) or CSV
    paths (CLI). With verbose=True returns (score, components_dict) instead.
    """
    sub = _load(submission, "submission")
    ans = _load(answers, "answers")
    sub, ans = _validate(sub, ans)

    sub = sub.sort_values([ID_COL, POS_COL], kind="mergesort")
    ans = ans.sort_values([ID_COL, POS_COL], kind="mergesort")

    pred_slates = sub.groupby(ID_COL)[CODE_COL].agg(list)
    # observed healthy positions per slate (the answers may be partial)
    truth_pos = {i: dict(zip(g[POS_COL], g["healthy_code"]))
                 for i, g in ans.groupby(ID_COL)}
    truth = ans.drop_duplicates(ID_COL).set_index(ID_COL)
    pred = (sub.drop_duplicates(ID_COL).set_index(ID_COL)
            .reindex(truth.index))

    ids = truth.index
    true_cond = truth[MODE_COL]                       # none / 3 failure modes
    pred_flag = pred[FLAG_COL]
    pred_mode = pred[MODE_COL]

    # ---- S_repair: bucket-balanced RBO@10 against the healthy slate -------
    # (buckets absent from the supplied answers are skipped)
    rbo = pd.Series({i: _rbo_at_k(pred_slates[i], truth_pos[i])
                     for i in ids})
    repair_by_bucket = {c: float(rbo[true_cond == c].mean())
                        for c in ALL_CONDITIONS if (true_cond == c).any()}
    s_repair = float(np.mean(list(repair_by_bucket.values())))

    # ---- S_audit: balanced detection --------------------------------------
    recalls = {m: float((pred_flag[true_cond == m] == 1).mean())
               for m in MODES if (true_cond == m).any()}
    audit_parts = []
    if recalls:
        audit_parts.append(float(np.mean(list(recalls.values()))))
    specificity = (float((pred_flag[true_cond == "none"] == 0).mean())
                   if (true_cond == "none").any() else None)
    if specificity is not None:
        audit_parts.append(specificity)
    s_audit = float(np.mean(audit_parts)) if audit_parts else 0.0

    # ---- S_mode: macro diagnosis recall over corrupted slates -------------
    mode_recalls = {m: float((pred_mode[true_cond == m] == m).mean())
                    for m in MODES if (true_cond == m).any()}
    s_mode = float(np.mean(list(mode_recalls.values()))) if mode_recalls else 0.0

    # ---- S_cons: logical consistency of the three outputs -----------------
    # compare submitted items to the emitted slate at the observed positions
    merged = ans[[ID_COL, POS_COL, "emitted_code"]].merge(
        sub[[ID_COL, POS_COL, CODE_COL]], on=[ID_COL, POS_COL], how="left")
    same_as_emitted = ((merged["emitted_code"] == merged[CODE_COL])
                       .groupby(merged[ID_COL]).all().reindex(ids))
    v1 = (pred_flag == 0) & ~same_as_emitted
    v2 = (pred_flag == 1) & same_as_emitted
    v3 = (pred_flag == 0) & (pred_mode != "none")
    v4 = (pred_flag == 1) & (pred_mode == "none")
    violated = v1 | v2 | v3 | v4
    s_cons = float(1.0 - violated.mean())

    base = W_REPAIR * s_repair + W_AUDIT * s_audit + W_MODE * s_mode
    final = float(np.clip(base * (CONS_FLOOR + (1 - CONS_FLOOR) * s_cons),
                          0.0, 1.0))

    if not verbose:
        return final
    components = {
        "final_score": round(final, 6),
        "S_repair": round(s_repair, 6),
        "S_audit": round(s_audit, 6),
        "S_mode": round(s_mode, 6),
        "S_consistency": round(s_cons, 6),
        "repair_rbo_by_condition": {k: round(v, 6)
                                    for k, v in repair_by_bucket.items()},
        "audit_recall_by_mode": {k: round(v, 6) for k, v in recalls.items()},
        "audit_clean_specificity": (round(specificity, 6)
                                    if specificity is not None else None),
        "mode_recall_by_mode": {k: round(v, 6)
                                for k, v in mode_recalls.items()},
        "consistency_violations": {
            "flag_clean_but_slate_differs": int(v1.sum()),
            "flag_corrupted_but_slate_unchanged": int(v2.sum()),
            "flag_clean_but_mode_set": int(v3.sum()),
            "flag_corrupted_but_mode_none": int(v4.sum()),
        },
    }
    return final, components


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _ungrounded_rate(sub_path: str, catalog_path: str) -> float | None:
    try:
        cat = set(pd.read_csv(catalog_path, dtype=str)["stock_code"]
                  .str.strip().str.upper())
        codes = (pd.read_csv(sub_path, dtype=str)["stock_code"]
                 .str.strip().str.upper())
        return float((~codes.isin(cat)).mean())
    except Exception:  # noqa: BLE001 — informational only
        return None


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Score a Slate Forensics submission.")
    parser.add_argument("submission", help="path to submission CSV")
    parser.add_argument("--answers", default=os.path.join(
        here, "dataset", "private", "answers.csv"))
    args = parser.parse_args()

    try:
        final, comp = grade(args.submission, args.answers, verbose=True)
    except InvalidSubmissionError as exc:
        print(json.dumps({"error": "invalid_submission", "message": str(exc)}))
        sys.exit(1)

    rate = _ungrounded_rate(args.submission,
                            os.path.join(here, "dataset", "public",
                                         "catalog.csv"))
    if rate is not None:
        comp["ungrounded_item_rate_informational"] = round(rate, 6)
    print(json.dumps(comp, indent=2))


if __name__ == "__main__":
    main()
