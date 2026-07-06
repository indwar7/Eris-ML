#!/usr/bin/env python3
"""prepare.py - split the raw grocery logs into public/private data for the
"Restock Radar: Next-Week Grocery Purchase Ranking" task.

Public  = weeks 0-11 of the interaction log (log-replay duplicates left in,
          as documented), the item catalog, user signup dates, and the full
          promotion calendar (weeks 0-12: promotions are planned ahead, so
          the evaluation week's promo list is legitimately known).
Private = the distinct (user_id, item_id) purchases made during evaluation
          week 12 by the 1,600 target users.

Target users = users with >= 3 distinct (item, week) purchases in the last
four public weeks (8-11) AND >= 1 purchase in evaluation week 12; from that
pool a seeded RNG draws 1,600. Deterministic: byte-identical on every run.
Only pandas / numpy are used (Kaggle Python Docker image).

Usage:
    python prepare.py [--raw dataset/raw] [--public prepared/public]
                      [--private prepared/private]
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260704
WEEK0 = pd.Timestamp("2025-01-06")          # Monday, start of week 0
EVAL_WEEK = 12
EVAL_START = WEEK0 + pd.Timedelta(days=7 * EVAL_WEEK)   # 2025-03-31
N_TARGETS = 1600
TOP_K = 20
RECENT_WEEKS = (8, 11)                       # inclusive window for eligibility


def prepare(raw: Path, public: Path, private: Path) -> None:
    raw, public, private = Path(raw), Path(public), Path(private)
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    inter = pd.read_csv(raw / "interactions.csv")
    items = pd.read_csv(raw / "items.csv")
    users = pd.read_csv(raw / "users.csv")
    promos = pd.read_csv(raw / "promotions.csv")

    ts = pd.to_datetime(inter["timestamp"])
    week = (ts - WEEK0).dt.days // 7

    # ---- public files ----------------------------------------------------
    pub_log = inter.loc[week < EVAL_WEEK]
    pub_log.to_csv(public / "train.csv", index=False)
    items.to_csv(public / "items.csv", index=False)
    users.to_csv(public / "users.csv", index=False)
    promos.to_csv(public / "promotions.csv", index=False)   # incl. week 12

    # ---- evaluation-week purchases & target users -------------------------
    purch = inter.loc[inter["event_type"] == "purchase",
                      ["user_id", "item_id"]]
    pweek = week[purch.index]

    eval_purch = purch.loc[pweek == EVAL_WEEK].drop_duplicates()

    recent = purch.loc[(pweek >= RECENT_WEEKS[0]) & (pweek <= RECENT_WEEKS[1])]
    recent = recent.assign(week=pweek.loc[recent.index]).drop_duplicates()
    recent_counts = recent.groupby("user_id").size()

    eligible = sorted(set(recent_counts[recent_counts >= 3].index)
                      & set(eval_purch["user_id"]))
    rng = np.random.default_rng([SEED, 7])
    targets = sorted(rng.choice(np.array(eligible), size=N_TARGETS,
                                replace=False).tolist())

    test = pd.DataFrame({
        "user_id": targets,
        "predict_week_start": EVAL_START.strftime("%Y-%m-%d"),
    })
    test.to_csv(public / "test.csv", index=False)

    sample = pd.DataFrame({
        "user_id": np.repeat(targets, TOP_K),
        "rank": np.tile(np.arange(1, TOP_K + 1), len(targets)),
        "item_id": np.tile([f"I{r:04d}" for r in range(1, TOP_K + 1)],
                           len(targets)),
    })
    sample.to_csv(public / "sample_submission.csv", index=False)

    answers = (eval_purch[eval_purch["user_id"].isin(set(targets))]
               .sort_values(["user_id", "item_id"]).reset_index(drop=True))
    answers.to_csv(private / "answers.csv", index=False)

    print(f"public/train.csv          {len(pub_log):,} rows")
    print(f"public/test.csv           {len(test):,} target users")
    print(f"public/sample_submission  {len(sample):,} rows")
    print(f"private/answers.csv       {len(answers):,} rows "
          f"({answers['user_id'].nunique()} users, "
          f"{answers.groupby('user_id').size().mean():.2f} items/user)")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=os.path.join(here, "dataset", "raw"))
    parser.add_argument("--public",
                        default=os.path.join(here, "prepared", "public"))
    parser.add_argument("--private",
                        default=os.path.join(here, "prepared", "private"))
    args = parser.parse_args()
    prepare(Path(args.raw), Path(args.public), Path(args.private))


if __name__ == "__main__":
    main()
