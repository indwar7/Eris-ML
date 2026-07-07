#!/usr/bin/env python3
"""probe.py - naive baseline for "Unfluke: Skill-vs-Luck Forensics for
Systematic Trading Records".

The baseline an allocator should NOT be: it trusts headline in-sample
performance. Using only public files (no hidden labels, no leakage), it

  - ranks strategies by in-sample total trade return,
  - sets p_skill to that rank (percentile),
  - predicts oos_sharpe as a crude annualized trade-Sharpe proxy,
  - selects the top 240 by the same ranking (respecting the consistency
    rules so no penalty applies).

Because lucky look-alikes are matched to the skilled cohort on exactly
these headline statistics, this baseline plateaus far below the
reference solution — which is the point being demonstrated.

Usage:
    python probe/probe.py

Writes probe/probe_submission.csv and, if the private answer key is
available locally, prints the graded score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "dataset" / "public"
ANSWERS = ROOT / "dataset" / "private" / "answers.csv"
OUT = Path(__file__).resolve().parent / "probe_submission.csv"
K_SELECT = 240


def main() -> None:
    test = pd.read_csv(PUBLIC / "test.csv")
    trades = pd.read_csv(PUBLIC / "trades.csv")

    g = trades.groupby("strategy_id")["trade_return"]
    stats = pd.DataFrame({
        "total_ret": g.sum(),
        "mean_ret": g.mean(),
        "std_ret": g.std().fillna(0.0),
        "n": g.size(),
    })
    # crude annualized trade-level Sharpe proxy (naive on purpose)
    stats["sharpe_proxy"] = np.where(
        stats["std_ret"] > 0,
        stats["mean_ret"] / stats["std_ret"] * np.sqrt(stats["n"]),
        0.0,
    ).clip(-10, 10)

    sub = test[["strategy_id"]].merge(
        stats, left_on="strategy_id", right_index=True, how="left")
    sub[["total_ret", "sharpe_proxy"]] = \
        sub[["total_ret", "sharpe_proxy"]].fillna(0.0)

    sub["p_skill"] = sub["total_ret"].rank(pct=True)
    sub["oos_sharpe"] = sub["sharpe_proxy"].round(4)

    # top-K by the same naive ranking, respecting the consistency rules
    ok = (sub["p_skill"] >= 0.2) & (sub["oos_sharpe"] >= 0)
    picks = sub.loc[ok].sort_values(
        "total_ret", ascending=False).index[:K_SELECT]
    sub["select"] = 0
    sub.loc[picks, "select"] = 1
    assert int(sub["select"].sum()) == K_SELECT

    out = sub[["strategy_id", "p_skill", "oos_sharpe", "select"]]
    out.to_csv(OUT, index=False)
    print(f"[ok] wrote {OUT} ({len(out)} rows)")

    if ANSWERS.exists():
        sys.path.insert(0, str(ROOT))
        from grade import grade  # noqa: PLC0415
        score = grade(str(OUT), str(ANSWERS))
        import json  # noqa: PLC0415
        print(json.dumps(grade.last_report, indent=2))
        print(f"[naive baseline score] {score:.4f} "
              "(compare: sample ~0.03, reference solution ~0.76)")
    else:
        print("[info] private answers not available; submission written "
              "but not scored.")


if __name__ == "__main__":
    main()
