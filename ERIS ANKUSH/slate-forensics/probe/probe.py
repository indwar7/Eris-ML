#!/usr/bin/env python3
"""probe/probe.py — sanity probes proving that naive strategies score poorly.

Each probe uses ONLY the public files (plus the grader). None of them touches
hidden generation internals, so none can "accidentally" solve the challenge
through leakage; together they establish the score floor that a real solution
must clear.

Probes
------
  copy_emitted     : declare every slate clean and resubmit the emitted slate
                     (this is exactly sample_submission.csv).
  random_guess     : random flags/modes and 10 random catalog items per slate.
  all_popular      : declare everything corrupted (mode=popularity_fallback)
                     and submit the global top-10 bestsellers as every repair.
  price_rule       : one handcrafted rule — flag a slate when its median item
                     price is far from the customer's median paid price; repair
                     by copying the emitted slate. Shows rules alone cap out.

Usage:
    python probe/probe.py
Writes probe_results.json next to this file. Only pandas / numpy are used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grade import grade  # noqa: E402

PUBLIC = ROOT / "dataset" / "public"
ANSWERS = ROOT / "dataset" / "private" / "answers.csv"
SEED = 20260705
FLOOR = 0.45          # every naive probe must stay below this


def load_public():
    test = pd.read_csv(PUBLIC / "slates_test.csv", dtype=str)
    test["position"] = test["position"].astype(int)
    cat = pd.read_csv(PUBLIC / "catalog.csv", dtype=str)
    tx = pd.read_csv(PUBLIC / "transactions.csv", dtype=str,
                     usecols=["invoice", "stock_code", "quantity",
                              "unit_price", "customer_id", "invoice_date"])
    tx["unit_price"] = pd.to_numeric(tx["unit_price"], errors="coerce")
    tx["quantity"] = pd.to_numeric(tx["quantity"], errors="coerce")
    return test, cat, tx


def as_submission(test: pd.DataFrame, slates: dict[str, list[str]],
                  flags: dict[str, int], modes: dict[str, str]) -> pd.DataFrame:
    ids = sorted(test["slate_id"].unique())
    rows = []
    for sid in ids:
        for pos, code in enumerate(slates[sid], start=1):
            rows.append({"slate_id": sid, "corrupted": flags[sid],
                         "mode": modes[sid], "position": pos,
                         "stock_code": code})
    return pd.DataFrame(rows)


def probe_copy_emitted(test, cat, tx):
    emitted = test.sort_values(["slate_id", "position"]).groupby(
        "slate_id")["stock_code"].agg(list).to_dict()
    ids = emitted.keys()
    return as_submission(test, emitted,
                         {i: 0 for i in ids}, {i: "none" for i in ids})


def probe_random(test, cat, tx):
    rng = np.random.default_rng(SEED)
    codes = cat["stock_code"].to_numpy()
    ids = sorted(test["slate_id"].unique())
    modes_vocab = ["none", "popularity_fallback", "price_band_shift",
                   "stale_index"]
    slates = {i: rng.choice(codes, size=10, replace=False).tolist()
              for i in ids}
    flags = {i: int(rng.random() < 0.5) for i in ids}
    modes = {i: (modes_vocab[rng.integers(1, 4)] if flags[i]
                 else "none") for i in ids}
    return as_submission(test, slates, flags, modes)


def probe_all_popular(test, cat, tx):
    ok = (tx["quantity"] > 0) & (tx["unit_price"] > 0) \
        & ~tx["invoice"].str.startswith("C") \
        & tx["stock_code"].str.match(r"^\d{5}[A-Z]*$")
    pop = (tx[ok].drop_duplicates(["invoice", "stock_code"])
           .groupby("stock_code").size().sort_values(ascending=False))
    top10 = pop.index[:10].tolist()
    ids = sorted(test["slate_id"].unique())
    emitted = test.sort_values(["slate_id", "position"]).groupby(
        "slate_id")["stock_code"].agg(list).to_dict()
    slates = {}
    for i in ids:
        slate = top10[:]
        if slate == emitted[i]:            # keep flag=1 consistent (v2)
            slate = top10[:9] + [pop.index[10]]
        slates[i] = slate
    return as_submission(test, slates, {i: 1 for i in ids},
                         {i: "popularity_fallback" for i in ids})


def probe_price_rule(test, cat, tx):
    """One-feature handcrafted audit: |log(slate price / customer price)|."""
    price = cat.set_index("stock_code")["median_unit_price"].astype(float)
    cust_med = (tx[(tx["quantity"] > 0) & (tx["unit_price"] > 0)
                   & (tx["customer_id"] != "")]
                .groupby("customer_id")["unit_price"].median())
    emitted = test.sort_values(["slate_id", "position"]).groupby(
        "slate_id")["stock_code"].agg(list).to_dict()
    cust_of = test.drop_duplicates("slate_id").set_index(
        "slate_id")["customer_id"].to_dict()
    ids = sorted(emitted.keys())
    flags, modes, slates = {}, {}, {}
    for i in ids:
        slate_price = float(np.nanmedian(
            [price.get(c, np.nan) for c in emitted[i]]))
        c_med = float(cust_med.get(cust_of[i], np.nan))
        ratio = abs(np.log(slate_price / c_med)) if (
            np.isfinite(slate_price) and np.isfinite(c_med) and c_med > 0
        ) else 0.0
        corrupted = int(ratio > 0.9)
        flags[i] = corrupted
        modes[i] = "price_band_shift" if corrupted else "none"
        if corrupted:                      # must alter the slate (v2 guard):
            s = emitted[i][:]              # crude "repair": reverse ranking
            slates[i] = s[::-1]
        else:
            slates[i] = emitted[i]
    return as_submission(test, slates, flags, modes)


def main() -> None:
    if not ANSWERS.exists():
        sys.exit("prepare.py has not been run; no answers to probe against.")
    test, cat, tx = load_public()
    ans = pd.read_csv(ANSWERS, dtype=str)

    probes = {
        "copy_emitted": probe_copy_emitted,
        "random_guess": probe_random,
        "all_popular": probe_all_popular,
        "price_rule": probe_price_rule,
    }
    results = {}
    print(f"{'probe':<15s} {'final':>7s} {'repair':>7s} {'audit':>7s} "
          f"{'mode':>7s} {'cons':>6s}")
    for name, fn in probes.items():
        sub = fn(test, cat, tx)
        final, comp = grade(sub, ans, verbose=True)
        results[name] = comp
        print(f"{name:<15s} {final:7.4f} {comp['S_repair']:7.4f} "
              f"{comp['S_audit']:7.4f} {comp['S_mode']:7.4f} "
              f"{comp['S_consistency']:6.3f}")
        assert final < FLOOR, (
            f"probe {name} scored {final:.4f} >= {FLOOR}: the challenge floor "
            f"is too high — naive strategies must not be competitive.")

    out = Path(__file__).resolve().parent / "probe_results.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nAll probes below the {FLOOR} floor. Results -> {out.name}")


if __name__ == "__main__":
    main()
