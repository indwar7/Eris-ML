#!/usr/bin/env python3
"""prepare.py - transform the raw Coolant Pump Fleet Telemetry dataset into
the public / private splits for the "Silent Degradation" task.

Split design
------------
* PUBLIC  train.csv            : full hourly streams of the 100 pilot-cohort
                                 pumps, with the per-row binary label
                                 `failure_within_48h` (1 if a failure event
                                 occurs in the 48 hours after the row's
                                 timestamp). The last 48 hours of each stream
                                 are dropped because their labels would be
                                 truncated by the end of the record.
* PUBLIC  test_features.csv    : 800 evaluation snapshots, one per rollout-
                                 cohort pump. Each snapshot is the 168-hour
                                 (7-day) telemetry window ending at the
                                 prediction origin. Nothing after the origin
                                 is included anywhere in the public split, so
                                 the private ground truth cannot be recovered
                                 from public data.
* PUBLIC  sample_submission.csv: correct format, constant prediction.
* PRIVATE answers.csv          : snapshot_id -> failure_within_48h.

Snapshot origins are sampled deterministically (fixed-seed RNG):
~15% of snapshots have their origin inside the 48 hours preceding a real
failure (label 1), the rest are sampled from operating periods at least 48
hours away from any failure (label 0). One snapshot per pump, so no snapshot
window can reveal another snapshot's outcome.

Determinism: running this script twice produces byte-identical outputs.
Only numpy + pandas are used (Kaggle Python Docker image).

Usage:
    python prepare.py [--raw-dir dataset/raw] [--out-dir prepared]
"""

import argparse
import os

import numpy as np
import pandas as pd

SEED = 424242                 # snapshot-sampling seed (fixed => deterministic)
HORIZON = 48                  # prediction horizon, hours
WINDOW = 168                  # snapshot history window, hours
TARGET_POSITIVES = 120        # ~15% of 800 snapshots
LABEL_COL = "failure_within_48h"

FEATURE_COLS = [
    "vibration_mm_s",
    "bearing_temp_c",
    "discharge_pressure_kpa",
    "flow_rate_m3_h",
    "motor_current_a",
    "rpm",
]


def hour_index(ts, start):
    return ((ts - start) / pd.Timedelta(hours=1)).astype(int)


def load_raw(raw_dir):
    readings = pd.read_csv(os.path.join(raw_dir, "sensor_readings.csv"),
                           parse_dates=["timestamp"])
    failures = pd.read_csv(os.path.join(raw_dir, "failure_log.csv"),
                           parse_dates=["failure_time"])
    meta = pd.read_csv(os.path.join(raw_dir, "pump_metadata.csv"),
                       parse_dates=["first_record", "last_record"])
    return readings, failures, meta


def label_rows(hours, fail_hours, horizon=HORIZON):
    """Binary label per row: 1 iff any failure in (t, t + horizon]."""
    labels = np.zeros(len(hours), dtype=np.int64)
    for t_fail in fail_hours:
        labels |= (hours < t_fail) & (t_fail <= hours + horizon)
    return labels


def build_train(readings, failures, meta):
    pilot_meta = meta[meta["cohort"] == "pilot"]
    pilot_ids = set(pilot_meta["pump_id"])
    df = readings[readings["pump_id"].isin(pilot_ids)].copy()
    start = df["timestamp"].min()
    df["t"] = hour_index(df["timestamp"], start)
    n_hours = int(df["t"].max()) + 1

    fail_by_pump = {
        pid: hour_index(grp["failure_time"], start).to_numpy()
        for pid, grp in failures[failures["pump_id"].isin(pilot_ids)]
        .groupby("pump_id")
    }

    parts = []
    for pid, grp in df.groupby("pump_id", sort=True):
        grp = grp.sort_values("t")
        grp[LABEL_COL] = label_rows(grp["t"].to_numpy(),
                                    fail_by_pump.get(pid, np.array([])))
        parts.append(grp)
    train = pd.concat(parts, ignore_index=True)

    # drop rows whose 48h label window is truncated by the end of the record
    train = train[train["t"] < n_hours - HORIZON].copy()

    train = train.merge(pilot_meta[["pump_id", "pump_model", "site"]],
                        on="pump_id", how="left")
    cols = (["pump_id", "timestamp", "pump_model", "site"]
            + FEATURE_COLS + ["maintenance_flag", LABEL_COL])
    train = train[cols].sort_values(["pump_id", "timestamp"]).reset_index(drop=True)
    return train


def build_snapshots(readings, failures, meta):
    rng = np.random.default_rng(SEED)
    rollout_meta = meta[meta["cohort"] == "rollout"]
    rollout_ids = sorted(rollout_meta["pump_id"])
    df = readings[readings["pump_id"].isin(set(rollout_ids))].copy()
    start = df["timestamp"].min()
    df["t"] = hour_index(df["timestamp"], start)
    n_hours = int(df["t"].max()) + 1

    roll_fail = failures[failures["pump_id"].isin(set(rollout_ids))].copy()
    roll_fail["t_fail"] = hour_index(roll_fail["failure_time"], start)
    fails = {pid: [] for pid in rollout_ids}
    for row in roll_fail.itertuples(index=False):
        fails[row.pump_id].append((int(row.t_fail), int(row.downtime_hours)))

    lo, hi = WINDOW - 1, n_hours - HORIZON - 1   # valid origin range [167, 431]

    def operational(pid, h):
        return all(not (tf <= h < tf + dt) for tf, dt in fails[pid])

    def failure_ahead(pid, h):
        return any(h < tf <= h + HORIZON for tf, dt in fails[pid])

    pos_cand, neg_cand = {}, {}
    for pid in rollout_ids:
        hours = np.arange(lo, hi + 1)
        ok = np.array([operational(pid, h) for h in hours])
        ahead = np.array([failure_ahead(pid, h) for h in hours])
        pos_cand[pid] = hours[ok & ahead]
        neg_cand[pid] = hours[ok & ~ahead]

    eligible_pos = [pid for pid in rollout_ids if len(pos_cand[pid]) > 0]
    n_pos = min(TARGET_POSITIVES, len(eligible_pos))
    pos_pumps = set(rng.choice(np.array(eligible_pos), size=n_pos, replace=False))

    origins, labels = {}, {}
    dropped = []
    for pid in rollout_ids:
        if pid in pos_pumps:
            origins[pid] = int(rng.choice(pos_cand[pid]))
            labels[pid] = 1
        elif len(neg_cand[pid]) > 0:
            origins[pid] = int(rng.choice(neg_cand[pid]))
            labels[pid] = 0
        elif len(pos_cand[pid]) > 0:   # pump is near-continuously failing
            origins[pid] = int(rng.choice(pos_cand[pid]))
            labels[pid] = 1
        else:                          # no valid origin at all (not expected)
            dropped.append(pid)
    if dropped:
        print(f"  WARNING: dropped {len(dropped)} pumps with no valid origin: "
              f"{dropped[:5]}...")

    kept = [pid for pid in rollout_ids if pid in origins]
    snap_ids = {pid: f"snap_{i + 1:04d}" for i, pid in enumerate(kept)}

    model_site = rollout_meta.set_index("pump_id")[["pump_model", "site"]]
    windows = []
    for pid in kept:
        h = origins[pid]
        w = df[(df["pump_id"] == pid) & (df["t"] >= h - WINDOW + 1)
               & (df["t"] <= h)].copy()
        assert len(w) == WINDOW, (pid, h, len(w))
        w.insert(0, "snapshot_id", snap_ids[pid])
        w["pump_model"] = model_site.loc[pid, "pump_model"]
        w["site"] = model_site.loc[pid, "site"]
        windows.append(w)
    test_features = pd.concat(windows, ignore_index=True)
    cols = (["snapshot_id", "pump_id", "timestamp", "pump_model", "site"]
            + FEATURE_COLS + ["maintenance_flag"])
    test_features = (test_features[cols]
                     .sort_values(["snapshot_id", "timestamp"])
                     .reset_index(drop=True))

    test_labels = pd.DataFrame(
        {"snapshot_id": [snap_ids[pid] for pid in kept],
         LABEL_COL: [labels[pid] for pid in kept]}
    ).sort_values("snapshot_id").reset_index(drop=True)
    return test_features, test_labels


def prepare(raw, public, private):
    """Platform entry point: transform the raw dataset in `raw` into the
    public split in `public` and the private split in `private`.

    Accepts str or pathlib.Path directories; creates the output directories
    if needed.
    """
    raw_dir, public_dir, private_dir = str(raw), str(public), str(private)
    os.makedirs(public_dir, exist_ok=True)
    os.makedirs(private_dir, exist_ok=True)

    print("loading raw data ...")
    readings, failures, meta = load_raw(raw_dir)

    print("building train split (pilot cohort) ...")
    train = build_train(readings, failures, meta)

    print("building evaluation snapshots (rollout cohort) ...")
    test_features, test_labels = build_snapshots(readings, failures, meta)

    sample = test_labels[["snapshot_id"]].copy()
    sample["prediction"] = 0.5

    ts_fmt = "%Y-%m-%d %H:%M:%S"
    train.to_csv(os.path.join(public_dir, "train.csv"),
                 index=False, date_format=ts_fmt)
    test_features.to_csv(os.path.join(public_dir, "test_features.csv"),
                         index=False, date_format=ts_fmt)
    sample.to_csv(os.path.join(public_dir, "sample_submission.csv"), index=False)
    test_labels.to_csv(os.path.join(private_dir, "answers.csv"), index=False)

    pos_rate_train = train[LABEL_COL].mean()
    print(f"train.csv           : {len(train):,} rows, "
          f"positive rate {pos_rate_train:.4f}")
    print(f"test_features.csv   : {len(test_features):,} rows, "
          f"{test_features['snapshot_id'].nunique()} snapshots")
    print(f"answers.csv         : {len(test_labels):,} snapshots, "
          f"{int(test_labels[LABEL_COL].sum())} positives "
          f"({test_labels[LABEL_COL].mean():.3f})")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=os.path.join(here, "dataset", "raw"))
    parser.add_argument("--out-dir", default=os.path.join(here, "prepared"))
    args = parser.parse_args()
    prepare(args.raw_dir,
            os.path.join(args.out_dir, "public"),
            os.path.join(args.out_dir, "private"))
