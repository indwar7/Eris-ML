#!/usr/bin/env python3
"""generate_dataset.py - synthetic legacy telemetry log generator for the
"Babel Gauge: Legacy Telemetry Log Canonicalization" task.

A fleet-monitoring vendor ingests equipment telemetry from six legacy
logging systems, each with its own undocumented line format, and needs every
line rewritten into one canonical record string. This script draws canonical
ground-truth records first, then renders each through one of six format
renderers (with site-local timestamps, unit conversions, implied decimals,
12-hour clocks, per-format status vocabularies, ...) and corrupts a small
fraction of lines (ingestion damage) whose target is the literal string
REJECT.

Every value is produced by the mechanistic renderers below (numpy RNG draws
+ deterministic string templates). No LLM output is used anywhere, and no
external dataset was copied or used.

Determinism: global seed 20260707; one RNG stream per concern. Running the
script twice produces byte-identical CSVs. Requires only numpy and pandas
(Kaggle Python Docker image).

Outputs (to --out-dir, default ./raw next to this script):
    telemetry_lines.csv  line_id, raw_line, canonical, format_family, split
    sites.csv            site code -> UTC offset (minutes to add to UTC)

Usage:
    python generate_dataset.py [--out-dir raw]
"""

import argparse
import os

import numpy as np
import pandas as pd

SEED = 20260707
N_TRAIN = 60000
N_TEST = 8000
T_MIN = pd.Timestamp("2025-02-01 00:00:00")
T_MAX = pd.Timestamp("2025-05-31 23:59:59")

DEV_TYPES = ["PMP", "CMP", "FAN", "VLV"]          # pump, compressor, fan, valve
TYPE_CHAR = {"PMP": "P", "CMP": "C", "FAN": "F", "VLV": "V"}
STATUSES = ["OK", "WARN", "FAULT"]
STATUS_P = [0.76, 0.16, 0.08]

# per-format status vocabularies (canonical -> rendered)
S_F2 = {"OK": "0", "WARN": "1", "FAULT": "2"}
S_F4 = {"OK": "O", "WARN": "W", "FAULT": "F"}
S_F5 = {"OK": "RUN", "WARN": "WRN", "FAULT": "FLT"}
S_F6 = {"OK": "ok", "WARN": "warn", "FAULT": "fault"}

SITES = [
    ("NE2", -300), ("NE7", -300), ("ATL4", -300), ("MID1", -360),
    ("MID9", -360), ("TEX2", -360), ("DEN3", -420), ("PHX1", -420),
    ("PAC5", -480), ("PAC8", -480), ("SEA2", -480), ("ANC1", -540),
    ("UK1", 0), ("UK4", 0), ("IRL2", 0), ("FRA3", 60), ("GER1", 60),
    ("POL2", 60), ("FIN1", 120), ("ATH2", 120), ("CAI1", 120),
    ("UAE3", 240), ("IND2", 330), ("IND7", 330), ("BKK1", 420),
    ("SIN2", 480), ("PER4", 480), ("JPN1", 540), ("SYD3", 600),
    ("NZL1", 720), ("BRZ2", -180), ("ARG1", -180), ("CHL3", -240),
    ("NFL1", -210), ("HAW2", -600), ("ICE1", 0),
]
FMT_TRAIN_P = {"F1": .28, "F2": .22, "F3": .20, "F4": .18, "F5": .105, "F6": .015}
FMT_TEST_P = {"F1": .20, "F2": .18, "F3": .16, "F4": .14, "F5": .12, "F6": .20}
CORRUPT_TRAIN = 0.04
CORRUPT_TEST = 0.05

STRUCT_CHARS = set("|<>{}:=@,\"")


def canonical_string(ts_utc, dtype, dnum, site, tc, pk, status):
    return (f"ts={ts_utc.strftime('%Y-%m-%dT%H:%M:%S')}Z|dev={dtype}-{dnum:04d}"
            f"|site={site}|temp_c={tc:.1f}|pres_kpa={pk:.1f}|status={status}")


def f_to_c_render(tc):
    return round(tc * 9 / 5 + 32, 1)


def kpa_to_psi_render(pk):
    return round(pk / 6.894757, 2)


def render(fmt, ts_utc, local, dtype, dnum, site, tc, pk, status, r):
    if fmt == "F1":      # Vendor A gateway; site-local time, imperial units
        return (f"{local.strftime('%Y-%m-%d %H:%M:%S')} | {dtype}-{dnum:04d} "
                f"@ {site} | T={f_to_c_render(tc)}F P={kpa_to_psi_render(pk)}psi "
                f"| {status}")
    if fmt == "F2":      # Vendor B bus dump; UTC compact, numeric status
        return (f"<{TYPE_CHAR[dtype]}{dnum:04d}|{site}|"
                f"{ts_utc.strftime('%Y%m%d%H%M%S')}|t:{tc:.1f}|p:{pk:.1f}|"
                f"s:{S_F2[status]}>")
    if fmt == "F3":      # key=value exporter; site-local DD/MM, shuffled keys
        toks = [f"dev={dtype}{dnum:04d}", f"site={site}",
                f"time={local.strftime('%d/%m/%Y %H:%M')}",
                f"temp={tc:.1f}C", f"pressure={pk:.1f}kPa", f"flag={status}"]
        order = r.permutation(6)
        return " ".join(toks[i] for i in order)
    if fmt == "F4":      # fixed-width mainframe; implied decimals, local time
        return (f"{TYPE_CHAR[dtype]}{dnum:04d}{site:<5}"
                f"{local.strftime('%y%m%d')}{local.strftime('%H%M')}"
                f"{int(round(tc * 10)):+05d}{int(round(pk * 10)):06d}"
                f"{S_F4[status]}")
    if fmt == "F5":      # US regional CSV; MM/DD/YY 12-hour, imperial units
        return (f"{site},{dtype}-{dnum:04d},"
                f"{local.strftime('%m/%d/%y %I:%M%p')},"
                f"{f_to_c_render(tc)},{kpa_to_psi_render(pk)},{S_F5[status]}")
    if fmt == "F6":      # modern collector; JSON with epoch seconds (UTC)
        epoch = int(ts_utc.value // 10**9)
        return ('{"st":"%s","d":"%s-%04d","e":%d,"tc":%.1f,"pk":%.1f,"x":"%s"}'
                % (site, dtype, dnum, epoch, tc, pk, S_F6[status]))
    raise ValueError(fmt)


def corrupt(line, r):
    """Ingestion damage: drop 2-4 structural chars + 1-4 arbitrary chars,
    then garble one remaining char. Always breaks the line's grammar."""
    chars = list(line)
    struct_pos = [i for i, ch in enumerate(chars) if ch in STRUCT_CHARS]
    n_struct = min(len(struct_pos), int(r.integers(2, 5)))
    drop = set(r.choice(struct_pos, n_struct, replace=False).tolist())
    others = [i for i in range(len(chars)) if i not in drop]
    n_other = int(r.integers(1, 5))
    drop |= set(r.choice(others, min(n_other, len(others)),
                         replace=False).tolist())
    kept = [ch for i, ch in enumerate(chars) if i not in drop]
    if kept:
        j = int(r.integers(0, len(kept)))
        kept[j] = chr(ord("!") + int(r.integers(0, 14)))
    return "".join(kept)


def make_split(n, fmt_probs, corrupt_frac, dnum_lo, dnum_hi, rng, id_start):
    site_codes = [s for s, _ in SITES]
    offset = dict(SITES)
    fmts = list(fmt_probs)
    probs = np.array([fmt_probs[f] for f in fmts])
    probs = probs / probs.sum()

    rows = []
    span = int((T_MAX - T_MIN).total_seconds())
    for k in range(n):
        r = np.random.default_rng([SEED, id_start + k])
        fmt = fmts[int(r.choice(len(fmts), p=probs))]
        sec = int(r.integers(0, span))
        ts = T_MIN + pd.Timedelta(seconds=sec)
        if fmt in ("F3", "F4", "F5"):       # these formats carry no seconds
            ts = ts.floor("min")
        site = site_codes[int(r.integers(0, len(site_codes)))]
        local = ts + pd.Timedelta(minutes=offset[site])
        dtype = DEV_TYPES[int(r.integers(0, len(DEV_TYPES)))]
        dnum = int(r.integers(dnum_lo, dnum_hi))
        tc = round(float(r.uniform(-5.0, 95.0)), 1)
        if tc == 0.0:
            tc = 0.0          # normalize -0.0 so canonical == renderings
        pk = round(float(r.uniform(80.0, 900.0)), 1)
        status = STATUSES[int(r.choice(3, p=STATUS_P))]

        raw = render(fmt, ts, local, dtype, dnum, site, tc, pk, status, r)
        canon = canonical_string(ts, dtype, dnum, site, tc, pk, status)
        if r.random() < corrupt_frac:
            raw = corrupt(raw, r)
            canon = "REJECT"
        rows.append((raw, canon, fmt))
    return rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=os.path.join(here, "raw"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng([SEED, 0])
    train = make_split(N_TRAIN, FMT_TRAIN_P, CORRUPT_TRAIN,
                       1, 7000, rng, 1_000_000)
    test = make_split(N_TEST, FMT_TEST_P, CORRUPT_TEST,
                      7000, 10000, rng, 2_000_000)

    all_rows = [("train",) + t for t in train] + [("test",) + t for t in test]
    df = pd.DataFrame(all_rows,
                      columns=["split", "raw_line", "canonical",
                               "format_family"])
    df.insert(0, "line_id", [f"L{i + 1:06d}" for i in range(len(df))])
    df = df[["line_id", "raw_line", "canonical", "format_family", "split"]]
    df.to_csv(os.path.join(args.out_dir, "telemetry_lines.csv"), index=False)

    sites = pd.DataFrame(SITES, columns=["site", "utc_offset_min"])
    sites.to_csv(os.path.join(args.out_dir, "sites.csv"), index=False)

    print(f"lines: {len(df):,} ({N_TRAIN:,} train / {N_TEST:,} test)")
    print("train format mix:\n",
          df[df.split == "train"].format_family.value_counts(
              normalize=True).round(4).to_string())
    print("test format mix:\n",
          df[df.split == "test"].format_family.value_counts(
              normalize=True).round(4).to_string())
    print("REJECT rate train:",
          round((df[df.split == 'train'].canonical == 'REJECT').mean(), 4),
          "test:",
          round((df[df.split == 'test'].canonical == 'REJECT').mean(), 4))


if __name__ == "__main__":
    main()
