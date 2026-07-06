#!/usr/bin/env python3
"""Generate the synthetic "Coolant Pump Fleet Telemetry" dataset.

Produces three raw CSV files in ./raw/ :
    sensor_readings.csv  - hourly telemetry for every pump
    failure_log.csv      - ops record of every failure event (trip time, mode, downtime)
    pump_metadata.csv    - static pump attributes (model, site, cohort)

The simulation is fully deterministic: every pump gets its own child RNG seeded
as (GLOBAL_SEED, pump_index), so results do not depend on iteration order.
Running this script twice produces byte-identical outputs.

No LLM was used to produce any data value in this dataset: all values come from
the mechanistic model below (numpy RNG draws + deterministic signal equations).

Usage:
    python generate_dataset.py [--out-dir raw]

Requires only numpy + pandas (both in the Kaggle Python Docker image).
"""

import argparse
import os

import numpy as np
import pandas as pd

GLOBAL_SEED = 20260703

# ---------------------------------------------------------------- cohorts ---
# Pilot cohort: long-horizon monitoring pilot (rich history, incident reviews done).
# Rollout cohort: fleet-wide rollout; only the first 20 days of data exist and the
# reliability team has not yet performed post-incident reviews for this period
# (hence maintenance_flag == 0 everywhere in the rollout cohort).
PILOT_PUMPS = 100
PILOT_HOURS = 2160                      # 90 days
PILOT_START = pd.Timestamp("2025-01-06 00:00:00")

ROLLOUT_PUMPS = 800
ROLLOUT_HOURS = 480                     # 20 days
ROLLOUT_START = pd.Timestamp("2025-04-07 00:00:00")

# ---------------------------------------------------------------- physics ---
# Baseline operating points per pump model.
MODELS = {
    #            vib   temp  press  flow   curr   rpm
    "HX-300": (2.60, 58.0, 380.0, 95.0, 32.0, 1465.0),
    "HX-500": (3.20, 63.0, 430.0, 118.0, 39.0, 1478.0),
    "MV-90":  (2.10, 55.0, 350.0, 76.0, 27.0, 1452.0),
}
MODEL_NAMES = list(MODELS.keys())
MODEL_PROBS = [0.40, 0.35, 0.25]

SITES = ["SITE-A", "SITE-B", "SITE-C"]
SITE_PROBS = [0.45, 0.35, 0.20]
SITE_AMBIENT_OFFSET = {"SITE-A": 0.0, "SITE-B": 1.5, "SITE-C": -1.0}

# Failure process (renewal process per pump).
MTBF_HOURS = 1100.0                     # mean time between failures (exponential)
MIN_TTF_HOURS = 24                      # never fail within 24h of (re)start
P_WEAR = 0.72                           # wear-out failure (has degradation ramp)
RAMP_HOURS_RANGE = (72.0, 216.0)        # wear ramp duration
RAMP_EXPONENT = 1.6                     # severity = progress**1.6 (accelerating)
DOWNTIME_RANGE = (24, 72)               # repair downtime, hours

# Degradation effect at full severity (s = 1).
VIB_GAIN = 1.00                         # vibration multiplied by (1 + s * VIB_GAIN)
TEMP_GAIN = 14.0                        # bearing temp += 14 degC
PRESS_LOSS = 0.10                       # discharge pressure * (1 - 0.10 s)
FLOW_LOSS = 0.14                        # flow * (1 - 0.14 s)
CURR_GAIN = 0.08                        # motor current * (1 + 0.08 s)
RPM_LOSS = 9.0                          # rpm -= 9 s

# Benign transients (process upsets / heat waves): raise temp & vibration only,
# then fully recover. They mimic the early part of a wear ramp on those two
# sensors and exist to defeat naive univariate thresholding.
BENIGN_MEAN_INTERARRIVAL = 400.0        # hours
BENIGN_DURATION_RANGE = (8.0, 36.0)
BENIGN_TEMP_RANGE = (5.0, 11.0)         # peak degC added
BENIGN_VIB_RANGE = (0.30, 0.70)         # peak fractional vibration increase

# Sensor dropout: outage bursts recorded as the sentinel value -999.0.
SENTINEL = -999.0
OUTAGE_MEAN_INTERARRIVAL = 450.0        # hours, per sensor channel
OUTAGE_LEN_RANGE = (2, 8)               # burst length, hours (inclusive)

# Retroactive annotation: during post-incident review the reliability team tags
# the 48 hours immediately preceding each failure. Reviews exist only for the
# pilot cohort (the rollout period has not been reviewed yet).
REVIEW_WINDOW_HOURS = 48

SENSOR_COLS = [
    "vibration_mm_s",
    "bearing_temp_c",
    "discharge_pressure_kpa",
    "flow_rate_m3_h",
    "motor_current_a",
    "rpm",
]
ROUNDING = {
    "vibration_mm_s": 3,
    "bearing_temp_c": 2,
    "discharge_pressure_kpa": 1,
    "flow_rate_m3_h": 2,
    "motor_current_a": 2,
    "rpm": 1,
}


def simulate_failure_schedule(rng, n_hours):
    """Draw failure times, modes, ramps and downtime for one pump.

    Returns a list of dicts with keys: t_fail, mode, ramp_start, downtime.
    All times are integer hour indices in [0, n_hours).
    """
    events = []
    t = 0  # pump (re)starts operational at t
    while True:
        ttf = max(float(rng.exponential(MTBF_HOURS)), float(MIN_TTF_HOURS))
        t_fail = int(round(t + ttf))
        if t_fail >= n_hours:
            break
        mode = "wear_out" if rng.random() < P_WEAR else "sudden"
        if mode == "wear_out":
            ramp = float(rng.uniform(*RAMP_HOURS_RANGE))
            ramp_start = max(t, int(round(t_fail - ramp)))
        else:
            ramp_start = t_fail  # no ramp
        downtime = int(rng.integers(DOWNTIME_RANGE[0], DOWNTIME_RANGE[1] + 1))
        events.append(
            {"t_fail": t_fail, "mode": mode, "ramp_start": ramp_start,
             "downtime": downtime}
        )
        t = t_fail + downtime  # next operational start
        if t >= n_hours:
            break
    return events


def severity_series(events, n_hours):
    """Severity s(t) in [0, 1] from wear ramps; 0 elsewhere."""
    s = np.zeros(n_hours)
    for ev in events:
        if ev["mode"] != "wear_out":
            continue
        a, b = ev["ramp_start"], ev["t_fail"]  # ramp over [a, b)
        if b <= a:
            continue
        idx = np.arange(a, min(b, n_hours))
        progress = (idx - a + 1) / float(b - a)
        s[idx] = np.maximum(s[idx], progress ** RAMP_EXPONENT)
    return s


def downtime_mask(events, n_hours):
    m = np.zeros(n_hours, dtype=bool)
    for ev in events:
        m[ev["t_fail"]: min(ev["t_fail"] + ev["downtime"], n_hours)] = True
    return m


def benign_series(rng, n_hours):
    """Additive temp bump and fractional vibration bump from benign transients."""
    temp_bump = np.zeros(n_hours)
    vib_bump = np.zeros(n_hours)
    t = float(rng.exponential(BENIGN_MEAN_INTERARRIVAL))
    while t < n_hours:
        dur = float(rng.uniform(*BENIGN_DURATION_RANGE))
        peak_temp = float(rng.uniform(*BENIGN_TEMP_RANGE))
        peak_vib = float(rng.uniform(*BENIGN_VIB_RANGE))
        start = int(round(t))
        end = min(int(round(t + dur)), n_hours)
        if end > start:
            idx = np.arange(start, end)
            # triangular profile: ramp up to the peak at the midpoint, back down
            frac = (idx - start + 1) / float(end - start)
            profile = 1.0 - np.abs(2.0 * frac - 1.0)
            temp_bump[idx] = np.maximum(temp_bump[idx], peak_temp * profile)
            vib_bump[idx] = np.maximum(vib_bump[idx], peak_vib * profile)
        t = t + dur + float(rng.exponential(BENIGN_MEAN_INTERARRIVAL))
    return temp_bump, vib_bump


def outage_mask(rng, n_hours):
    m = np.zeros(n_hours, dtype=bool)
    t = float(rng.exponential(OUTAGE_MEAN_INTERARRIVAL))
    while t < n_hours:
        length = int(rng.integers(OUTAGE_LEN_RANGE[0], OUTAGE_LEN_RANGE[1] + 1))
        start = int(round(t))
        m[start: min(start + length, n_hours)] = True
        t = t + length + float(rng.exponential(OUTAGE_MEAN_INTERARRIVAL))
    return m


def simulate_pump(pump_index, pump_id, cohort, n_hours, start_ts):
    """Simulate one pump. Returns (readings_df, failure_rows, meta_row)."""
    rng = np.random.default_rng([GLOBAL_SEED, pump_index])

    model = MODEL_NAMES[int(rng.choice(len(MODEL_NAMES), p=MODEL_PROBS))]
    site = SITES[int(rng.choice(len(SITES), p=SITE_PROBS))]
    base_vib, base_temp, base_press, base_flow, base_curr, base_rpm = MODELS[model]

    # Per-pump baseline spread (foundation, alignment, impeller wear state,
    # VFD setpoint, duty point). Deliberately wide: absolute readings overlap
    # heavily across healthy and degrading pumps, so detecting degradation
    # requires reasoning relative to each pump's own recent baseline.
    base_vib *= float(np.exp(rng.normal(0.0, 0.45)))
    base_temp += rng.normal(0.0, 6.5)
    base_press *= float(np.exp(rng.normal(0.0, 0.14)))
    base_flow *= float(np.exp(rng.normal(0.0, 0.16)))
    base_curr *= float(np.exp(rng.normal(0.0, 0.14)))
    base_rpm += rng.normal(0.0, 16.0)

    events = simulate_failure_schedule(rng, n_hours)
    s = severity_series(events, n_hours)
    down = downtime_mask(events, n_hours)
    temp_bump, vib_bump = benign_series(rng, n_hours)

    t = np.arange(n_hours)
    ts = start_ts + pd.to_timedelta(t, unit="h")
    hour_of_day = (start_ts.hour + t) % 24
    abs_hour = t + start_ts.dayofyear * 24  # phase anchor for weekly cycle

    diurnal = np.sin(2.0 * np.pi * (hour_of_day - 15.0) / 24.0)
    weekly_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    daily_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    load = (1.0
            + 0.05 * np.sin(2.0 * np.pi * abs_hour / 168.0 + weekly_phase)
            + 0.03 * np.sin(2.0 * np.pi * abs_hour / 24.0 + daily_phase))

    ambient = 21.0 + SITE_AMBIENT_OFFSET[site]

    vib = base_vib * (1.0 + VIB_GAIN * s) * (1.0 + vib_bump) \
        + rng.normal(0.0, 0.16, n_hours)
    temp = (base_temp + SITE_AMBIENT_OFFSET[site] + 2.5 * diurnal
            + TEMP_GAIN * s + temp_bump + rng.normal(0.0, 0.9, n_hours))
    press = base_press * (1.0 - PRESS_LOSS * s) * (1.0 + 0.02 * (load - 1.0)) \
        + rng.normal(0.0, 5.0, n_hours)
    flow = base_flow * (1.0 - FLOW_LOSS * s) * load + rng.normal(0.0, 2.2, n_hours)
    curr = base_curr * (1.0 + CURR_GAIN * s) * (0.6 + 0.4 * load) \
        + rng.normal(0.0, 0.6, n_hours)
    rpm = base_rpm - RPM_LOSS * s + rng.normal(0.0, 2.0, n_hours)

    # downtime overrides: pump stopped, sensors read a stopped machine
    vib[down] = np.abs(rng.normal(0.03, 0.02, int(down.sum())))
    temp[down] = ambient + rng.normal(0.0, 1.5, int(down.sum()))
    press[down] = 12.0 + rng.normal(0.0, 2.0, int(down.sum()))
    flow[down] = 0.0
    curr[down] = 0.0
    rpm[down] = 0.0

    vib = np.clip(vib, 0.0, None)
    flow = np.clip(flow, 0.0, None)
    curr = np.clip(curr, 0.0, None)
    rpm = np.clip(rpm, 0.0, None)

    # retroactive review flag (pilot cohort only)
    flag = np.zeros(n_hours, dtype=np.int64)
    if cohort == "pilot":
        for ev in events:
            flag[max(0, ev["t_fail"] - REVIEW_WINDOW_HOURS): ev["t_fail"]] = 1

    values = {
        "vibration_mm_s": vib,
        "bearing_temp_c": temp,
        "discharge_pressure_kpa": press,
        "flow_rate_m3_h": flow,
        "motor_current_a": curr,
        "rpm": rpm,
    }
    # sensor outages -> sentinel, drawn independently per channel
    for col in SENSOR_COLS:
        out = outage_mask(rng, n_hours)
        values[col] = np.where(out, SENTINEL, values[col])
        values[col] = np.round(values[col], ROUNDING[col])

    readings = pd.DataFrame({"pump_id": pump_id, "timestamp": ts, **values,
                             "maintenance_flag": flag})

    failure_rows = [
        {"pump_id": pump_id,
         "failure_time": start_ts + pd.Timedelta(hours=ev["t_fail"]),
         "failure_mode": ev["mode"],
         "downtime_hours": ev["downtime"]}
        for ev in events
    ]
    meta_row = {"pump_id": pump_id, "pump_model": model, "site": site,
                "cohort": cohort,
                "first_record": ts[0], "last_record": ts[-1]}
    return readings, failure_rows, meta_row


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    specs = []
    for i in range(PILOT_PUMPS):
        specs.append((i, f"P{i + 1:04d}", "pilot", PILOT_HOURS, PILOT_START))
    for j in range(ROLLOUT_PUMPS):
        specs.append((PILOT_PUMPS + j, f"R{j + 1:04d}", "rollout",
                      ROLLOUT_HOURS, ROLLOUT_START))

    all_readings, all_failures, all_meta = [], [], []
    for idx, (pump_index, pump_id, cohort, n_hours, start_ts) in enumerate(specs):
        readings, failures, meta = simulate_pump(
            pump_index, pump_id, cohort, n_hours, start_ts)
        all_readings.append(readings)
        all_failures.extend(failures)
        all_meta.append(meta)
        if (idx + 1) % 200 == 0:
            print(f"  simulated {idx + 1}/{len(specs)} pumps")

    readings = pd.concat(all_readings, ignore_index=True)
    readings = readings.sort_values(["pump_id", "timestamp"]).reset_index(drop=True)

    failures = pd.DataFrame(
        all_failures,
        columns=["pump_id", "failure_time", "failure_mode", "downtime_hours"])
    failures = failures.sort_values(["pump_id", "failure_time"]).reset_index(drop=True)

    meta = pd.DataFrame(all_meta).sort_values("pump_id").reset_index(drop=True)

    readings.to_csv(os.path.join(out_dir, "sensor_readings.csv"), index=False)
    failures.to_csv(os.path.join(out_dir, "failure_log.csv"), index=False)
    meta.to_csv(os.path.join(out_dir, "pump_metadata.csv"), index=False)

    n_fail_pilot = (failures["pump_id"].str.startswith("P")).sum()
    n_fail_roll = (failures["pump_id"].str.startswith("R")).sum()
    print(f"sensor_readings.csv : {len(readings):,} rows")
    print(f"failure_log.csv     : {len(failures):,} failures "
          f"(pilot {n_fail_pilot}, rollout {n_fail_roll})")
    print(f"pump_metadata.csv   : {len(meta):,} pumps")
    print(f"sentinel (-999) share per sensor column: "
          f"{[(c, round(float((readings[c] == SENTINEL).mean()), 4)) for c in SENSOR_COLS]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "raw"))
    args = parser.parse_args()
    main(args.out_dir)
