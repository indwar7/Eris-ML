# Synthetic Coolant Pump Fleet Telemetry — Hourly Condition Monitoring & Failure Log for 48-Hour Failure-Risk Prediction

Hourly condition-monitoring telemetry for a fleet of 900 industrial coolant
pumps, with a ground-truth failure log. Built for the **"Silent Degradation:
48-Hour Pump Failure Risk"** task: predicting whether a pump will fail
within the next 48 hours from its recent sensor history.

---

## 1. Provenance & license

- **Type:** fully synthetic. Every value is produced by the mechanistic
  simulation in [`generate_dataset.py`](generate_dataset.py) (numpy RNG draws
  + deterministic signal equations). **No LLM output is used anywhere in the
  data**, and no Kaggle competition or external dataset was copied or used.
- **Reproducibility:** the generator is fully deterministic (global seed
  `20260703`; each pump uses a child RNG seeded `(20260703, pump_index)` so
  output does not depend on iteration order). Running
  `python generate_dataset.py` twice produces **byte-identical** CSVs.
  Requires only `numpy` and `pandas` (Kaggle Python Docker image).
- **License:** CC0 1.0 (public domain dedication) — synthetic data created for
  this task; commercial use permitted.

## 2. Fleet story (what the data represents)

A plant operator instruments its coolant pumps in two waves:

| Cohort | Pumps | IDs | Record span | Hours/pump |
|---|---|---|---|---|
| `pilot` | 100 | `P0001`–`P0100` | 2025-01-06 00:00 → 2025-04-05 23:00 | 2,160 (90 days) |
| `rollout` | 800 | `R0001`–`R0800` | 2025-04-07 00:00 → 2025-04-26 23:00 | 480 (20 days) |

The pilot cohort has a long history **and** completed post-incident reviews.
The rollout cohort is recent: its failures are in the ops `failure_log.csv`
(trips are recorded in real time), but the reliability team has **not yet
performed post-incident reviews** for this period — see `maintenance_flag`
below. This asymmetry is deliberate and documented; solvers must
reason about which columns are available at prediction time.

## 3. Files & schema

This dataset upload contains exactly four files, all at the top level: the
three CSVs documented below plus `generate_dataset.py` (the reproducible
generation script).

### `sensor_readings.csv` — 600,000 rows (hourly, per pump)

| Column | Type | Unit | Description |
|---|---|---|---|
| `pump_id` | string | — | Pump identifier (`P####` pilot, `R####` rollout). |
| `timestamp` | datetime (`YYYY-MM-DD HH:MM:SS`, UTC, hourly) | — | Reading time. |
| `vibration_mm_s` | float | mm/s | RMS casing vibration velocity. **`-999.0` = sensor outage.** |
| `bearing_temp_c` | float | °C | Drive-end bearing temperature. **`-999.0` = sensor outage.** |
| `discharge_pressure_kpa` | float | kPa | Discharge pressure. **`-999.0` = sensor outage.** |
| `flow_rate_m3_h` | float | m³/h | Volumetric flow. `0.0` while the pump is stopped. **`-999.0` = sensor outage.** |
| `motor_current_a` | float | A | Motor phase current. `0.0` while stopped. **`-999.0` = sensor outage.** |
| `rpm` | float | rev/min | Shaft speed. `0.0` while stopped (repair downtime). **`-999.0` = sensor outage.** |
| `maintenance_flag` | int {0,1} | — | **Retroactive** annotation added during post-incident review: reviewers tag the 48 hours preceding each confirmed failure. Populated for the pilot cohort only; the rollout period has not been reviewed yet, so it is `0` on every rollout row. It is *not* a live signal available at prediction time. |

### `failure_log.csv` — 535 events

| Column | Type | Description |
|---|---|---|
| `pump_id` | string | Pump that failed. |
| `failure_time` | datetime | Hour at which the pump tripped (start of downtime). |
| `failure_mode` | string | `wear_out` (~72%: preceded by a gradual degradation ramp) or `sudden` (~28%: no precursor in the sensors). |
| `downtime_hours` | int | Repair downtime (24–72 h) following the trip. |

### `pump_metadata.csv` — 900 rows

| Column | Type | Description |
|---|---|---|
| `pump_id` | string | Pump identifier. |
| `pump_model` | string | `HX-300` (40%), `HX-500` (35%), `MV-90` (25%) — different baseline operating points. |
| `site` | string | `SITE-A` / `SITE-B` / `SITE-C` — small ambient-temperature offsets. |
| `cohort` | string | `pilot` or `rollout`. |
| `first_record` | datetime | First telemetry timestamp for the pump. |
| `last_record` | datetime | Last telemetry timestamp for the pump. |

## 4. Generation process (mechanistic model)

Full details and exact constants are in `generate_dataset.py`; summary:

1. **Baselines.** Each pump draws a model-specific operating point, then wide
   per-pump baseline jitter (e.g. vibration ×`exp(N(0, 0.45))`, bearing temp
   ±6.5 °C, rpm ±16). Healthy and degrading pumps therefore **overlap heavily
   in absolute values** — degradation is only reliably visible relative to a
   pump's own recent baseline.
2. **Operating cycles.** Diurnal ambient effect on bearing temperature
   (±2.5 °C, site-offset), and weekly + daily plant-load cycles (±5%/±3%)
   modulating flow, current and pressure. Gaussian sensor noise per channel.
3. **Failure process.** Per pump, a renewal process: time-to-failure ~
   Exponential(mean 1,100 h, min 24 h). With p=0.72 the failure is `wear_out`:
   a degradation ramp of 72–216 h precedes the trip, with severity
   `s = progress^1.6` rising 0→1 and coherently shifting **all six channels**
   (vibration ×(1+s), temp +14 °C·s, pressure −10%·s, flow −14%·s, current
   +8%·s, rpm −9·s). With p=0.28 it is `sudden`: **no precursor at all** —
   these failures are irreducibly unpredictable and cap the attainable score.
4. **Downtime.** After each trip the pump is down 24–72 h: `rpm`, `flow`,
   `current` read 0, temperature decays to ambient, vibration ≈ 0. Repair
   restores the healthy baseline.
5. **Benign transients** (process upsets/heat waves, mean interarrival 400 h):
   raise **only** temperature (+5–11 °C) and vibration (+30–70%) for 8–36 h,
   then fully recover. They mimic the early part of a wear ramp on those two
   channels and defeat naive univariate thresholding; true degradation moves
   all six channels coherently.
6. **Sensor outages.** Independent per channel: bursts of 2–8 h at mean
   interarrival 450 h are overwritten with the sentinel `-999.0` (~1.1% of
   readings per channel).
7. **Retroactive review flag.** For every pilot-cohort failure, the 48 hours
   preceding the trip get `maintenance_flag = 1`.

## 5. Controlled complexity (planted, documented)

| Property | Where | Skill it tests |
|---|---|---|
| `maintenance_flag` nearly equals the training label but is all-zero for unreviewed (test-time) data | pilot vs rollout | leakage detection: reading column provenance, not just correlations |
| `-999.0` outage sentinels (~1.1%/channel) | all sensor columns | missing-value handling on time series |
| ~4.2% positive rate in training streams | labels | class-imbalance handling |
| Downtime rows (rpm = 0) inside training streams | pilot streams | sample curation: stopped-pump rows teach "low flow/rpm = safe", inverting the true degradation signature |
| Wide per-pump baselines | all channels | per-entity normalization / trend features over absolute thresholds |
| Benign transients on temp+vibration only | pilot & rollout | multivariate reasoning over univariate rules |
| 28% `sudden` failures with no precursor | failure process | score-ceiling awareness, calibrated expectations |
| Highly autocorrelated hourly rows | all streams | group/temporal validation (random row K-fold overfits) |

## 6. Intended downstream splits (context only — not part of this upload)

A separate task submission provides a `prepare.py` that transforms the raw
files above into public/private splits during task creation on the platform.
The split files described below (`train.csv`, `test_features.csv`,
`sample_submission.csv`, `answers.csv`) are **not included in this
dataset upload**; they are documented here so reviewers can see the intended
use.

- **Public `train.csv`** (211,200 rows): pilot-cohort streams + per-row label
  `failure_within_48h` (1 iff a failure occurs in `(t, t+48h]`). The last 48 h
  of each stream are dropped (truncated label window).
- **Public `test_features.csv`** (134,400 rows = 800 snapshots × 168 h): one
  evaluation snapshot per rollout pump — the 7-day window ending at the
  prediction origin. **No data after any origin is published**, and each pump
  contributes exactly one snapshot, so the private labels cannot be recovered
  from the public split (failures are otherwise visible in retrospect as
  downtime signatures — that is precisely why the evaluation uses
  forward-censored snapshots).
- **Public `sample_submission.csv`**: format example, constant prediction.
- **Private `answers.csv`** (800 rows): `snapshot_id → failure_within_48h`;
  120 positives (15%) by stratified origin sampling (seeded, deterministic).

## 7. Sample-size summary

| Piece | Count |
|---|---|
| Raw telemetry rows | 600,000 |
| Pumps | 900 (100 pilot + 800 rollout) |
| Failure events | 535 (188 pilot, 347 rollout) |
| Training rows (prepared) | 211,200 (~4.2% positive) |
| Evaluation snapshots | 800 (120 positive) |

## 8. Regenerating

```bash
python generate_dataset.py --out-dir .   # regenerates the three CSVs next to the script
```

Byte-identical on every run. (Without `--out-dir`, the script writes into a
`raw/` subfolder next to itself.)
