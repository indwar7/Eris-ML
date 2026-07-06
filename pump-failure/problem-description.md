# Silent Degradation: 48-Hour Pump Failure Risk

> **Domain: From Scratch.** You **build and train a model from scratch** on
> the provided data — no pretrained models, no foundation models, no LLMs,
> and no external data. Everything you need is in the public files; the whole
> pipeline (load → features → train → predict) must run end-to-end in your
> notebook. It is a from-scratch supervised machine-learning problem:
> engineer your own features from raw industrial sensor readings and train
> your own classifier/ranker to score failure risk.

## Task

You **train a failure-risk model from scratch** on hourly condition-monitoring
data from a fleet of industrial coolant pumps. For each of 800 **evaluation
snapshots** — a 7-day (168-hour) window of six sensor readings for one pump —
predict the **risk that the pump suffers a failure event within the 48 hours
immediately following the snapshot's final timestamp** (its *prediction
origin*).

There is no pretrained model to fine-tune and no off-the-shelf solution: you
design the features and build the model yourself, from the raw signals up.
The predictive signal lives in how each pump's readings *evolve over its
history* — trends, drifts, and coherent multi-channel shifts relative to the
pump's own recent baseline — not in any single hourly reading. Scores are
evaluated with **Average Precision** (ranking), so only the *ordering* of
your risk scores matters.

## The sensor channels

Every pump reports the same six hourly sensor readings. These are the raw
signals you engineer features from:

| Channel | Unit | Notes |
|---|---|---|
| `vibration_mm_s` | mm/s | RMS casing vibration velocity. |
| `bearing_temp_c` | °C | Drive-end bearing temperature. |
| `discharge_pressure_kpa` | kPa | Discharge pressure. |
| `flow_rate_m3_h` | m³/h | Volumetric flow. `0.0` while the pump is stopped. |
| `motor_current_a` | A | Motor phase current. `0.0` while stopped. |
| `rpm` | rev/min | Shaft speed. `0.0` while stopped for repair. |

Each reading also carries an ambient/site identifier (`pump_model`, `site`)
and, in training only, a retroactive `maintenance_flag` (see notes below).

## Data files

The data is delivered in **long format** — one row per (pump, hour), ordered
by `timestamp`. Group by the pump/snapshot id and sort by `timestamp` to
reconstruct each pump's history before building features.

### `train.csv` — 211,200 rows

Continuous hourly sensor histories for 100 pumps (`pump_id` `P0001`–`P0100`)
over ~88 days — one uninterrupted, chronologically ordered history per pump,
labeled at **every hour** so you can learn what the hours leading up to a
failure look like.

| Column | Type | Role |
|---|---|---|
| `pump_id` | str | **Pump/history id** — group by this, sort by `timestamp` to reconstruct a stream. |
| `timestamp` | str, `YYYY-MM-DD HH:MM:SS` (UTC, hourly) | Reading time (regular, 1-hour step). |
| `pump_model` | str | `HX-300` / `HX-500` / `MV-90` — different baseline operating points. |
| `site` | str | `SITE-A` / `SITE-B` / `SITE-C` — small ambient-temperature offsets. |
| six sensor channels | float | The raw signals (see table above). |
| `maintenance_flag` | int {0,1} | See **Important data notes** below. |
| `failure_within_48h` | int {0,1} | **Per-row label.** 1 iff a failure event occurs in `(t, t+48h]` for this pump. |

### `test_features.csv` — 134,400 rows (800 snapshots × 168 hours)

The same six channels (no label), plus `snapshot_id`
(`snap_0001`…`snap_0800`) identifying each evaluation snapshot. Each
`snapshot_id` is one pump's history of exactly **168 consecutive hourly
rows** (`pump_id` `R0001`–`R0800`; **each pump appears in exactly one
snapshot**). The last row of a snapshot is its prediction origin; you emit
one risk score per snapshot for the 48 hours after that origin. The
evaluation is **forward-censored** — no data after any snapshot's origin
appears anywhere in the public data, so the horizon is genuinely a
prediction, not a lookup.

### `sample_submission.csv`

The required output format, filled with a constant prediction.

## Important data notes (read carefully)

1. **Sensor outages (missing values):** in every sensor channel, the value
   **`-999.0` is a sentinel meaning "sensor outage"** — a burst of 2–8
   consecutive hours (~1% of rows per channel) where that channel is missing.
   It is not a physical reading; treat it as missing and impute/interpolate
   within the pump's own history.
2. **`maintenance_flag` is a retroactive annotation**, not a live sensor:
   during *post-incident review*, reliability engineers tag the 48 hours
   preceding each confirmed failure. Reviews exist for the training-period
   data. The evaluation snapshots come from a recent period that **has not
   been reviewed yet, so `maintenance_flag` is `0` on every row of
   `test_features.csv`.**
3. **Repair downtime:** after a failure a pump is stopped for 24–72 hours
   (`rpm`, `flow_rate_m3_h`, `motor_current_a` read `0`), then returns to
   service restored. Such stopped-then-restored stretches appear inside the
   training histories and may appear *inside* an evaluation snapshot's
   history (a failure+repair that finished before the origin). The label
   always refers to failures in the *next 48 h*, regardless of the pump's
   operating state at the origin — so a snapshot containing an old downtime
   segment is not itself a positive.
4. **Per-pump baselines differ widely** (installation and duty-point
   differences). An absolute reading that is alarming for one pump is normal
   for another — degradation is only visible *relative to each pump's own
   recent history*, which is why per-pump normalization and trend features
   beat global thresholds.
5. **Some failures have no precursor.** A substantial minority of failures
   are sudden (e.g. electrical faults) with no drift in any of the six
   channels beforehand. A perfect score is not attainable; do not expect to
   rank every positive snapshot highly.
6. **Class imbalance.** The evaluation snapshots were sampled so ~**15%** are
   positive (failure within 48 h of the origin); the training histories have
   a much lower per-row positive rate (~4%). The metric is ranking-based, so
   calibration to either rate is not required.
7. All evaluation origins are operating points (the pump is running at the
   last row of the snapshot).

## Metric

**Average Precision (AP)** over the 800 evaluation snapshots, computed
exactly as `sklearn.metrics.average_precision_score(y_true, y_score)`:

AP = Σₙ (Rₙ − Rₙ₋₁) · Pₙ

where Pₙ and Rₙ are precision and recall at the n-th threshold of the
descending score ordering. Higher is better. A constant or random submission
scores ≈ 0.15 (the evaluation positive rate).

## Submission format

Write a CSV file named **`submission.csv`** with exactly these two columns:

```
snapshot_id,prediction
snap_0001,0.07
snap_0002,0.31
...
```

- Exactly **one row per evaluation snapshot** — one line per `snapshot_id`
  in `test_features.csv` (800 rows + header), in any order.
- `prediction` is your risk score (higher = more likely to fail within 48 h
  of the origin). Values are recommended in `[0, 1]`; any finite real numbers
  are scored (ranking metric). NaN/infinite/non-numeric values, missing or
  unknown `snapshot_id`s, and duplicate rows make the submission invalid.

## Rules & environment

- **Build from scratch.** Train your own model on the provided data — **no
  pretrained or foundation models, no LLMs**, and **no external data**.
  Everything needed is in the public files.
- Use only libraries available in the **Kaggle Python Docker image** (pandas,
  numpy, scikit-learn, xgboost, lightgbm, tensorflow, pytorch, …).
- **No LLM-generated outputs** may be used anywhere in your solution.
- Your notebook must run **end-to-end, top to bottom**: load the public data
  → build features → train → predict → write `submission.csv`.
- Seed everything; your run should be reproducible.
