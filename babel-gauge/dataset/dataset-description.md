# Legacy Telemetry Log Canonicalization — Six Undocumented Equipment-Monitoring Formats with Ground-Truth Canonical Records

68,000 equipment-telemetry log lines as emitted by **six legacy logging
systems with undocumented, mutually incompatible line formats**, each paired
with its canonical normalized record. Built for the **"Babel Gauge: Legacy
Telemetry Log Canonicalization"** task: learning the raw→canonical string
transduction from examples.

---

## 1. Provenance & license

- **Type:** fully synthetic. Every line is produced by the mechanistic
  renderers in [`generate_dataset.py`](generate_dataset.py) (numpy RNG draws
  + deterministic string templates). **No LLM output is used anywhere in the
  data**, and no Kaggle competition or external dataset was copied or used.
- **Reproducibility:** the generator is fully deterministic (global seed
  `20260707`; every line is drawn with a child RNG seeded by its own index,
  so output does not depend on iteration order). Running
  `python generate_dataset.py` twice produces **byte-identical** CSVs.
  Requires only `numpy` and `pandas` (Kaggle Python Docker image).
- **License:** CC0 1.0 (public domain dedication) — synthetic data created
  for this task; commercial use permitted.

## 2. Story (what the data represents)

A fleet-monitoring vendor ingests hourly equipment telemetry (temperature,
pressure, status) from customers running six generations of logging
firmware. Every system reports the same six facts — timestamp, device,
site, temperature, pressure, status — but each renders them differently:
different field orders and separators, site-local vs UTC clocks, °F vs °C,
psi vs kPa, implied decimal points in a fixed-width mainframe format,
12-hour vs 24-hour time, opposite slash-date orders, and per-system status
vocabularies. The ingestion pipeline also occasionally damages lines in
transit; damaged lines must be flagged rather than half-parsed.

The vendor's target is one canonical record string per line:

```
ts=2025-04-30T12:06:09Z|dev=VLV-2004|site=UAE3|temp_c=76.7|pres_kpa=311.8|status=OK
```

or the literal string `REJECT` for damaged lines.

## 3. Files & schema

This dataset upload contains exactly two files: the raw telemetry CSV and the
site reference table. The `prepare.py` script (submitted separately with the
task) transforms these into public/private splits for solvers.

### `telemetry_lines.csv` — 68,000 rows

| Column | Type | Description |
|---|---|---|
| `line_id` | string | `L000001`–`L068000`. |
| `raw_line` | string | The log line as received (one of six legacy formats, or a corrupted line). |
| `canonical` | string | The canonical record (see §2), or `REJECT` for corrupted lines. |
| `format_family` | string | Generator label `F1`–`F6` (**for review only** — withheld from solvers by the prepare script). |
| `split` | string | `train` (60,000) / `test` (8,000) — fixed by the generator; the prepare script publishes train pairs and withholds test targets. |

### `sites.csv` — 36 rows

| Column | Type | Description |
|---|---|---|
| `site` | string | Site code appearing in the log lines (e.g. `NE2`, `IND7`). |
| `utc_offset_min` | int | Minutes to **add to UTC** to get that site's local clock. Range −600…+720, including half-hour offsets (+330, −210). |

## 4. Generation process (mechanistic model)

Full details and exact constants are in `generate_dataset.py`; summary:

1. **Ground truth first.** Each line draws a canonical record: timestamp
   (2025-02-01 → 2025-05-31 UTC), device type (`PMP/CMP/FAN/VLV`) + 4-digit
   number, site (36 codes), temperature (−5…95 °C, 1 decimal), pressure
   (80…900 kPa, 1 decimal), status (`OK` 76% / `WARN` 16% / `FAULT` 8%).
2. **Rendering.** The record is rendered through one of six format
   renderers (F1 gateway text, F2 bus dump, F3 shuffled key=value, F4
   fixed-width mainframe, F5 US regional CSV, F6 JSON collector). Formats
   F1/F3/F4/F5 print **site-local** times; F2/F6 are UTC. F1/F5 print
   °F and psi (psi with 2 decimals so the kPa value is exactly
   recoverable); F4 prints temperature and pressure ×10 as integers
   (implied decimal). F3 renders `DD/MM/YYYY`, F5 renders `MM/DD/YY` with
   a 12-hour `AM/PM` clock. F3/F4/F5 carry no seconds — their canonical
   timestamps end `:00`. Each format uses its own status vocabulary
   (`0/1/2`, `O/W/F`, `RUN/WRN/FLT`, `ok/warn/fault`).
3. **Corruption.** ~4% of train and ~5% of test lines suffer ingestion
   damage: 2–4 structural characters plus 1–4 arbitrary characters are
   deleted and one surviving character is garbled. Damage always breaks
   the line's grammar (verified: **zero** corrupted lines still parse), so
   `REJECT` is exactly decidable.
4. **Split design.** Train and test are drawn with **disjoint device-number
   ranges** (0001–6999 vs 7000–9999), so identifiers must be copied from
   the input, never memorized. The rare format F6 is deliberately scarce in
   training (1.5%) but common in test (20%).

**Recoverability guarantee (verified):** an independent oracle parser
reconstructs the canonical string **exactly for 100.00% of the 65,239
uncorrupted lines**, and **0** of the 2,761 corrupted lines parse under any
format grammar.

## 5. Controlled complexity (planted, documented)

| Property | Where | Skill it tests |
|---|---|---|
| Six undocumented grammars | `raw_line` | inducing structure/rules from paired examples |
| Site-local vs UTC clocks (offsets −600…+720 min incl. half-hours) | F1/F3/F4/F5 + `sites.csv` | learned time normalization incl. date rollover across midnight |
| °F→°C and psi→kPa with exact re-rounding | F1/F5 | discovering numeric transforms from pairs, not assuming identity |
| Implied ×0.1 decimals in fixed-width integers | F4 | fixed-width legacy conventions |
| `DD/MM/YYYY` (F3) vs `MM/DD/YY` (F5) + 12-hour clock with `12AM→00` | F3/F5 | resolving date ambiguity from evidence, not convention |
| Four distinct status vocabularies + one-letter device-type codes | F2/F4/F5/F6 | vocabulary alignment from pairs |
| ~4–5% corrupted lines → `REJECT` | both splits | strict grammar validation vs best-guess parsing |
| F6 = 1.5% of train but 20% of test | split design | not neglecting the low-resource format |
| Test device numbers disjoint from train | both splits | copying identifiers instead of memorizing them |

## 6. Data splits created by prepare.py

The task submission includes a `prepare.py` script that, when run on this raw
dataset, creates the public and private files for solvers:

- **Public `train.csv`** (60,000 rows): `line_id, raw_line, canonical`.
- **Public `test.csv`** (8,000 rows): `line_id, raw_line`.
- **Public `sites.csv`**: copy of the uploaded file above.
- **Public `sample_submission.csv`**: required format (`line_id, output`),
  filled with the constant `REJECT`.
- **Private `answers.csv`** (8,000 rows): `line_id, canonical` for the test
  lines. The `format_family` and `split` columns are never published.

## 7. Sample-size summary

| Piece | Count |
|---|---|
| Lines | 68,000 (60,000 train / 8,000 test) |
| Train format mix | F1 28% · F2 22% · F3 20% · F4 18% · F5 10.5% · F6 1.5% |
| Test format mix | F1 21% · F6 20% · F2 18% · F3 16% · F4 14% · F5 12% |
| Corrupted (`REJECT`) | 3.9% of train · 5.0% of test |
| Sites / device types | 36 / 4 |

## 8. Regenerating

```bash
python generate_dataset.py --out-dir .   # regenerates the two CSVs next to the script
```

Byte-identical on every run. (Without `--out-dir`, the script writes into a
`raw/` subfolder next to itself.)
