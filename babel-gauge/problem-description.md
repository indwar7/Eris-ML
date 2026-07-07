# Babel Gauge: Legacy Telemetry Log Canonicalization

> **Domain: Sequence-to-Sequence.** You learn a string-to-string
> transduction from 60,000 paired examples — no pretrained models, no LLMs,
> and no external data. Any approach is valid (learned rule systems,
> grammar induction, neural sequence models, hybrids) as long as it is
> derived from the provided data and runs end-to-end in your notebook.

## Task

A fleet-monitoring vendor ingests equipment telemetry from **several legacy
logging systems, each with its own undocumented line format**. Some lines
are also damaged in transit. For every raw log line, output either

1. the **canonical record** — a single string of exactly six `key=value`
   fields joined by `|`, in this fixed order:

   ```
   ts=2025-04-30T12:06:09Z|dev=VLV-2004|site=UAE3|temp_c=76.7|pres_kpa=311.8|status=OK
   ```

   | Field | Format |
   |---|---|
   | `ts` | UTC timestamp, `YYYY-MM-DDTHH:MM:SSZ` (seconds always present) |
   | `dev` | 3-letter device type (`PMP`, `CMP`, `FAN`, `VLV`) + `-` + 4-digit number |
   | `site` | site code, as listed in `sites.csv` |
   | `temp_c` | temperature in °C, exactly one decimal (`%.1f`) |
   | `pres_kpa` | pressure in kPa, exactly one decimal (`%.1f`) |
   | `status` | `OK`, `WARN`, or `FAULT` |

2. or the literal string **`REJECT`** if the line is corrupted (damaged
   beyond reliable parsing).

The train split gives you `raw_line → canonical` pairs; the test split
gives you raw lines only. **Every uncorrupted line contains all six facts,
and they are exactly recoverable** — the mapping is deterministic, with no
irreducible ambiguity. The legacy formats differ in far more than layout:
expect differences in clock convention, units, decimal representation,
date-field order, and status vocabulary. None of this is documented —
it must be learned from the pairs.

## Data files

All public files are in `./dataset/public/`.

### `train.csv` — 60,000 rows

| Column | Type | Role |
|---|---|---|
| `line_id` | str | Unique id. |
| `raw_line` | str | The log line as received. |
| `canonical` | str | Target: canonical record, or `REJECT` (~4% of rows). |

### `test.csv` — 8,000 rows

`line_id`, `raw_line` — produce one output per row. Note: the mix of
source systems in the test split **differs from the training split**, and
test device numbers never appear in training (identifiers must be carried
over from the input, not looked up).

### `sites.csv` — 36 rows

`site`, `utc_offset_min` — minutes to **add to UTC** to get each site's
local clock (some sites use half-hour offsets).

### `sample_submission.csv`

The required output format (`line_id, output`), filled with the constant
`REJECT`.

## Metric

**Mean per-line field accuracy** over the 8,000 test lines.

```python
FIELDS = ["ts", "dev", "site", "temp_c", "pres_kpa", "status"]

def line_score(truth: str, out: str) -> float:
    out = out.strip()
    if truth == "REJECT":
        return 1.0 if out == "REJECT" else 0.0
    if not out or out == "REJECT":
        return 0.0
    pieces = out.split("|")
    got = {}
    for p in pieces:
        if "=" in p:
            k, v = p.split("=", 1)
            got.setdefault(k, v)
    tru = dict(p.split("=", 1) for p in truth.split("|"))
    correct = sum(got.get(k) == tru[k] for k in FIELDS)
    return correct / max(6, len(pieces))
```

The final score is the unweighted mean of `line_score` over all lines.
Range [0, 1]; higher is better. Values match by **exact string
comparison** — respect the canonical formats above (one decimal, zero-
padded device numbers, uppercase status). Extra or malformed fields dilute
the denominator; missing fields earn nothing.

Reference points: copying the input scores ≈ 0.00; answering `REJECT`
everywhere scores ≈ 0.05; a parser that recognizes every line shape but
ignores the systems' hidden conventions scores ≈ 0.76; the task is exactly
solvable — a fully correct transducer scores 1.0.

## Submission format

Write a CSV file named **`submission.csv`** to `./working/` with exactly
these two columns:

```
line_id,output
L060001,ts=2025-03-14T18:22:00Z|dev=PMP-8413|site=NE2|temp_c=41.3|pres_kpa=312.9|status=OK
L060002,REJECT
...
```

- Exactly **one row for each `line_id` in `test.csv`** (8,000 rows +
  header), in any order.
- `output` is the canonical record string or `REJECT`. Empty outputs score
  0 for their line. Missing/unknown/duplicated `line_id`s make the
  submission invalid.
- The canonical string contains `|` characters — write the CSV with
  standard quoting (pandas `to_csv` handles this; `|` needs no quoting in
  comma-separated files).

## Rules & environment

- **Learn from the provided data only** — no pretrained or foundation
  models, no LLMs, and no external data.
- Use only libraries available in the **Kaggle Python Docker image**
  (pandas, numpy, scikit-learn, xgboost, lightgbm, tensorflow, pytorch, …).
- **No LLM-generated outputs** may be used anywhere in your solution.
- Your notebook must run **end-to-end, top to bottom**: load the public
  data → learn the transduction → write `./working/submission.csv`.
- Seed everything; your run should be reproducible.
