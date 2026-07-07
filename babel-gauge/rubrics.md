# Rubrics — Babel Gauge: Legacy Telemetry Log Canonicalization

Task-specific criteria for what good ML engineering looks like on this
task. Evaluated alongside the mean-field-accuracy score.

| # | Level | Type | Rubric |
|---|---|---|---|
| 1 | **REQUIRED** | DATA_HANDLING | **Decides `REJECT` by strict grammar validation**: corrupted lines (~4% of train, ~5% of test) are identified because they fail every source system's full-line structure — not by ad-hoc heuristics like line length. The solution must NOT emit best-guess canonical records parsed out of damaged lines, and must NOT blanket-`REJECT` lines from valid but less common source systems. |
| 2 | **REQUIRED** | TRAINING | **Validates the learned transducer on held-out training pairs** (exact-match rate and/or the task's field-accuracy metric) before predicting the test set, with per-source-system accuracy inspected — and must NOT report validation only on the same examples used to derive the rules/model. On this task a correct pipeline reaches ~100% held-out exact match; a solution validating far below that should iterate rather than submit. |
| 3 | **REQUIRED** | DATA_HANDLING | **Normalizes clocks to UTC using the per-site offsets** (learnable from the pairs and cross-checkable against `sites.csv`): source systems that print site-local times are shifted by the site's offset — including half-hour offsets and shifts that cross midnight and change the calendar date. Copying local timestamps into `ts` unchanged fails this criterion (it alone costs ≈0.1 score). |
| 4 | **REQUIRED** | MODELING | **Derives every numeric and vocabulary mapping from the data rather than assuming it**: unit transforms for temperature/pressure (some systems report °F and psi), the fixed-width system's implied ×0.1 integer encoding, per-system status vocabularies, and device-type letter codes — each verified against training pairs (near-zero reconstruction error) with exact one-decimal re-rounding into the canonical format. |
| 5 | **RECOMMENDED** | DATA_HANDLING | **Resolves date-field order per source system from evidence**: the two slash-date systems use opposite orders (`DD/MM` vs `MM/DD`), and one combines it with a 12-hour `AM/PM` clock (including `12:xxAM → 00:xx`). The solution demonstrates the order empirically (e.g. consistency counts against canonical timestamps) instead of assuming one convention globally — day≤12 dates make the wrong assumption look mostly right in spot checks. |
| 6 | **RECOMMENDED** | MODELING | **Handles the low-resource source system**: one format is ~1.5% of training but ~20% of the test set (the test mix differs from train, as stated in the problem). The solution treats all discovered formats as first-class — per-format validation coverage rather than optimizing only the frequent ones. |
| 7 | UNIVERSAL | CODE_QUALITY | **Runs end-to-end reproducibly and emits a valid submission**: fixed seeds, reads only `./dataset/public/`, writes `./working/submission.csv` with exactly one non-empty output per `line_id` in `test.csv`, verified in-notebook (e.g. asserts) before writing. |

**Level summary:** 4 REQUIRED + 2 RECOMMENDED + 1 UNIVERSAL (majority
REQUIRED/RECOMMENDED, as the spec requires).
