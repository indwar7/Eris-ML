# Rubrics — Silent Degradation: 48-Hour Pump Failure Risk

Task-specific criteria for what good ML engineering looks like on this
task. Evaluated alongside the Average Precision score.

| # | Level | Rubric |
|---|---|---|
| 1 | **REQUIRED** | **Excludes `maintenance_flag` from the model's inputs** (or demonstrably neutralizes it), explicitly reasoning that it is a *retroactive post-incident annotation* that nearly duplicates the training label and is all-zero in `test_features.csv` — i.e., unavailable at prediction time. Simply never mentioning the column is not enough: the solution must show it was considered and rejected (e.g., notes its near-perfect correlation with the label or its constant value in the test period). |
| 2 | **REQUIRED** | **Treats `-999.0` sensor values as missing data**, converting the sentinel to NaN (or equivalent) *before* any statistics, imputation, or model fitting, and imputes within each pump's/snapshot's own time series (e.g., forward/backward fill or interpolation) rather than leaving -999 as a numeric reading or imputing with a global constant across pumps. |
| 3 | **REQUIRED** | **Uses leakage-safe validation**: splits by `pump_id` (e.g., GroupKFold / grouped holdout) or strictly by time — never shuffled row-level K-fold, which places near-duplicate adjacent hours of the same pump in both train and validation folds. Validation is scored with Average Precision (or a directly comparable ranking metric), matching the evaluation metric. |
| 4 | **REQUIRED** | **Handles the class imbalance deliberately** (~4% positive rows in train): uses an imbalance-appropriate objective/evaluation (AP / PR-based), and where relevant class weights, resampling, or threshold-free scoring — rather than optimizing accuracy or relying on a default 0.5 threshold anywhere in model selection. |
| 5 | **RECOMMENDED** | **Curates training examples around repair downtime**: rows where the pump is stopped (`rpm == 0` / `flow == 0`) are excluded from the training matrix or otherwise specially handled — training on them teaches the model that low flow/rpm is safe, the opposite of the degradation signature. (Downtime hours may still legitimately appear inside rolling-history windows.) |
| 6 | **RECOMMENDED** | **Builds per-pump *relative* temporal features** from the history windows — rolling means/stds, short-vs-long-window trends, deltas, or normalization against the pump's own recent baseline (e.g., value ÷ its 168-h median) — rather than relying on absolute instantaneous readings, which are confounded by wide healthy-baseline differences between pumps. Test-time features must use only each snapshot's 168-hour window (no cross-snapshot statistics). |
| 7 | OPTIONAL | **Shows awareness of the score ceiling**: identifies that a minority of failures have no sensor precursor (sudden failures), e.g. by inspecting positive training windows without drift, and calibrates expectations/model complexity accordingly instead of chasing the irreducible errors. |

**Level summary:** 4 REQUIRED + 2 RECOMMENDED + 1 OPTIONAL (majority
REQUIRED/RECOMMENDED, as the spec requires).
