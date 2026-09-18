# The Edit That Moved the Answer — Rubrics

Criteria are graded alongside the leaderboard score. They describe what good
ML engineering looks like *for this task specifically*: recovering the
ORDER in which an edit operation reshaped four candidate input kinds, from
before/after behaviour measurements alone.

---

### 1. Validates on a framing-level split — REQUIRED · TRAINING

Every item's released measurements come from one of thirteen training
prompt framings; the three test framings (analyst, productcopy, travel)
never appear in train.csv at all. Validation must hold out **whole
framings**, mirroring the real split, not individual rows.

*Fails if:* a row-level random split is used for validation, letting items
from the same framing sit on both sides.

### 2. Does not reduce to "which edit is this" — REQUIRED · MODELING

Detecting the edit operation from formatting alone (JSON requests, bullet
requests, brevity requests are all near-trivially visible in the released
measurements) answers a different, easier question than how the edit
reordered the four shown input kinds by effect. A solution that only
models edit-operation identity and ignores the per-slice measurements is
answering the wrong half of the task.

*Fails if:* the model's only input is edit_op and family, with the
per-slice measurement columns unused.

### 3. Predicts an ordering, not a single label — REQUIRED · MODELING

The target is rank_1..rank_4, a full ordering of the four shown candidates
by effect, not a single most-affected class. A solution that only ever
predicts one slice and leaves the remaining three positions arbitrary (or
copies them through unreordered, as sample_submission.csv does) has not
engaged with the ranking structure of the task, even if its top pick is
often correct.

*Fails if:* rank_2..rank_4 are filled by copying the item's slice_2..
slice_4 through unchanged rather than being genuinely ordered by the
solution's own predicted effect size.

### 4. Does not simply recompute distance on the released means — RECOMMENDED · FEATURE_ENGINEERING

Ordering candidates by plain Euclidean (or cosine) distance between fp_A
and fp_B scores 0.418 on this data (positional-credit metric; see grade.py)
— real signal, but not the ceiling. The true ordering also depends on how
consistent the model's behaviour was across the four probe questions per
slice, a property not directly released. A stronger solution engineers
features that approximate that consistency (e.g. relative distances across
all released fields rather than one aggregate norm, or per-field rather
than pooled comparisons) rather than treating "distance on the raw means"
as the whole feature set.

*Fails if:* the only engineered feature is a single Euclidean or cosine
distance per candidate slice.

### 5. Beats the trained-ridge-regression baseline — RECOMMENDED · TRAINING

A plain ridge regression on the raw fp_A/fp_B measurements, predicting each
candidate's rank position directly, scores 0.493 on the held-out test
framings (positional-credit metric). A solution that does not clear this
has not found signal beyond what a linear model already finds in the
released features.

*Fails if:* the final held-out score is at or below 0.493.

### 6. Handles a variable candidate set correctly — REQUIRED · DATA_HANDLING

Each item shows a different four of the six input kinds (slice_1..slice_4),
and the submitted ordering must be exactly a permutation of those four. A
solution that predicts from a fixed six-way vocabulary and does not
restrict its output to the item's own shown candidates will emit invalid
orderings on many rows.

*Fails if:* rank_1..rank_4 for any row contains a slice name outside that
row's own slice_1..slice_4, a repeated slice, or fewer than four entries.

### 7. Emits a structurally valid submission — REQUIRED · CODE_QUALITY

Exactly one row per item_id in test.csv with rank_1..rank_4 forming a
permutation of that row's shown candidates. `sample_submission.csv` in
`public/` shows the exact expected format.

*Fails if:* the grader cannot match a row's item_id, or any of the
rank_1..rank_4 columns is missing entirely.

### 8. Completes within the compute budget — REQUIRED · CODE_QUALITY

The full pipeline — loading train.csv (1,544 rows, ~130 columns), fitting
whatever model is chosen, and writing submission.csv for the 356 test rows
— must finish well inside the 1.5-hour GPU-time limit. The released
feature set is small and tabular; there is no reason a competent solution
should approach the limit.

*Fails if:* the notebook does not complete top-to-bottom inside the budget.

### 9. Reports which framings it generalizes to — RECOMMENDED · COMMUNICATION

Because the test set holds out only three framings, per-framing scores on
a held-out validation framing (not just an aggregate number) are
informative about whether a solution is learning an edit-to-ordering
interaction that transfers, or overfitting to quirks of the thirteen
training framings.

*Fails if:* no per-framing breakdown is reported anywhere in the notebook,
only a single aggregate score.
