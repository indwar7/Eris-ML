# Digitization Pipeline Attribution — design record

## The task in one line

Given a shuffled bag of real historical-newspaper OCR text snippets, each from
a DIFFERENT article/topic, recover which real digitization batch each snippet
came from — using only the OCR engine's character-error fingerprint, never the
words themselves.

## Why this shape

An earlier occupancy scan across candidate GPU-native NLP tasks killed four
spaces: stylometric authorship-vs-topic separation (PAN 2026 runs this exact
shared task), sentence/paragraph intrusion detection (INSteD, 170k+ real docs,
same idea), and a text-steering-hidden-classifier idea that was a reskin of
this portfolio's own prior work (blind-spot / galaxy-blind-spot). The one
candidate that came back with no direct hit: recovering which real
**digitization pipeline** produced a piece of OCR'd text, from its error
pattern, ignoring content. This instantiates the same "grade against a real
recorded system identity" engine as blind-spot, in a new domain, with the
ground truth being the archive's own ingest record (a `batch_id`) rather than
a hidden model's measured behaviour.

## Data

Real Library of Congress NDNP (National Digital Newspaper Program) newspaper
pages, fetched from the bulk mirror `chroniclingamerica.loc.gov/data/batches/`
(the modern API/loc.gov endpoints are Cloudflare-blocked, 403 — this mirror is
not). Each page's ALTO XML carries genuinely noisy OCR text AND real recorded
vendor metadata (`softwareCreator`/`softwareName`/`processingAgency`) — a
literal recorded pipeline fingerprint, not inferred. 35 batches, 8,417
snippets (one per `TextBlock`, i.e. one per article/column region), 34
distinct institution prefixes. `software_creator` populated for all 35
batches; `processing_agency` for 17/35 (some awardees never recorded it).

Fetch throughput was rate-limited by the LOC mirror (~1 batch per 2-5 min); 35
batches was the practical ceiling within the build's time budget, not a
target. `build_raw.py` is resumable and 250+ further verified candidate batch
IDs remain untried if the corpus needs to grow later.

## Split: batch-disjoint (open-set), 45% test by batch

`train.csv` is flat and fully labeled (`row_id, text, batch_id`) from 19
batches. `test.csv` is bag-structured and unlabeled from the other 16 batches
— **test batches never appear in train.** This forces a solver to learn a
GENERAL notion of "what does this OCR pipeline's noise look like," not
memorize a closed vocabulary of known batch IDs. Since v2 (see "Review round
1" below) `batch_id` values in every public file are opaque per-batch aliases
(`batch_NN`), never the archive's own batch names, and every test row is a
distinct source snippet — v1 sampled with replacement across bags, v2 does
not.

The 45% test fraction (19 train / 16 test batches) was NOT the first choice —
it is the empirically safest point found after three measured attempts (see
"the adversarial finding" below). **On the test/train ratio convention
(15–25% by rows):** this split is by *batch*, and because test bags draw
without replacement from every held-out batch, the test set carries most of
its batches' snippets — 3,568 test rows against 4,194 train rows (0.85 by
rows; 16/35 = 0.46 by batch). It is deliberately kept there: holding out
*fewer* batches makes the pooled-clustering shortcut a reviewer flagged in
round 2 *stronger* (with 9 test batches it beat the trained reference; see
the adversarial finding), and every held-out batch is needed to keep 3–4
distinct batches per bag across 127 bags. Train still exceeds test in rows
and the split is group-disjoint, which are the two properties the
convention exists to protect. Standard practice elsewhere in this
portfolio holds out ~25-30% for test; here more of the batch pool had to go to
test specifically to blunt a shortcut that measurement caught, not intuition.

## Task artifact: a partition, not a label

Each bag draws 3-4 DISTINCT test batches and 6-10 snippets per chosen batch (v3; v2 drew 3-5 and 3-8) —
both randomized per bag, so a fixed bag shape can never itself be a tell.
Snippet order within a bag is shuffled. Submission is one row per bag,
`(bag_id, batch_id)`, where `batch_id` lists an arbitrary label per snippet in
the bag's snippet order (v3; earlier builds used one row per snippet — see
"Review round 2" for why that changed). Labels are meaningful only within
that bag — the solver never has to name the real batch, only recover which
snippets travelled together. This keeps the graded artifact a
**partition**, which is none of classification / regression / ranking /
retrieval / OCR / segmentation / captioning / sequence-prediction /
information-extraction — the review-committee's excluded-task list.

## Metric: mean per-bag Adjusted Rand Index (ARI), clipped to [0, 1]

**This was not the first metric shipped, and the reason it changed is the
single most important measured finding in this build.**

The first version scored mean per-bag PAIRWISE CO-MEMBERSHIP F1 (chosen
specifically to avoid a "matched assignment" trap — a prior challenge's
review in this portfolio flagged that Hungarian-style best-alignment scoring
can silently hand out credit a model never earned on clustering-shaped
tasks). Measuring the actual baseline ladder caught a DIFFERENT problem with
that choice: raw pairwise F1 structurally rewards "put everyone in one
group" — it gets perfect recall on every true same-batch pair for free and
only loses on precision, which for bags with a handful of groups isn't
enough of a penalty. Measured: with an earlier bag composition (5-9 groups
per bag), "everyone in one group" scored **0.376** under pairwise F1 —
comfortably beating a properly trained embedding-based reference (0.289).
This is exactly the failure this project's rejection register warns about
under "reference below a naive baseline."

Adjusted Rand Index fixes this by construction: it is chance-normalized, so
both degenerate strategies (one giant group, or every snippet its own
singleton) score ~0.0, and only genuine above-chance agreement with the true
partition earns credit — re-measured, both degenerate baselines now score
**exactly 0.000**. ARI still needs no group-label alignment step, so it keeps
the original anti-Hungarian-matching property.

## A measurement bug found and fixed mid-build

Before trusting any of the numbers below: an early version of the baseline
harness (`work/baselines.py`) keyed its `assignments` dict by bare
`snippet_id`. Because v1 bags deliberately reused snippets across DIFFERENT
bags (sampling was with replacement across bags, without replacement within
one bag; v2 removed the reuse entirely — see "Review round 1"), a snippet
appearing in two bags had its
second-computed label silently overwrite the first in that flat dict, so
every bag containing that snippet got graded against whichever bag happened
to be processed last. This was caught by a large, unexplained gap between
the harness's number for the trained reference (0.148) and a freshly written
`reference_solution.py`'s number on the same data (0.316) — the two
computed the same clustering logic but only one indexed by `(bag_id,
snippet_id)`. Fixed by keying every assignment dict that way throughout.
The corrected numbers below are uniformly HIGHER and more separated than the
buggy ones — including the safety margin against the adversarial attack,
which went from a worryingly thin 0.009 to a comfortable 0.177. The
qualitative finding that motivated the 45% test-split fix (see next section)
was still directionally correct even under the buggy measurement; only the
absolute numbers and margin size were wrong.

## The adversarial finding: a real transductive/cross-bag shortcut, measured and fixed

*(Numbers in this section are from the v1 build, where the split fraction
was chosen. The v2 re-measurement is in the ladder below; the conclusion —
the reference beats the attack by ~0.18 under both realistic and oracle-k
conditions — holds unchanged.)*

A rejection this project received on a different challenge flagged: *"it fits
global models on the test data, and groups separate test rows together to
intersect their candidate banks, exploiting a dataset artifact... instead of
relying on a generalizable model."* Before freezing this spec, that exact
attack was built and run against this design: **ignore bag boundaries
entirely, run one unsupervised clustering pass over the WHOLE test corpus at
once**, using zero train data.

First measurement (9 test batches, the original 30%-test split, BEFORE the
scoring bug above was found): this attack, given the true number of test
batches as `k`, scored an ARI above both trained references measured at the
time. The exact historical numbers from that pass are superseded by the bug
fix and not reproduced here, but the qualitative conclusion held up after
the fix was applied and re-measured: with too few held-out test batches, the
shortcut is genuinely competitive with or ahead of a trained reference. The
shortcut was real, not theoretical — only the precise margins reported at
each stage needed correcting.

Root cause: with very few real batches held out for test, an unsupervised
pass can trivially discover "the" small number of true clusters without
learning anything about the fingerprint signal itself. Fix attempts and
their measured effect:
- Raising the test-batch fraction to 45% (16 test batches, 19 train): the
  **realistic** version of the attack — a solver estimating cluster count
  itself via silhouette score, no cheating — sits at 0.072, comfortably below
  the trained reference (0.316, post bug-fix). The ORACLE version of the
  attack (told the true k) reaches 0.139, still well below the reference.
- Raising it further to 55% (19 test / 16 train) was tested and rejected: it
  made the trained reference WORSE, not better (fewer train batches hurt the
  reference more than the extra test batches hurt the unsupervised
  attacker), the opposite of the intended effect. This closed off "just hold
  out more batches" as a monotonic fix and is why 45%, not the largest tried
  fraction, is what shipped.
- 45% was kept as the final setting: it is the measured optimum among the
  fractions tried, not a rounded compromise.

**Honest residual note.** Against an OMNISCIENT version of the attack — where
the solver is simply told the exact true number of test batches, which is
never published and has no legitimate way to be derived — the reference still
wins comfortably: 0.316 (reference) vs 0.139 (oracle-k global clustering), and
0.329 vs 0.139 in a fair oracle-vs-oracle comparison where the reference is
ALSO given the true k. Margin ~0.18-0.19 in both cases. This is treated the
way this project's other challenges treat ceiling/oracle disclosures (e.g.
repeat-harvest's answer-knowing-adversary checks): documented plainly here
rather than hidden, on the grounds that a real solver has no channel to learn
the true global batch count in the first place — it is not a column in any
released file, not derivable from train (train batches are disjoint from
test), and not something an unsupervised method can obtain "for free" the way
the diagnostic version of this check hands it over.

## Review round 1: two real findings, both fixed by redesign

The v1 build above went to review. The reviewer found two things, both
correct, both confirmed by re-measurement before anything was changed:

**Finding 1 — a source-lookup path scored 1.000.** `dataset_stats.json`
listed all 16 held-out test batch ids, and the task page cites the
Chronicling America bulk mirror. The reviewer fetched those 16 batches,
exact-matched every one of the 4,457 test rows against the original OCR
text, and recovered the true grouping with no modeling. Re-checked locally:
100% of test rows were byte-identical to an archive region, and `train.csv`
additionally shipped 19 real batch names, which reveal the archive's naming
scheme and prefix set.

**Finding 2 — 2,577 distinct snippets backed 4,457 test rows.** v1 sampled
with replacement across bags (documented as harmless because each bag is
graded independently). Re-checked: 70.1% of rows came from a snippet that
appeared elsewhere in the test set, one snippet 7 times. The reviewer's
concern — repeated examples can dominate the leaderboard — stands regardless
of per-bag grading.

### What was changed (all in `prepare.py`, all verified by script)

| Fix | Mechanism | Verified |
|---|---|---|
| No batch identity anywhere public | `dataset_stats.json` carries counts only; every `batch_id` in `train.csv` and `answers.csv` is an opaque alias `batch_NN`; real ids + alias map live in `private/split_manifest.json` | no list-valued field in public stats; every public label matches `batch_\d\d` |
| Enumeration recipe withheld | The candidate-selection rule ("up to 6 per prefix, alphabetically first") and the skipped-batch list are removed from public docs — with the public mirror they let the batch set be rebuilt | grep |
| No byte-identical text | Every shipped snippet (train and test) is a contiguous window of its source region, 80–97% of it, bounds from a private salt; a window that coincides with any other region's full text is rejected | 0 / 3,649 test rows equal any of 8,417 archive regions |
| No snippet reuse | Sampling without replacement globally; distinctness enforced on shipped TEXT, not just id (two regions can carry identical boilerplate) | 3,649 / 3,649 distinct |
| Bag shape enforced | (v2: 3–5 groups × 3–8 snippets; v3: 3–4 × 6–10) The shape is enforced, not targeted: under-filled groups are dropped, under-filled bags discarded (169 of 200 attempted survive) | groups/bag ∈ {3,4,5}; group size ∈ [3,8] |
| Split re-drawn | New seed; 7 of 16 test batches differ from v1 | manifest |
| Explicit rule | Problem statement and a new `[REQUIRED]` rubric: data directory only, no archive lookup, no external data; solver environment expected offline | — |

### What the lookup fix can and cannot do — stated plainly

Windowing closes exact matching. It does **not** close substring matching:
a window is a verbatim sub-span, and against an adversary who already holds
the source text of the test batches, any signal-preserving transform of a
unique document remains findable (a single rare OCR junk token is itself a
page fingerprint). Character-level perturbation was considered and rejected:
it would break the dataset's core claim that the noise is the archive's own,
uncorrected output, and it only raises the attack's cost, never to infinity.

So the closure is layered, and the layers are: (1) nothing public names a
batch, (2) nothing public lets the batch set be re-enumerated cheaply, (3)
nothing public is byte-identical to the archive, (4) the rule makes lookup
an invalid submission regardless of score, and (5) the solver environment is
expected to have no network access. The reviewer's path — stats file → batch
ids → fetch → exact match — is cut at every link. A reviewer who retained v1
artifacts can still substring-match against them; no dataset built from a
public archive can prevent that, which is exactly why (4) is load-bearing
and is stated as such in the rubrics rather than hidden behind obfuscation.

### Windowing costs signal, and how much was measured before choosing

| Window | Exact matches | Reference ARI |
|---|---|---|
| none (v2 split, dedup only) | 100% | 0.298 |
| absolute 160–400 chars | 0 | 0.191 |
| keep 50–80% | 0 | 0.239 |
| keep 65–95% | 0 | 0.270 |
| **keep 80–97% (shipped)** | **0** | **0.278** |
| keep 90–99% | 0 | 0.278 |

A small absolute window destroyed a third of the reference's signal —
region length and the noise statistics that need enough characters to
estimate are both part of the fingerprint. Since windowing's only job is to
break byte-equality, the fraction is kept high. The reuse fix on its own
*raised* the v1-split reference (0.316 → 0.334), so none of the drop below
is from de-duplication.

## Review round 2: four findings, all fixed

The v2 build went back to review. Four findings, each confirmed before
anything was changed:

**1. The archive was still named.** Round 1 removed batch identities and
windowed the text, but the descriptions still named the digitization program
and (in the dataset description) its bulk-data URL. The OCR is real and
searchable there, so a rule alone does not close lookup while the search
index is handed over. Fix: the program and archive are not named anywhere in
solver-visible text — problem statement, dataset description, rubrics, and
the data directory. The citation lives in the raw-side `ATTRIBUTION.txt` and
in `config.yaml`, neither of which solvers see. Code comments in `prepare.py`
were genericised too, since the file ships to the platform. One reference is
forced: the platform's dataset validator fetches the Source URL and rejects
any that is unreachable or "does not match" — and every page of the program
except the static bulk-data root returns 403 to automated requests (the CC
Public Domain Mark deed and the program's about pages were all rejected).
The Source URL therefore points at the data root. The lookup closure does
not depend on hiding it: no batch identity is published (the corpus is 35
unnamed batches out of hundreds), labels are opaque, no shipped row is
byte-identical to any archive region, and lookup is prohibited by rule in an
offline solver environment.

**2. The leaderboard cut every bag in two.** `answers.csv` had one row per
snippet, and the leaderboard splits answers into public/private slices BY
ROW. Every bag therefore became two fragments, each graded as a bag of its
own. 24 public fragments had three rows or fewer — on a 3-row fragment whose
rows happen to come from three batches, an all-singleton guess is a perfect
partition and ARI gives 1.0 — so the all-singleton sample scored 0.048 on
the public board (0.000 private), and one solver run was exactly that. Fix,
structural: `answers.csv` and `submission.csv` are now **one row per bag**
(`bag_id, batch_id`, with `batch_id` holding one label per snippet in the
snippet order of `test.csv`). Any row-wise slice now keeps whole bags on
one board, every bag is scored against its full snippets and its full 3–4 true
groups, and the degenerate partitions score exactly 0 on every slice.
`test.csv` is unchanged. Verified: a row-wise half-split of the new
`answers.csv` scores the same per-bag ARI as the full file for the reference
submission, and the all-singleton sample scores 0.000 on both halves.

**3. Per-bag scoring does not block pooled clustering.** The problem
statement claimed that scoring each bag independently blocks a global
clustering pass over the pooled test set. It does not: a solver can cluster
all test rows at once and cut the result back into bags. With only 16
batches under the test bags the shortcut is real, if weak (v2: 0.089
realistic / 0.097 oracle-k against the 0.276 reference; v3: 0.103 / 0.105
against 0.326). Fix: the claim is
corrected in the description, and the shortcut is banned outright — a
"partition each bag on its own" rule in the problem statement and a
matching `[REQUIRED]` rubric. Building bags so that batches never repeat
across them is not possible with 16 test batches and was not attempted.

**4. Grader edge cases.** The grader kept the first of a duplicated row id
(so a repeated id scored as if correct) and raised an unhandled exception on
a row id without `::`. Both are handled cleanly in the per-bag grader, and
in line with the platform's degrade-never-reject rule (rejection register
§1.13): a duplicated `bag_id` is a conflict the grader will not resolve —
every row for it is discarded and that bag scores 0 (it can no longer score
as if correct, and nothing is raised); a malformed `bag_id` cannot crash the
grader since it is simply an unknown, ignored bag; a `batch_id` value that is
empty or has the wrong label count scores 0 for that bag. Only a missing
required column raises (`InvalidSubmission`, a `ValueError`).
`work/test_grade.py` covers each case (21 tests). Also fixed from the same review: the problem statement referenced
`config.yaml`, which is not shipped (references removed), and quoted only
the 0.276 reference while platform solver runs reached 0.40–0.43 (both
numbers now stated, with the reference described as a floor).

Re-running the ladder under the per-bag grader on the unchanged v2 data
gave every deterministic entry identical to four decimals — the partitions
were unchanged, only their packaging. (That re-run caught a harness bug
worth recording: one baseline labelled snippets with tuples, whose string
form contains a space, and the per-bag grader split them into two tokens —
every bag scored 0 until the harness was made to re-code labels as
integers. Solver-facing docs say labels must not contain whitespace; the
sample submission and validator show the shape.)

**5. Difficulty band (same round).** Platform solver runs on v2 landed at
0.40–0.43 against the 0.276 reference — the bottom edge of the target band,
and the agent/reference ratio (~1.5×) is the only proxy available locally
(register §6). The bag-shape knob was swept on the reference:

| groups × snippets per group | reference ARI |
|---|---|
| 3–5 × 3–8 (v2) | 0.276 |
| 3–4 × 3–8 | 0.295 |
| 3–5 × 5–8 | 0.289 |
| 3–4 × 5–8 | 0.296 |
| **3–4 × 6–10 (v3, shipped)** | **0.326** |
| 3–3 × 5–8 | 0.325 — rejected: a fixed group count is a structural tell |
| 2–4 × 5–8 | 0.328 — rejected: 2-group bags are weak instances |
| v2 shape, window 90–99% | 0.276 — windowing is not the lever |

More snippets per group gives each cluster more evidence; fewer groups per
bag keeps the decision set small; the group count still varies (62 bags of
3, 65 of 4). Expected solver band by the v2 ratio: ~0.47–0.51. Cost: 127
bags instead of 169 (3,568 test rows), since without-replacement sampling
exhausts the 16 test batches sooner. The whole ladder was re-measured on
v3 (table below): strictly ordered, reference on top, margin 0.22 over the
strongest adversary, and knowing the true group count no longer helps the
reference at all (0.310 vs 0.326 — `guess_k = n/6` already lands on 3–6 for
these bag sizes).

**6. Representative leaderboard slices (register §1.10).** With one row per
bag, bag *order* decides what each board sees. Bags are now sorted by
(group count, size) and spread along the file with a golden-ratio
permutation before ids are assigned; contiguous fifths of `answers.csv`
score 0.305–0.341 around the 0.326 full set for the reference (v2 order:
0.239–0.319 around 0.276).

**8. GPU leverage in the reference notebook (register §0).** The shipped
notebook previously ended on the CPU RandomForest. It now runs five
progressive submissions: singleton → unsupervised features → RandomForest
embedding (the deterministic graded reference, 0.326) → a
supervised-contrastive character-CNN trained on the GPU (CUDA on the A10G,
CPU fallback; ~0.13 alone) → RandomForest + CNN ensemble (~0.33, final;
the notebook ships whichever of the last two scores higher, so the final is
never below the reference). The encoder is inlined so the notebook is
self-contained on the platform. The CNN's gain is marginal and stated as
such — it is the GPU-native path the Compute section points solvers at,
and its weakness at this data scale is the headroom.

**7. Platform mechanics (register §1.4, §1.13).** Every markdown table in
the two solver-visible descriptions was converted to bullet lists (the
renderer drops pipe characters), and the duplicate-id policy was aligned to
degrade-never-reject as described in finding 4.

## Review round 3: three findings, all fixed

A third review found two real accuracy problems and asked for a difficulty
retune, all against the v3 build:

1. **Word TF-IDF was quoted as proof of content-blindness; it is not.**
   Batches vary in mean snippet length by roughly 10x (~290–3,050 characters,
   driven by article length and how much of the source region survived
   windowing), and one held-out batch is majority-Spanish. A trivial
   length-only feature (agglomerative clustering on character count alone)
   scored 0.093 on v3 — ABOVE the 0.079 word-TF-IDF number being used as the
   "noise floor" reference. Re-cutting snippets to a uniform length was
   considered and rejected: it would lower every measured score, including
   the reference's, without closing the confound (length still correlates
   with the digitization run that produced a snippet, since it derives from
   article length and windowing, both batch-correlated). Fixed by disclosure
   instead — both description files now state the length/language confound
   explicitly and its measured size, so a solver cannot be misled into
   reading the TF-IDF ablation as a content-blindness proof.

2. **Duplicate `bag_id` description bug.** The problem statement's
   "Rejected outright" list included "a submission that lists the same
   `bag_id` more than once," but `grade.py` (since review round 2) only
   zeroes that one bag — it does not raise `InvalidSubmission`. Fixed by
   moving the duplicate-bag_id case into the "costs points, not the run"
   list, matching actual grader behaviour (re-verified: 21/21 grader unit
   tests, including the duplicate-id case).

3. **Compute framing pointed at the wrong lever.** The Compute section
   claimed GPU acceleration was the load-bearing path ("this task benefits
   genuinely from GPU acceleration... a from-scratch neural encoder... has
   real... headroom"), and a rubric recommended it as "[RECOMMENDED] Uses
   the available GPU." The measured ladder says the opposite: the CPU-only
   tree-based RandomForest reference (0.326 on v3) beat the GPU char-CNN
   (0.127) on every build measured. Fixed by moving the task to CPU-only —
   `config.yaml`'s `environment` field, the Compute section, and the GPU
   rubric line were all rewritten to state plainly that the strongest
   approach measured is tree-based, not neural, and that no GPU is
   provisioned or required. The notebook's device selection
   (`"cuda" if torch.cuda.is_available() else "cpu"`) was already
   CPU-safe; only the surrounding prose was GPU-first.

**Difficulty band, again.** The best of three platform solver runs on v3
reached 0.38 against the 0.326 reference (ratio ~1.17x — well below the
~1.5x ratio seen on v2 and v1), short of the 0.40 band floor. The bag shape
was swept again on the reference, this time also varying group count
downward (not just snippets-per-group upward, which the v3 retune had not
tried):

| Shape (groups × snippets/group) | Reference ARI | n bags |
|---|---|---|
| 3-4 × 6-10 (v3, shipped) | 0.326 | 127 |
| 3-4 × 8-14 | 0.313 | 87 |
| 3-4 × 8-16 | 0.302 | 87 |
| 3-4 × 10-16 | 0.287 | 79 |
| 2-3 × 6-12 | 0.328 | 164 |
| 2-3 × 8-12 | 0.339 | 141 |
| 2-4 × 8-12 | 0.338 | 110 |
| **2-4 × 8-14 (v4, shipped)** | **0.342** | **110** |
| 2-3 × 8-14 | 0.352 | 111 |
| 2-3 × 10-16 | 0.321 | 111 |

Holding group count at 3-4 and only growing snippets-per-group made the
reference WORSE, not better (0.326 → 0.313 → 0.302 → 0.287) — larger groups
dilute per-bag classifier confidence and shrink the number of scoreable
bags faster than they add signal. Lowering group count helped consistently.
`2-3 × 8-14` scored highest (0.352) but was rejected: 53% of its bags
(59/111) would have been exactly 2 groups, making the dataset close to a
binary-clustering task rather than the "unknown group count, 2 to 4"
framing the task states — the same "2-group bags are weak instances"
concern the v3 retune had already flagged and avoided. `2-4 × 8-14` shipped
instead: reference 0.342, a genuine 2/3/4-group spread (33/32/45 bags), and
every other round-1/round-2/round-3 measurement re-verified clean on the
new data (pooled-clustering attack 0.094, length-only baseline 0.089, min
bag size 16 — all still well clear of their respective danger thresholds).

## Measured ladder (v4, shipped configuration; v3/v2/v1 alongside for the record)

| Strategy | v4 ARI | v3 ARI | v2 ARI | v1 ARI |
|---|---|---|---|---|
| all-singleton / all-same-group | 0.000 | 0.000 | 0.000 | 0.000 |
| random partition | 0.019 | 0.018 | 0.032 | 0.024 |
| artifact attack (position + length only, no content read) | 0.013 | 0.018 | 0.026 | 0.021 |
| content-only: char n-gram TF-IDF, per-bag (ablation) | 0.033 | 0.038 | 0.047 | — |
| content-only: word TF-IDF, per-bag (ablation) | 0.079 | 0.079 | 0.084 | — |
| length-only baseline (character count alone; round-3 finding) | 0.089 | 0.093 | — | — |
| adversarial global clustering, REALISTIC (silhouette-estimated k) — banned by rule | 0.094 | 0.103 | 0.089 | 0.072 |
| adversarial global clustering, ORACLE (true k given) | 0.094 | 0.105 | 0.097 | 0.139 |
| full hand-feature set, unsupervised per-bag clustering | 0.120 | 0.111 | 0.082 | 0.099 |
| char-CNN embedding (supervised-contrastive), per-bag clustering | 0.129 | 0.127 | 0.118 | 0.133 |
| single cheap noise scalar, k-means | 0.155 | 0.130 | 0.092 | 0.096 |
| closed-set argmax over the 19 train batches (rubric violation) | 0.194 | 0.210 | 0.203 | — |
| reference given the true group count (diagnostic) | 0.335 | 0.310 | 0.281 | 0.329 |
| **reference: RandomForest-probability embedding, per-bag clustering** | **0.342** | **0.326** | **0.276** | **0.316** |
| oracle (true batch_id revealed) | 1.000 | 1.000 | 1.000 | 1.000 |

Produced by `work/baselines_v2.py` on the shipped files; every number quoted
in `config.yaml`, the problem statement and the dataset description comes
from that one run. The reference's margin over the strongest adversary is
0.18–0.25 across every build. The content-only ablations sitting at the
noise floor while noise features reach 0.342 is the measured form of the
"this is not stylometry" claim — with the round-3 caveat that word TF-IDF
alone is not sufficient evidence of content-blindness, since the length-only
baseline (0.089) sits just below it; see Review round 3, point 1. Knowing
the true group count does not help the reference on v4 (0.335 vs 0.342, a
smaller gap than v3's 0.310 vs 0.326), so group-count estimation is not
where the difficulty lives — representation quality is. Platform solver
runs: 0.432 on v1, 0.40–0.43 on v2, best-of-three 0.38 on v3 (short of the
band floor, hence v4); v4's expected agent range has not yet been observed.

Note the char-CNN embedding (0.129) is still measurably WEAKER than the
simpler hand-crafted-feature RandomForest approach (0.342) at this data
scale, though both improved substantially after the bug fix. Its training
is not bit-reproducible across torch builds (±0.01 between runs). This is
reported honestly rather than smoothed over — with only 19 train batches, a
from-scratch contrastive character encoder does not yet out-learn a
classical-ML baseline on hand-engineered noise features, which is why the
task ships CPU-only (Review round 3, point 3) rather than GPU-first. This is
itself a legitimate, disclosed finding about where the current skill ceiling
sits, and leaves real headroom for a stronger encoder or more data to climb
further, which is what makes the task non-saturated.

## What is new here

- Ground truth is a real digitization pipeline's own identity, not a human
  annotation or another model's output — the same "grade against a recorded
  real system" principle as blind-spot/galaxy-blind-spot, in a domain (OCR
  pipeline forensics) with no prior benchmark found.
- The graded artifact is a partition recovered from content-disjoint items,
  open-set across pipelines never seen in training.
- The metric (per-bag ARI) and the split ratio (45% test by batch) were both
  set by adversarially attacking the design before freezing it, not by
  convention.

## Relation to prior work (cite distant antecedents only)

- PRNU/sensor-noise camera fingerprinting (Lukas, Fridrich & Goljan, 2006) is
  the nearest conceptual neighbour — recovering device identity from
  low-level imaging noise. This task inverts the axis (which processing
  PIPELINE handled real text, not which physical sensor captured an image)
  and the domain (OCR error patterns in text, not photographic sensor noise).
- OCR error analysis / post-OCR correction literature (ICDAR 2017/2019
  post-OCR competitions, HIPE-OCRepair) treats noise as something to REMOVE;
  this task treats the noise pattern itself as the signal to be classified.
- PAN author clustering (CLEF 2016/2017) is the closest STRUCTURAL
  antecedent: collections of documents, recover the partition by author,
  number of authors not given. The target differs (a writer, not a
  processing pipeline), the metric differs (BCubed F / MAP, not ARI), and
  this task deliberately confounds topic with the target by drawing every
  snippet in a bag from a different article — the measured content-only
  baselines (word / char n-gram TF-IDF) sit near the noise floor precisely
  because of that.
- Vogler et al., "Contrastive Attention Networks for Attribution of Early
  Modern Print" (AAAI 2023) attributes anonymous books to printers from
  damaged-type images. Same intellectual move — recover the producing
  apparatus from its systematic defects, in a historical-documents setting —
  executed in the image domain; this task does it from the OCR transcript
  after the images are gone.
