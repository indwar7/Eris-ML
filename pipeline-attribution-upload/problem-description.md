# Recovering OCR Batch Origin from Character Noise

**Domain: NLP — unsupervised text clustering / partition recovery.** Given a
bag of unlabeled text snippets, group them by shared origin — the same
family of problems as document clustering or authorship clustering, applied
here to OCR provenance instead of authorship. There is no single-item
anomaly score and nothing is flagged as outlying; every snippet in a bag
belongs to some group, and the task is to recover the grouping.

Real historical newspapers were digitized in batches by a national,
multi-decade newspaper-digitization program. Different batches
were processed by different digitization pipelines — different OCR engines,
engine versions, vendor settings, and scanning eras — over many years. Your
task is to recover which digitization batch produced each piece of OCR'd
text — using only the systematic character-error fingerprint that batch's
pipeline left behind, never the actual words.

## How this differs from adjacent tasks

- **Not OCR accuracy / quality scoring** — nothing here is graded against a
  corrected transcript; the OCR is never cleaned or corrected, and its
  errors are the signal, not the thing being measured.
- **Not authorship or stylometry clustering** — the grouping variable is a
  *processing pipeline*, not a *writer*, and the design deliberately blocks
  the authorship shortcut: every snippet in a bag is drawn from a different
  article, so topic and voice carry no grouping signal.
- **Not closed-set classification** — the 16 test batches never appear in
  training; there is no fixed label set to predict into, only an unseen
  grouping to recover.
- **Not a fixed-k clustering benchmark** — the true number of groups is
  never given and varies bag to bag (2 to 4 here), so a solution can't
  assume or be handed the cluster count.

## Task

You are given `bags` of text snippets. Every snippet in a bag is drawn from
a DIFFERENT real newspaper article, so the article's subject matter tells you
nothing about which snippets belong together — only the OCR noise pattern
does. Some unknown number of snippets in a bag came from each of several real
digitization batches, and your job is to recover that grouping.

- You do not need to name which real batch a group came from — only which
  snippets travelled together.
- The number of true groups in a bag is not given and varies bag to bag.
- Each snippet's text is genuinely noisy, uncorrected OCR output: character
  substitutions, junk characters, and broken word boundaries are real
  artifacts of period OCR engines running on period newspaper print.

## Why this is not a text-generation or classification task

This is a program-writing and modeling task, not a text-generation task. You
are not asked to clean the text, name a batch, or classify a snippet into a
known category — every test batch is entirely unseen in training. You are
asked to recover a **partition**: which items share a hidden real-world
grouping variable. Write a fallback `submission.csv` (every snippet in its
own group is a safe, valid, scoreable starting point) before you start
building your real approach.

## Dataset

Discover the data directory rather than assuming a fixed path — it may be
mounted differently across environments:

```python
from pathlib import Path

DATA = next(
    p for p in (Path("/data"), Path("dataset/public"),
                Path("/kaggle/input/pipeline-attribution-dataset/public"))
    if (p / "test.csv").exists()
)
```

Files in the data directory:

- `train.csv` — columns `row_id, text, batch_id`. Every row is labeled
  with the digitization batch it came from. `batch_id` is an **opaque label**
  (`batch_00`, `batch_07`, …) that is consistent per real batch but is not
  the archive's own batch name — the label's value carries no information,
  only its grouping does. `row_id` is likewise an opaque per-row index
  (e.g. `train_00000`); it exists only so every file shares the same
  identifier-first schema. Use `text`/`batch_id` however you like: train a classifier, an embedding model, or
  hand-engineer features — the point is to learn what makes one pipeline's
  OCR noise look different from another's, in a way that generalizes to
  pipelines you have never seen labeled examples from.
- `test.csv` — columns `row_id, text`. No labels. `row_id` is formatted
  `{bag_id}::{opaque sequence number}` — split it on `::` to recover the bag
  a row belongs to; the part after `::` is the snippet's position within
  its bag (0, 1, 2, …) — it identifies nothing about the snippet's origin,
  and it is the order in which your submission lists that bag's labels. `bag_id` is deliberately
  the *only* thing `row_id` reveals: keeping every file's column set minimal
  (`row_id` + one payload column) means `test.csv` and `answers.csv` share
  only the join key, never a duplicated identifier or a disguised label. The
  withheld target is the batch of each row — present only in
  `private/answers.csv` (one row per bag: `bag_id, batch_id`), never in
  `test.csv`.
  Rows whose `row_id` shares a `bag_id` prefix are the snippets you must
  partition together. A bag's rows are listed in shuffled order — position
  carries no information.
- `sample_submission.csv` — columns `bag_id, batch_id`, **one row per bag**.
  A valid, minimum-viable submission (every snippet placed in its own group:
  `batch_id` = `"0 1 2 … n-1"`). It scores 0.000, not an error.
- `dataset_stats.json` — counts and split statistics for the shipped data.
  Counts only: no batch identity, train or test, is published anywhere in
  the data directory.

Every test row is a distinct source snippet — no snippet appears twice in
`test.csv` — so each row contributes to exactly one bag's score. Shipped
text is a contiguous window of the source OCR region (most of it, drawn
from a private seed), not the region verbatim; the OCR noise the task is
about is a property of the character stream and is unaffected.

## Rules

**Use only the data directory.** The text is real, uncorrected OCR from a
public archive, and searchable there. Recovering the grouping by matching
test rows back to that archive, or to any external copy or derivative of it,
is not solving the task — a submission built that way is invalid regardless
of its score, and so is any submission that depends on data not shipped in
the data directory. The solver environment is expected to have no network
access; your solution must run to completion from the data directory alone.

**Partition each bag on its own.** Only 16 batches underlie the 110 test
bags, so pooling every test row, clustering the whole test set globally, and
cutting the result back into bags is a transductive shortcut — it recovers
corpus structure, not a generalisable notion of pipeline similarity. It is
prohibited: when you partition a bag, the only test rows you may use are
that bag's own. Using `train.csv` however you like (classifier, metric
learning, embedding) is the intended path; using other *test* bags' rows is
not. This shortcut was measured during construction at 0.094 ARI
(below the 0.342 reference) and is banned as a matter of rule, not because
it is strong.

## Evaluation

Submissions are scored using **mean per-bag Adjusted Rand Index (ARI)**,
clipped to `[0, 1]` and averaged across all test bags.

```python
from sklearn.metrics import adjusted_rand_score

def score_bag(true_batch_ids: list, submitted_groups: list) -> float:
    return max(0.0, adjusted_rand_score(true_batch_ids, submitted_groups))
```

**Why ARI, and why chance-correction matters here.** The task is partition
recovery, not classification against fixed class names — a bag's true group
count and identities are unknown to the solver, so any metric has to compare
two *groupings* of the same snippets rather than two label vectors. Plain
pairwise co-membership agreement (e.g. F1 over "same group?" pairs) was tried
first and rejected after direct measurement on this dataset: putting every
snippet in a bag into one giant group scored **0.376** under raw pairwise
F1 — HIGHER than a real trained baseline's 0.289 — because most snippet
pairs in a bag are correctly *not* co-members by chance alone, and an
uncorrected metric rewards that chance agreement instead of penalizing it.
ARI subtracts out exactly this expected-by-chance agreement, which is why
both degenerate submissions (one giant group, or every snippet its own
group) score **0.000** here rather than a misleadingly positive number.

**Why average per-bag rather than pool all snippets into one global score.**
Bags vary in size (16 to 50 snippets) and in how many true batches they
contain (2 to 4, by construction — see Data). Pooling every snippet across
all 110 bags into one flat ARI computation would let large bags dominate the
score. Scoring each bag in isolation and averaging means every bag counts
equally regardless of size. Note what per-bag scoring does *not* do: it does
not by itself stop a solver from clustering the pooled test set and slicing
the result back into bags — that is why the Rules above prohibit it
explicitly.

**The signal is noise, not content — measured.** Because every snippet in a
bag comes from a different article, approaches that read what the text *says*
rather than how it is *garbled* were measured to fail: word-level TF-IDF
clustering scores **0.079** and character n-gram TF-IDF scores **0.038** —
both at the noise floor — against **0.342** for the shipped reference solution
built on content-free OCR-noise features. A closed-set classifier that simply
assigns each test snippet to its most likely *training* batch — ignoring that
every test batch is unseen — reaches only **0.210**. The separation is real,
and it comes from the pipeline fingerprint rather than the subject matter.

**Word TF-IDF is a weak baseline, not a content-blind one — snippet length is
a real confound.** Batches vary in mean text length by roughly 10x (measured:
~290 to ~3,050 characters per snippet, driven by article length and how much
of the source region survived windowing), and one held-out batch is
majority-Spanish. A trivial length-only feature (clustering on character
count alone) scores **0.089** per bag — above the word-TF-IDF number above.
This does not mean the task reduces to length or language: length-only is
still well below the 0.342 reference and gives no bag-to-bag consistency an
agent can build on, but it is a real, measured shortcut and not fully blocked
by the current windowing. Solutions that lean on snippet length or language
identity rather than character-level noise statistics will not generalise
across the held-out batches the way the reference approach does.

- Your submission's `batch_id` labels are your **guessed groups**, not real
  batch names. Values are entirely up to you — any tokens. They only need to
  be consistent WITHIN one bag (two snippets sharing a label means "I believe
  these came from the same batch"). Reusing the same label across different
  bags has no special meaning — every bag is scored independently.
- ARI is chance-corrected: putting every snippet in one group, or every
  snippet in its own group, both score close to 0.000. This is intentional —
  those submissions carry no real information.
- No group-count guess is required to be exactly right; ARI degrades
  gracefully with imperfect partitions rather than requiring an exact match.

## Submission

Submit a CSV file named `submission.csv` with **one row per bag** and the
following columns:

- `bag_id` — the bag (the part of `row_id` before `::`), copied from
  `test.csv`.
- `batch_id` — your predicted group label for each snippet of that bag,
  **whitespace-separated, in snippet order** (the integer after `::`:
  position 0 first, then 1, 2, …). Labels are any tokens without whitespace;
  two snippets sharing a label means "same batch". Labels only need to be
  consistent within one bag.

Example: a bag `bag_0007` with snippets `bag_0007::0` … `bag_0007::4` that
you believe come from three batches might be submitted as
`bag_0007,"0 1 0 2 1"`. `sample_submission.csv` shows the exact shape.

**Requirements**

- Exactly one row per bag in `test.csv` — **110 rows** plus the header — each
  `bag_id` exactly once.
- `batch_id` must contain exactly as many labels as the bag has snippets.
- Include a header row.

**Rejected outright** (the run does not score): a submission missing a
required column.

**Everything else costs points, not the run:**

- A missing bag scores 0 for that bag only; the rest is still graded.
- A `batch_id` value that is empty or has the wrong number of labels scores 0
  for that bag only.
- A `bag_id` listed more than once: every conflicting row for that bag is
  discarded and the bag scores 0 — the grader does not pick one of the
  duplicates. The rest of the submission is still graded normally.
- Rows for bags not in the current grading slice are ignored.

## Compute

This task runs on CPU. The strongest measured approach is a tree-based model
over hand-engineered character/noise features (the shipped reference scores
0.342 this way); a from-scratch neural encoder was also measured during
construction and did not beat it (~0.13 alone, see the reference notebook).
You have 10 CPU cores and 62 GB RAM, with a 1.5 hour runtime budget. No GPU
is provided or required.
