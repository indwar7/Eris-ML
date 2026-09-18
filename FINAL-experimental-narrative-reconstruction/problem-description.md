# The Order of Discovery

A Results section is not a list of findings. It is a **chain of experiments**,
and the chain has a direction. "We first showed that X is expressed at the
synapse — so we asked whether X binds Y — and having found that it does, we
tested whether the binding is required for Z." Each subsection answers a
question the previous one raised.

We have taken the Results section of a real paper, cut it into its titled
subsections, and shuffled them. Your task is to put the experiments back in the
order the authors ran them.

## Task

Each item is one paper. You receive its Results subsections — title plus text —
in a random order, each tagged with an opaque slot letter (`A`, `B`, `C`, …).
Return the slots in the authors' original sequence.

- Papers have 4 to 5 subsections. The slot letters were assigned **after**
  shuffling and carry no positional information.
- Figure and table numbers are masked to `<FIG>`. Explicit sequencing words —
  *first*, *next*, *finally*, *having shown*, *as described above* — are masked
  to `<SEQ>`. You cannot read the order off the page.
- The paper's scientific discipline is published as a feature.
- No paper appears in both train and test.

What remains is the **logic of the science**: which claim presupposes which.
A subsection that characterises a protein's binding partner cannot come before
the one establishing the protein is present. A control experiment ruling out
an alternative explanation follows the result it protects. A mechanism study
follows the phenotype it explains.

## Why this is a reasoning task, not a style task

Every Results subsection reports statistics, so the numeric-density cue that
separates Results from Introduction is useless here — all the items are
Results. Every subsection is in the same register, from the same authors, in
the same paper. There is no stylistic tell. The only signal is
**presupposition**: reading two claims and judging which one the other depends
on. That requires understanding what the experiments *mean*.

## Dataset

Discover the data directory rather than assuming a fixed path:

```python
from pathlib import Path

DATA = next(
    p for p in (Path("/data"), Path("dataset/public"),
                Path("/kaggle/input/experimental-narrative-reconstruction/public"))
    if (p / "test.csv").exists()
)
```

Both splits share one flat layout — **one row per paper**:

- `train.csv` — columns `paper_id, field, n_slots, slot_A_title, slot_A_text,
  … slot_E_title, slot_E_text, true_order`. Slot columns beyond `n_slots` are
  blank. `true_order` (e.g. `"CADB"`) is the answer.
- `test.csv` — the same columns without `true_order`.
- `sample_submission.csv` — columns `paper_id, predicted_order`, returning
  each paper's slots in the shipped (shuffled) order. Valid; scores chance
  (≈0.22).
- `dataset_stats.json` — paper counts, slots-per-paper distribution, field
  distribution, and the chance level.

`paper_id` is opaque. No DOI, article title, author, or journal is shipped
anywhere in the data directory.

## Rules

**Use only the data directory.** The subsections are real published text and
the publisher's full-text search is public. Recovering the order by locating
the source article — by subsection title, by a phrase, by any key — and
reading its document order is not solving the task; a submission built that
way is invalid regardless of its score, and so is any submission that depends
on data not shipped in the data directory. The solver environment is expected
to have no network access; your solution must run to completion from the data
directory alone.

## Evaluation

Submissions are scored using **mean per-paper position accuracy** — for each
paper, the fraction of its subsections you placed at exactly their true
position, averaged over papers:

```python
def score_paper(predicted: str, true: str) -> float:
    return sum(p == t for p, t in zip(predicted, true)) / len(true)
    # 1.0 = every experiment in its place; 1/n = chance
```

**Why position accuracy rather than exact match or Kendall's τ.** Ordering
five experiments exactly right is 1-in-120 by chance; an exact-match metric
would give almost every submission zero and no gradient to improve along.
Position accuracy is graded — three of five experiments in place earns 0.6 —
so partial recovery of the logic is rewarded. Kendall's τ was considered and
rejected because its chance level is 0.5 by construction, which leaves the
bottom half of the scale unreachable and compresses every real submission
into the top half. Position accuracy is also deliberately stricter than τ in
one way: a narrative that is entirely right *except* shifted by one step
scores 0, because no experiment is where the authors put it. Recovering the
arc is not enough; each experiment has to land in its slot.

**Chance is 1/n per paper** — a random permutation, or the shuffled input
returned unchanged, places each subsection correctly with probability 1/n.
Over the shipped test split (papers of 4 and 5 subsections) that averages to
**≈0.22**. Any score above it reflects genuine ordering signal; the gap to 1.0
is the headroom.

## Submission

| Column | Type | Description |
|---|---|---|
| `paper_id` | str | Copy from `test.csv` |
| `predicted_order` | str | That paper's slot letters in your predicted sequence, e.g. `"CADB"` |

**Requirements**

- One row per `paper_id` in `test.csv`.
- `predicted_order` must be a permutation of exactly that paper's slots.
- Include a header row.

**Structural slips cost points, not the run.** A missing paper, or a
`predicted_order` that is not a permutation of that paper's slots, scores 0
for that paper only — no position can be verified. Only a submission missing
required columns is rejected.

## Compute

This task benefits genuinely from GPU acceleration. The natural approach is a
**pairwise or listwise model**: a cross-encoder that reads two subsections and
predicts which comes first, trained on the ordered training papers, then
decoded into a full permutation. That is a transformer over thousands of
subsection pairs — real GPU work — and it has substantial headroom over any
feature-based heuristic. You have a 1x NVIDIA A10G GPU, 10 CPU cores, and
62 GB RAM, with a 1.5 hour runtime budget.
