# The Edit That Moved the Answer

## Overview

Two prompts are shown, A and B, where B is A with exactly one edit applied
from a known vocabulary of eleven edit operations - things like "ask for
JSON", "add a persona", "tell it to hedge", "tell it to be confident". Both
prompts were sent to the same instruction-tuned language model against six
different kinds of input: factual questions, ambiguous ones, unanswerable
ones, reasoning problems, harmful requests, and creative prompts. For every
response the model gave, fifteen numeric behaviour measurements were
recorded - how long it was, whether it used bullet points, whether it
hedged, whether it refused, and so on. No response text is released, only
these measurements.

Your task is to rank four of the six input types by how much the edit
changed the model's behaviour on each one, from most affected to least.

## Task

Each row shows four of the six input types (a different four each time),
the model's behaviour measurements on each one before the edit and after
the edit, and which edit operation produced B. You submit an ORDERING of
those four input types, from the one you believe moved most to the one
that moved least.

The edit operation is given to you; recovering the ordering of effect is
the graded half. Some edit operations are trivially detectable from
surface formatting alone (asking for JSON almost always produces JSON),
but which input types that same operation actually shifts, and in what
order, is a separate, harder question - detecting the operation tells you
almost nothing about the ordering.

The split is by **prompt family**, not by row. Three everyday framings -
out of sixteen total - are held out entirely for testing. Every
measurement you train on comes from a different set of framings than the
ones you are tested on, so nothing about a specific family's baseline
behaviour can be memorized and looked up.

## What makes this hard

**The true ordering is not a distance you can recompute from what you are
given.** An earlier version of this task ordered candidates by comparing
the released measurements before and after the edit, directly - which
meant a solver that recomputed that exact same comparison matched the
answer key's top choice on essentially every held-out item. That version
was scrapped. The true ordering now also depends on how consistent the
model's behaviour was across the several questions asked per input type, a
property that is not itself released - only the average is shown to you.
Recomputing distance on the released averages alone gets the true top
choice right less than half the time.

**Two different signals point in different directions.** Whether an edit
operation is easy to detect from formatting, and which input type it
actually reshapes, are close to independent. An edit that is instantly
recognisable from its output format can still be the one that barely moves
any of the four candidates, while a subtle edit with no visible formatting
signature can be the one that moves a candidate the most.

**Sixteen prompt families, eleven edits, six input types.** A pattern that
holds for a customer-support framing does not necessarily hold for a
meeting-minutes framing. What has to generalize is how an edit interacts
with an input type, not a memorized fact about one specific framing.

Measured on this data with the official grader (credit for placing the
true most-affected input type correctly in your submitted ordering; see
Metric below):

- Copy the four candidates through in their given order, unranked: 0.337
- A random permutation of the four candidates: 0.350
- Order candidates by plain distance on the released measurements: 0.418
- A ridge regression trained on the released measurements: 0.493

## Data files

All public files are in ./dataset/public/

**train.csv** - 1,544 rows, one row per item.

- item_id (str): Unique row id.
- family (str): Which of the sixteen training prompt framings this item
  came from.
- edit_op (str): Which edit operation produced prompt B from prompt A.
- slice_1..slice_4 (str): The four input types shown for this item, out of
  the six that exist. A different four each time, in no particular order.
- fp_A__\<field\>__s1..s4 (float): Behaviour measurements for prompt A, for
  each of the four shown input types.
- fp_B__\<field\>__s1..s4 (float): The same measurements for prompt B, for
  each of the four shown input types.
- rank_1..rank_4 (str): Target. slice_1..slice_4 reordered from most- to
  least-affected by the edit.

**test.csv** - 356 rows. Same columns without rank_1..rank_4.

**sample_submission.csv** - the required format, filled by copying
slice_1..slice_4 through unreordered. It scores 0.337.

## Metric

**Positional credit**: for each test row, find where the true
most-affected input type (the true rank_1) landed in your submitted
ordering, and award credit by its position there - full credit in first
place, partial credit in second, none in third or fourth. Average over all
356 test rows.

```python
CREDIT = [1.0, 0.33, 0.0, 0.0]

def positional_credit(pred_rank, true_rank):
    true_top1 = true_rank[0]
    if true_top1 not in pred_rank:
        return 0.0
    return CREDIT[pred_rank.index(true_top1)]
```

Range: 0.0 to 1.0, higher is better. A submitted ordering that is not a
valid permutation of that row's four shown candidates (a repeated entry, a
name outside the four, a missing entry) scores 0.0 for that row rather
than being rejected.

## Submission format

Write a CSV named submission.csv to ./working/ with exactly five columns:

```
item_id,rank_1,rank_2,rank_3,rank_4
I00001,harmful,ambiguous,creative,factual
I00002,ambiguous,reasoning,creative,factual
...
```

- One row for each item_id in test.csv, 356 rows plus a header, in any
  order.
- rank_1..rank_4 together must be a permutation of that row's four shown
  candidates (slice_1..slice_4 in test.csv); any other combination scores
  0.0 for that row, never rejected.
- An item_id you omit scores 0 for that row rather than voiding the
  submission; duplicated ids keep their first row and unknown ids are
  ignored. Only an unreadable file or missing rank_1..rank_4 columns are
  unscorable.

## Rules & environment

- Learn from the provided data only. No pretrained models, no LLMs, and no
  external data of any kind - the released measurements per slice per side
  are the entire feature set.
- Use only libraries available in the Kaggle Python Docker image (pandas,
  numpy, scikit-learn, xgboost, lightgbm, tensorflow, pytorch and similar).
- No LLM-generated outputs may be used anywhere in your solution.
- Your notebook must run end to end, top to bottom: load the public data,
  fit whatever you fit, then write ./working/submission.csv
- Seed everything; your run should be reproducible.
