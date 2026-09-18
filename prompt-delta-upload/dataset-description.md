# Dataset Description

## Overview

The raw data is a grid of measurements taken by running one open-weights
instruction-tuned language model, SmolLM2-1.7B-Instruct, over a hand-written
set of prompts and recording, for every response, sixteen numeric
properties of how it answered - never the response text itself.

Twelve everyday task framings were written by hand (answering questions,
replying to a customer, tutoring a student, summarizing a finding,
reviewing code, and so on). Each framing has a base instruction and, for
six kinds of input (factual, ambiguous, unanswerable, reasoning, harmful,
creative), four hand-written probe questions. Eleven single-sentence edits
were written by hand and applied to each base instruction one at a time -
"respond only in JSON", "think step by step first", "explain it to a
five-year-old", and so on.

For every (framing, edit or no-edit, input kind, probe question)
combination, the edited or unedited instruction plus the probe question was
sent to the model with greedy decoding, and the response was reduced to
sixteen numbers: word count, character count, sentence count, mean sentence
length, lexical diversity, whether it looks like JSON, whether it uses
bullet points, how many bullets, whether it admits not knowing, whether it
refuses, how many hedging words it uses, how many first-person words,
how many questions it asks back, how many digits it uses, its ratio of
capital letters, and how many line breaks it produces.

No response text is stored or released anywhere in this dataset. The
framings, the edits, the input kinds and the probe questions are all
hand-written; the model is used only as an instrument being measured, and
the recorded numbers are the measurement.

Model: HuggingFaceTB/SmolLM2-1.7B-Instruct, licensed Apache-2.0. Apache-2.0
governs the model's weights and code, not the text it generates; the model
card places no claim or restriction on outputs. The recorded measurements
are therefore an original, generated artifact, released here under CC0 1.0
Universal - no rights reserved, no attribution required.

Decoding was greedy (do_sample=False) with a fixed seed, so the raw file is
reproducible bit-for-bit on the same hardware.

## File Structure

This dataset upload contains exactly these files at the top level of the
raw/ directory:

- fingerprints.json - one JSON object holding every recorded measurement.
- grid.json - the twelve framings, eleven edits, six input kinds and their
  probe questions, exactly as used to build fingerprints.json.
- meta.json - build metadata: model id, seed, generation count, elapsed
  time, device, and a sha256 of fingerprints.json for integrity checking.

The generator script that produced these files, generator/gen_raw.py, is
included in this upload. It is self-contained and deterministic: given the
same model weights, seed and hardware, it reproduces fingerprints.json
byte-for-byte.

## Features

### fingerprints.json

- schema (int): Format version.
- model_id (str): HuggingFaceTB/SmolLM2-1.7B-Instruct.
- model_licence (str): Apache-2.0 (weights); no restriction on outputs.
- dataset_licence (str): CC0-1.0.
- seed (int): Fixed generation seed.
- max_new_tokens (int): Generation cap per response, 72.
- decoding (str): greedy (do_sample=False).
- fingerprint_fields (list of str): The sixteen measurement names, in
  the order every measurement vector uses.
- ops (list of str): All twelve conditions, including the unedited
  baseline "none".
- edit_ops (list of str): The eleven edit operations, excluding "none".
- slices (list of str): The six input kinds.
- families (list of str): The sixteen prompt framings (twelve used in
  this upload's problem plus four added for split balance; see Task
  Construction).
- probes_per_slice (int): Probe questions per input kind, 4.
- n_generations (int): Total responses measured, 4,608.
- responses (object): Keyed "family|op|slice|probe_index", each value a
  list of 16 floats in fingerprint_fields order. This is the entire
  measured dataset; everything in public/ and private/ is derived from
  this object plus grid.json.

### grid.json

- families (list of objects): Each has family_id, base (the hand-written
  instruction), and probes (an object mapping each of the six input
  kinds to its four hand-written probe questions).
- ops (object): Each of the twelve condition names, self-mapped for
  reference.
- slices (list of str): The six input kind names, in fixed order.

### meta.json

- generated_by, seed, model_id, n_generations, fingerprints_sha256,
  elapsed_seconds, device.

## Task Construction

The task's items are built by prepare.py from fingerprints.json and
grid.json:

- Sixteen prompt framings, eleven edit operations, and 15 ways to pick
  four of the six input kinds give a maximum of 2,640 candidate items.
- For each item, the true ORDERING of the four candidates (most- to
  least-affected by the edit) is computed from the per-probe fingerprints
  -- not the released per-slice averages alone (see Measured Difficulty).
- Within each framing, items are capped per which candidate lands FIRST in
  the true ordering, so no single input kind dominates every item's top
  spot (a first, unbalanced build let a solver that always guessed the
  framing's single most-common top candidate score well above chance;
  capping closed that).
- 1,900 items survive balancing: 1,544 train, 356 test.
- Of the sixteen raw fingerprint fields, one (is_json) is constant at 0.0
  across every released fp_A__ column (this model never spontaneously
  produces JSON-shaped output on an unedited prompt) and is dropped from
  the fp_A__ side only; fp_B__ keeps all sixteen, since some edit
  operations do trigger JSON-shaped output. A handful of other fields were
  constant, or dominated by one value in 97%+ of rows, within one split by
  chance, and were dropped from both sides for that build (see prepare.py's
  column-count printout at build time).
- Four fields whose raw per-probe value is strictly binary (is_json,
  has_bullets, says_idk, refuses) become, after averaging over the 4
  probes per slice, a small ordinal set of fractions (0, 0.25, 0.5, 0.75,
  1.0). Numerically these sit at 0 for most items with an occasional
  nonzero value, which is real signal (a count out of 4 probes) rather
  than a continuous measurement, but reads as an extreme-outlier-heavy
  column to a naive numeric scan. These four fields are therefore released
  as "k/4" strings (e.g. "2/4") rather than floats; solution.ipynb shows
  how to parse them back to a 0.0-1.0 fraction for modeling.
- The remaining numeric fingerprint columns are winsorized: clipped to
  their [15th, 85th] percentile, computed from train and test COMBINED
  (this affects only the clip range applied to every value in a column --
  it never lets one row's value influence another row's engineered
  features, and the rank_1..rank_4 targets are computed upstream of this
  step and are untouched by it). A handful of fields (n_bullets, n_digits,
  n_first_person, n_hedges, caps_ratio, lex_div) are naturally right-skewed
  count/ratio measurements with a long tail; winsorizing compresses that
  tail (the event still reads as "large" post-clip) without discarding the
  field.
- Six columns still carry a genuinely non-degenerate tail on one split or
  the other after winsorizing (n_digits, n_bullets, n_first_person, and
  lex_div on specific slots) and are dropped outright rather than clipped
  further, since more aggressive clipping was measured to degrade signal
  broadly with diminishing returns on the outlier share. This is a small,
  targeted loss (6 of ~130 feature columns) rather than a global one.

## Split Design

The split is by **prompt framing**, not by row. Every item built from one
framing lands entirely in train or entirely in test. Thirteen framings are
used for training; three (analyst, productcopy, travel) are held out
entirely for testing, chosen by exact search over which three framings
land the test/train item-count ratio inside 15-25% while also keeping a
majority-vote shortcut (always guessing the train-wide most common top
candidate) near chance on the resulting test set - both properties
measured, not assumed.

With the framing-disjoint split, a solver has never seen the held-out
framings' baseline behaviour during training and cannot look an answer up;
it has to learn how an edit interacts with an input kind in a way that
transfers across framings.

## Measured Difficulty

Positional-credit score on the held-out test items, computed with the
task's own grader: full credit if a submission's ordering places the true
most-affected candidate first, partial credit if second, none if third or
fourth (see grade.py).

- Copy the four candidates through in their given order, unranked: 0.337
- A random permutation of the four candidates: 0.350
- Order candidates by plain distance on the released measurements directly
  (the shortcut an earlier, simpler ordering rule matched almost exactly,
  before being closed - see prepare.py's "LABEL RULE"): 0.418
- A ridge regression trained on the released measurements: 0.493

The gap between the naive baselines (~0.34-0.35) and the trained solver
(0.493) is the actual skill this task measures. The gap between the
distance-recomputation shortcut (0.418) and the trained solver is smaller
but real, and both sit well below a perfect score - there is room for a
better solution above 0.493 without the task being solvable by formula
alone.

## Intended Use

This dataset is built for one task: given the released before/after
behaviour fingerprints for a prompt edit, submit an ORDERING of four shown
input kinds from most- to least-affected by the edit (see
problem-description.md). The submitted answer is a permutation of a
per-row candidate set, scored by where the true top candidate lands in
that ordering, so a solution to this task is a ranking/ordering-recovery
model, not a single-label classifier - the description does not claim
otherwise. It is not intended as a general-purpose corpus of model
responses, a benchmark of SmolLM2's overall capability, or a source of
training data for other tasks; no response text is released, which alone
rules out most other uses.

## Known Limitations

- **Responses were capped at 72 generated tokens.** Longer or shorter caps
  could shift some behaviour measurements (especially n_words, n_chars,
  n_sents) and would very likely change the true ordering (rank_1..rank_4)
  for at least some items. The cap was fixed for the entire build, so it
  is consistent within this dataset, but the released numbers should not
  be read as the model's unconstrained behaviour.
- **One model only.** Every measurement comes from a single 1.7B-parameter
  instruction-tuned model (SmolLM2-1.7B-Instruct). Which input kind an
  edit operation moves most is not claimed to generalize to other models,
  other model sizes, or other instruction-tuning recipes - it is a
  property measured on this model, on this prompt grid, under greedy
  decoding.
- **Greedy decoding is deterministic per-hardware, not necessarily
  bit-identical across hardware.** The raw file was generated once, on
  one GPU, with a fixed seed; floating-point non-associativity means a
  regeneration on different hardware (a different GPU model, a different
  CUDA/driver version) could produce slightly different token-level
  outputs and, in turn, slightly different fingerprint values, even
  though the generation procedure itself is fully deterministic given
  fixed hardware. The shipped raw/fingerprints.json is the authoritative
  artifact; prepare.py derives train/test/answers from it directly and
  does not regenerate anything.
- **Sixteen fingerprint fields are surface statistics, not a semantic
  measure.** They count things like word length, bullet usage, and
  hedging phrase frequency; they do not verify factual correctness,
  measure reasoning quality, or capture meaning. Two responses with
  identical fingerprints could differ substantially in content, and this
  dataset cannot distinguish that.
- **All sixteen prompt framings and eleven edit operations were
  hand-written for this task**, not drawn from a broader corpus of
  real-world prompts - so the diversity of framings and edits is
  representative of the categories chosen here, not of prompt-engineering
  practice in general.
