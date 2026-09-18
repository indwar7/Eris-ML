# The Accent Translator

You have a pronunciation. Not a spelling — an actual transcription, as a
speaker from one place says it. *Bath* from London: `/bɑːθ/`. Now say it as
someone from Chicago would. You don't start from scratch; you already know
the sounds. You need to know **which of them change, and into what**.

That is this task. Given a word's transcription in one accent, rewrite it for
another.

## Task

Each row gives you:

- `word` — `bath`
- `src_accent` — `RP`
- `source_pronunciation` — `bɑːθ`
- `tgt_accent` — `GA`

Output the phoneme string in the target accent: `bæθ`.

- Accents: `RP` (Received Pronunciation), `GA` (General American), `AU`
  (Australian), `NZ` (New Zealand). All twelve directions occur.
- **89% of rows require a real change.** The rest are identical across the
  two accents — and knowing *not* to rewrite is scored too.
- Test words are unseen in training. You learn the rules, not the lexicon.

## What you are actually being asked

Not grapheme-to-phoneme. You already have the phonemes. The question is the
**transfer function between two dialects**: which segments are stable, which
shift, and what governs the shift. Post-vocalic `/ɹ/` disappears going RP→GA?
No — it *appears*. RP's `/ɑː/` in *bath* becomes GA `/æ/`, but the same
vowel in *father* does not. NZ centralises `/ɪ/`; AU fronts the vowel of
*goat*. Each is a context-sensitive rule you must recover from examples and
apply to words you have never seen.

## Evaluation

**Change-segment accuracy.** For each row, the true target is aligned to the
source and the segments that differ are identified — those are the accent's
rewrites. Your prediction is scored only on those positions:

```
per-row = (changed segments you got right) / (changed segments)
```

Rows with no change score 1 if you leave the source untouched, 0 if you alter
it. Mean over rows.

**Why not edit similarity.** About 80% of any transcription survives an accent
change. Under edit similarity, copying the source and doing nothing scores
≈0.80, leaving a 0.2 band to distinguish every real model. Scoring only the
changed segments sends copy-source to ≈0.11 (credit only from no-change rows)
and spends the full range on the one thing that matters: do you know the
rewrite.

Baselines, measured on the shipped test set:

- copy the source through — **0.112** (the sample submission)
- copy the spelling — 0.088
- constant `ə` — 0.108

Stress, length and syllable marks are stripped from all strings before
alignment; emitting them neither helps nor hurts.

## Dataset

```python
from pathlib import Path
DATA = next(p for p in (Path("/data"), Path("dataset/public"),
                        Path("/kaggle/input/accent-translator/public"))
            if (p / "test.csv").exists())
```

- `train.csv` — `row_id, word, src_accent, source_pronunciation, tgt_accent, phonemes`
- `test.csv` — the same without `phonemes`
- `sample_submission.csv` — `row_id, phonemes`, source copied through (≈0.11)
- `dataset_stats.json` — counts, direction distribution, share of rows needing change

Word-disjoint split. `row_id` is opaque.

## Submission

- `row_id` — from `test.csv`
- `phonemes` — predicted target-accent phoneme string

One row per `row_id`, header, UTF-8. Missing or blank rows score 0; only
missing columns reject.

## Compute

An encoder–decoder over `<src_accent> <tgt_accent> source-phonemes`, trained
to emit the target string, is the natural model, and the change-segment
metric rewards exactly the context-sensitive rewrites it can learn that rule
tables cannot. 1x NVIDIA A10G, 10 CPU cores, 62 GB RAM, 1.5 h.
