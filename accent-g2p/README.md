# The Accent Translator

**Domain:** natural-language-processing · **Task type:** sequence-to-sequence
**Metric:** change-segment accuracy · **Runtime:** 1x NVIDIA A10G, <1.5h
**Licence:** CC BY-SA 4.0

## The task

Given a word's phoneme transcription in one accent (RP, GA, AU, NZ), rewrite
it for another. You already have the sounds; the question is the **transfer
function between dialects** — which segments shift and what governs it.

The metric scores **only the segments that change**, so the ~80% of every
string that survives untouched earns nothing. Copying the source scores 0.11;
the CPU rule-table reference scores 0.70. Test words are unseen in training.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Challenge configuration and measured baseline anchors |
| `problem-description.md` | The prompt an agent sees |
| `dataset-description.md` | Source, fetch method, schema, intended use, limitations |
| `rubrics.md` | 9 rubrics — 4 REQUIRED, 4 RECOMMENDED, 1 UNIVERSAL |
| `prepare.py` | Deterministic raw → public/private split (seed 20260918), word-disjoint |
| `grade.py` | `grade(submission, answers) -> float` |
| `solution.ipynb` | Reference solution, 3 progressive submissions (GPU final) |
| `dataset/` | `raw/`, `public/`, `private/`, `generator/` |

## Reproduce

```bash
python prepare.py
python grade.py dataset/public/sample_submission.csv dataset/private/answers.csv   # ≈0.11 floor
```

## Data

English Wiktionary, official dump. Accent tags and IPA are the archive's own
`{{IPA|en|…|a=…}}` template values — no annotation, no generated content.
**CC BY-SA 4.0**, confirmed via the MediaWiki siteinfo API. See
`dataset/raw/ATTRIBUTION.txt`.
