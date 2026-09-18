# The Order of Discovery

**Domain:** natural-language-processing · **Tags:** text
**Metric:** mean per-paper position accuracy (chance ≈0.22) · **Runtime:** 1x NVIDIA A10G, <1.5h
**Licence:** CC BY 4.0

## The task

A Results section is a chain of experiments with a direction: each subsection
answers a question the previous one raised. We shuffle a paper's Results
subsections and ask you to put the experiments back in the order the authors
ran them.

Figure numbers and sequencing words (*first*, *next*, *as shown above*) are
masked. Every item shares register, authorship and statistical density — there
is no stylistic tell. The only signal is **scientific presupposition**: which
claim depends on which.

**Chance is 1/n per paper (≈0.22 over the test split).** Papers are disjoint
across splits.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Challenge configuration and measured baseline anchors |
| `problem-description.md` | The prompt an agent sees |
| `dataset-description.md` | Source, fetch method, schema, intended use, limitations |
| `rubrics.md` | 10 rubrics — 5 REQUIRED, 4 RECOMMENDED, 1 UNIVERSAL |
| `prepare.py` | Deterministic raw → public/private split (seed 20260913) |
| `grade.py` | `grade(submission, answers) -> float` |
| `solution.ipynb` | Reference solution, 3 progressive submissions |
| `dataset/` | `raw/`, `public/`, `private/`, `generator/` |

## Reproduce

```bash
python prepare.py                                        # raw -> public/private
python grade.py dataset/public/sample_submission.csv \
                dataset/private/answers.csv              # -> ~0.22 (chance)
```

## Data

Real PLOS research articles. The label is the authors' own ordering of their
Results subsections, as published — no annotation created for this dataset,
no model-generated content.

**CC BY 4.0** — every PLOS research article carries the Creative Commons
Attribution licence in its own `<license>` block. Commercial use permitted with
attribution. See `dataset/raw/ATTRIBUTION.txt`.
