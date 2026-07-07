# Unfluke: Skill-vs-Luck Forensics for Systematic Trading Records

An Eris task in the **From Scratch** domain. Solvers receive the
in-sample track records (trade logs + anonymized market bars) of 6,080
systematic trading strategies, of which a hidden subset carries a
genuine persistent edge while a matched cohort merely got lucky in
sample. For each of 2,432 held-out test strategies they must output a
skill probability, a predicted out-of-sample Sharpe ratio, and an
exact-240 portfolio selection, scored by a composite metric.

- Solver-facing statement: [`problem_description.md`](problem_description.md)
- Data provenance, license, transformations: [`dataset_description.md`](dataset_description.md)
- Grading criteria for solutions: [`rubrics.md`](rubrics.md)
- Design screening (15 candidates): [`candidate_ideas.md`](candidate_ideas.md)
- Compliance review: [`self_audit.md`](self_audit.md)

## Why this task

Backtest overfitting — telling skill from luck in a pile of strategy
track records — is a real, hard evaluation problem with no public
leaderboard equivalent. The design makes headline statistics
deliberately uninformative (lucky strategies are retained only when
their in-sample Sharpe and win rate match the skilled cohort), so
solvers must learn conditional, temporal signatures from the raw trade
sequences. Train and test strategies live on disjoint anonymized
markets, and all hidden targets depend on undisclosed strategy
mechanics, so external data cannot reconstruct the answers.

## Build & validate

```bash
python download_data.py       # fetch public source archive (checksum-verified), build dataset/raw/prices.csv
python prepare.py             # deterministic: dataset/public/* + dataset/private/answers.csv
python grade.py dataset/public/sample_submission.csv   # ~0.03
python probe/probe.py         # naive baseline submission + score
python probe/test_grade.py    # grader behavior tests
jupyter nbconvert --to notebook --execute solution.ipynb  # writes ./working/submission.csv
python grade.py working/submission.csv                    # reference score
```

`prepare.py` is deterministic (base seed 20260707; reruns are
byte-identical) and also supports the platform entrypoint
`prepare(dataset_dir, public_dir, private_dir)` with robust raw-file
discovery and a copy-through fallback when prepared files already
exist.

## Scoring summary

`score = clip(0.40*D + 0.25*R + 0.35*P - penalty, 0, 1)`, higher is
better: D = family-stratified skill-detection AUC rescaled to [0, 1];
R = mean within-arena Spearman correlation of predicted vs realized
out-of-sample Sharpe; P = portfolio precision over the skilled base
rate; penalty = 0.001 per logically inconsistent row (cap 0.10). Exact
formulas in `problem_description.md` and `grade.py`.

Validated anchor scores: sample submission ≈ 0.03, naive
in-sample-performance baseline (probe) ≈ 0.38, reference notebook
≈ 0.76 (perfect submission ≈ 0.996).

## Repository layout

```text
download_data.py   # credential-free public download + provenance (data/source_metadata.json)
prepare.py         # deterministic transformation: raw prices -> arenas -> strategy population
grade.py           # validation + composite scoring (JSON report)
solution.ipynb     # reference solution (public files only -> ./working/submission.csv)
probe/probe.py     # naive baseline; probe/test_grade.py: grader tests
data/              # raw archive + extraction + checksums + machine-readable provenance
dataset/raw        # canonical raw table + license notice
dataset/public     # solver-visible files
dataset/private    # hidden answer key
```

Note: `data/raw/source_daily_ohlcv.zip` and `data/raw/historical_data/`
are large download artifacts recreated by `download_data.py`; the upload
archive omits them (checksums are recorded in `data/raw/checksums.sha256`
and `data/source_metadata.json`).
