# Project Eris — Solver Guide

How to solve Project Eris benchmark challenges and compete on the leaderboard.

Source: [shipd.ai/quests/eris](https://shipd.ai/quests/eris) · Creator guide: [project-eris.md](project-eris.md)

---

## How Solving Works

1. **Pick a challenge** — each one bundles a problem description, downloadable public data, and an automated grader.
2. **Submit a Jupyter notebook** (`solution.ipynb`) that solves the challenge end-to-end.
3. **Get scored automatically** — `grade.py` compares your predictions against private ground truth.
4. **Compete on the leaderboard** — your placement is determined by your score.

## Important Rules

- **No LLM outputs** may be used as part of your submission.
- Solutions run against the **Kaggle Python Docker image** environment (pandas, numpy, scikit-learn, xgboost, lightgbm, tensorflow, pytorch, etc.) — stay within those libraries.
- Your work is also judged against the challenge's **rubrics** (task-specific ML engineering criteria marked REQUIRED / RECOMMENDED), not just the raw score.

## Challenge Lifecycle (Timing Matters)

- Once **10 distinct solvers** have graded solutions, a **24-hour closing countdown** begins — visible on the challenge page.
- When the countdown ends, the challenge **closes**: no new submissions, leaderboard finalized.
- Don't sit on a near-ready notebook; a popular challenge can close within a day.

---

## Workflow

1. **Read the problem description carefully.** It's written to be unambiguous — the metric, submission format, and constraints are all specified. Missing a detail costs points.
2. **Check the rubrics first.** REQUIRED rubrics tell you exactly what the reviewer/grader expects from a good solution — treat them as a spec, not a suggestion.
3. **Explore the public data.** Creators deliberately plant pitfalls: missing values, class imbalance, leakage traps, distribution shifts. Assume they're there and look for them.
4. **Establish a baseline early.** Get a simple end-to-end notebook producing a valid submission before optimizing anything.
5. **Match the exact submission format.** Column names, file names, ID ordering, dtypes — format errors waste a submission.
6. **Optimize against the stated metric.** Read exactly how the score is computed (especially non-standard formulas) and validate locally with the same computation.
7. **Guard against overfitting.** Your score comes from the **private split** you never see — use proper cross-validation on the public data.
8. **Keep the notebook clean and end-to-end.** It should run top-to-bottom: load data → features → model → predictions → submission file. Rubrics reward sound ML engineering.

## Practical Tips

- **Deterministic runs:** seed everything; a solution that scores differently per run is a liability.
- **Watch edge cases:** graders handle malformed input gracefully but score valid submissions strictly — validate your output shape before submitting.
- **Feature engineering usually beats model complexity** on tabular challenges of this size.
- **Budget submissions:** treat each one as expensive; test locally first, submit when confident.
