# Project Eris — Creator Guide

Project Eris is an **ML benchmark creation quest on Shipd** (Datacurve AI). You build evaluation challenges that test how well AI agents approach real machine learning problems.

Think of it as creating Kaggle competitions — except the format is designed so **AI agents (not humans)** read your challenge, attempt a solution, and get scored automatically. Every contribution becomes training data for the next generation of AI models.

Source: [shipd.ai/quests/eris](https://shipd.ai/quests/eris) · Solver guide: [solving-challenges.md](solving-challenges.md)

---

## Important Rules

- **No copying Kaggle competitions** in any form.
- You **may** use publicly available Kaggle **datasets** that allow commercial use.
- **No LLM outputs** may be used as part of your submission.

---

## Deliverable Anatomy

A complete challenge contribution consists of:

| Piece | What it is |
|---|---|
| **Dataset** | Raw data (public or synthetic) + description & metadata |
| **Challenge** | Problem description + `prepare.py` + `grade.py` + config, tied to an accepted dataset |
| **Rubrics** | 5+ initial rubrics defining what good ML engineering looks like for this task |
| **Reference solution** | `solution.ipynb` that solves the challenge end-to-end and proves solvability |

---

## 1. Datasets

A dataset is the raw material for a challenge. Every challenge references an **accepted dataset** and includes a prepare script that transforms raw data into **public and private splits**.

Think about what skills the dataset should test — handling missing values, class imbalance, feature engineering — and what pitfalls might trip up a naive agent.

### Public data
- Document the **source URL** and verify the license **allows commercial use**.
- Ensure it's **legal to redistribute**.
- Prioritize **recent and less well-known** datasets.

### Synthetic data
- Document the generation process and include the **reproducible script**.
- Ensure **realistic distributions and relationships**.
- Introduce **controlled complexity**: noise, missing values, edge cases.

### Files & description
- Raw upload can follow any structure — the key requirement is that **everything is documented in the dataset description**.
- The description must document **all columns and types**.

### Dataset checklist
- [ ] Description documents all columns and types
- [ ] Sufficiently many samples
- [ ] Prepare script is reproducible and deterministic
- [ ] License allows commercial use (if public data)
- [ ] Source URL documented (if public data)

---

## 2. Challenges

A challenge is the competition an AI agent enters: **problem description + grading script + config file**, tied to an accepted dataset. Creating a challenge also means defining the **initial rubrics**.

### Problem description
The problem description is **the prompt the agent sees**. The more precisely you define the task, the more meaningful the evaluation — the agent shouldn't have to deal with ambiguity. An agent must be able to solve it **without asking clarifying questions**.

### `prepare.py`
- Transforms the raw dataset into **public and private splits**.
- Must be **deterministic** — running it twice produces identical outputs.
- Runs during challenge creation to generate the prepared data solvers download.
- Only use libraries from the **Kaggle Python Docker image** (pandas, numpy, scikit-learn, xgboost, lightgbm, tensorflow, pytorch, etc.).
- Must produce the required split files (format differs for tabular vs. image/audio data).

### `grade.py`
- Compares an agent's predictions against **private ground truth** and returns a score.
- Without a solid grader, there's no way to tell if an agent did well or just got lucky.
- Same library constraint: **Kaggle Python Docker image only**.
- **Handle edge cases gracefully** — return a score when possible; raise an exception only for truly invalid submissions.

### Challenge checklist
- [ ] Problem description is complete and unambiguous
- [ ] An agent could solve it without asking clarifying questions
- [ ] Evaluation metric clearly defined (formula included if non-standard)
- [ ] Submission format precisely specified
- [ ] `prepare.py` produces correct public/private splits, deterministically
- [ ] `grade.py` scores valid submissions correctly and handles edge cases gracefully
- [ ] 5+ initial rubrics, majority REQUIRED or RECOMMENDED
- [ ] Rubrics are specific to this task, not generic

---

## 3. Rubrics

Rubrics describe **what good ML engineering looks like for this specific task** — they're evaluated alongside the score.

- Provide **at least 5** initial rubrics per challenge.
- The **majority** should be **REQUIRED** or **RECOMMENDED**.
- Make them **task-specific**, not generic (e.g. "handles the class imbalance in `target` via resampling or class weights", not "writes clean code").

---

## 4. Reference Solution

A solution is a Jupyter notebook (**`solution.ipynb`**) that solves the challenge end-to-end. As the challenge creator, your reference solution **proves solvability** and validates the grading pipeline.

---

## Creator Tiers

Challenge submissions are gated by a trust tier system:

| Tier | Pending Reviews | Weekly Submissions | How to Unlock |
|---|---|---|---|
| New User | 1 | 5 | Default |
| Problem Creator | 3 | 20 | 1 accepted challenge |
| Trusted Creator | Unlimited | Unlimited | 3 accepted challenges |

> **Quality over quantity.** One well-crafted accepted challenge is worth more than many rushed rejections.

---

## Challenge Lifecycle

1. Dataset accepted → challenge published with prepared public data.
2. Solvers submit notebooks; leaderboard updates as they're graded.
3. Once **10 distinct solvers** each have a graded solution, a **24-hour closing countdown** starts (visible to everyone on the challenge page).
4. Challenge **closes** — no new submissions; leaderboard is finalized.

---

## Design Principles

- **Objectivity first.** If `grade.py` can't score it deterministically, the task isn't ready.
- **No leakage.** Private ground truth must never be recoverable from the public split.
- **Determinism everywhere.** Same inputs → same splits, same scores.
- **Calibrated difficulty.** Hard for a naive agent, fair for a strong approach.
- **Adversarial mindset.** Assume solvers will exploit any loophole in the spec or grader.
- **Document everything.** Provenance, columns, metric rationale — reviewers and agents both depend on it.
