# Project Eris

## What is Eris?

Project Eris is an ML benchmark creation quest on Shipd. You build evaluation challenges that test how well AI agents approach real machine learning problems — then you get paid when your work passes review.

Think of it like creating Kaggle competitions, except the format is designed so AI agents (not humans) can read your challenge, attempt a solution, and get scored automatically. Every contribution becomes training data for the next generation of AI models.

### Two ways to participate

**I want to create challenges**
Build a dataset, define a challenge with a grading script and rubrics, and write a reference solution that proves it works.
- Datasets
- Challenges
- Rubrics

**I want to solve challenges**
Pick a challenge, submit Jupyter notebooks that compete on the leaderboard, and earn payouts based on your score.
- Solutions
- Rubrics

> **Important rules:** You cannot copy competitions from Kaggle in any form. You can use publicly available Kaggle datasets that allow commercial use. You may not use any LLM outputs as part of your submission.

---

## Datasets

A dataset is the raw material for a challenge. Every challenge references an accepted dataset and includes a prepare script that transforms the raw data into public and private splits.

### Data Sources

You can use existing public data or create synthetic data. Think about what skills the dataset should test — handling missing values, class imbalance, feature engineering — and what pitfalls might trip up a naive agent.

**Public Data**
- Document the source URL and verify the license allows commercial use
- Ensure it's legal to redistribute
- Prioritize recent and less well-known datasets

**Synthetic Data**
- Document the generation process and include the reproducible script
- Ensure realistic distributions and relationships
- Introduce controlled complexity (noise, missing values, edge cases)

### Files

Your raw dataset upload can follow any structure. For example, for tabular data:

```
data.csv                 # Your source data (or multiple files)
```

The exact file names and formats depend on your data. The key requirement is that everything is documented in the dataset description.

### Description & Metadata

Create a dataset description with these required sections:

```markdown
# Dataset Description

## Overview
Where does this data come from? What does it represent?

## File Structure
- `data.csv` — labeled data
[Any other files in the raw dataset]

## Features
| Column | Type | Description |
|--------|------|-------------|
| id     | int  | Unique identifier |
| ...    | ...  | ... |

## Notes
[Domain-specific context the agent needs]
```

### Datasets Checklist

- [ ] Dataset description documents all columns and types
- [ ] Dataset has sufficiently many samples
- [ ] Prepare script is reproducible and deterministic
- [ ] License allows commercial use (if public data)
- [ ] Source URL is documented (if public data)

---

## Challenges

A challenge is the competition an AI agent enters. It bundles a problem description, a grading script, and a config file, all tied to an accepted dataset. When you create a challenge, you also define initial rubrics that describe what good ML engineering looks like for this specific task.

### Problem Description

The problem description is the prompt the agent sees. The more precisely you define the task, the more meaningful the evaluation becomes as the agent doesn't have to deal with ambiguity.

```markdown
# [Challenge Title]

## Overview
[1-2 paragraphs: What is being predicted? What data is available?
What's the real-world context?]

## Evaluation
Submissions are scored using **[metric name]** (e.g., AUC-ROC, RMSE).

```python
def evaluate(y_true, y_pred):
    # Your metric calculation (if non-standard)
    return score
```

## Dataset
[Describe the prepared dataset structure — what files are in
public/, column descriptions, data types]

## Submission
Submit a CSV file with the following format:

| Column     | Type  | Description                    |
|------------|-------|--------------------------------|
| id         | int   | Row identifier from test.csv   |
| prediction | float | Predicted probability (0-1)    |

**Requirements:**
- Must contain exactly [N] rows (one per test sample)
- Include header row
```

### Prepare Script

`prepare.py` transforms the raw dataset into public and private splits. It must be deterministic — running it twice produces identical outputs. This script runs during challenge creation to generate the prepared data for solvers to download.

```python
from pathlib import Path

def prepare(raw: Path, public: Path, private: Path) -> None:
    # Read raw data
    # Perform train/test split (set random seeds!)
    # Write public/ (train.csv, test.csv, sample_submission.csv)
    # Write private/ (answers.csv)
```

The prepare script should only use libraries defined in the Kaggle Python Docker image (pandas, numpy, scikit-learn, xgboost, lightgbm, tensorflow, pytorch, etc.).

#### Required Files

The prepare script must produce these files.

**Tabular Data**

```
public/
├── train.csv              # Labeled training data
├── test.csv               # Unlabeled test data
└── sample_submission.csv  # Example of expected format

private/
└── answers.csv            # Test data with labels
```

**Image / Audio Data**

*(layout not provided in source docs)*

### Grading Script

`grade.py` compares an agent's predictions against private ground truth and returns a score. Without a solid grader, there's no way to tell if an agent did well or just got lucky.

```python
import pandas as pd
import numpy as np

def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Score a submission against ground truth answers.

    Args:
        submission: The agent's predictions (loaded from submission.csv)
        answers: Ground truth labels (loaded from private/answers.csv)

    Returns:
        A float score. Direction (minimize/maximize) is set in config.yaml.

    Raises:
        Exception if the submission format is invalid.
    """
    # Validate submission format
    # Calculate and return score
    return 1.0
```

The grading script should only use libraries defined in the Kaggle Python Docker image (pandas, numpy, scikit-learn, xgboost, lightgbm, tensorflow, pytorch, etc.). Handle edge cases gracefully — try to return a score when possible, and raise an exception only for truly invalid submissions.

### Creator Tiers

Challenge submissions are gated by a trust tier system. New users start with conservative limits that increase as their challenges get accepted.

| Tier | Pending Reviews | Weekly Submissions | How to Unlock |
|------|-----------------|--------------------|---------------|
| New User | 1 | 5 | Default |
| Problem Creator | 3 | 20 | 1 accepted challenge |
| Trusted Creator | Unlimited | Unlimited | 3 accepted challenges |

> Focus on quality over quantity. One well-crafted challenge that gets accepted is worth more than many rushed submissions that get rejected.

### Closing & Payouts

Challenges close automatically to ensure timely payouts:

1. **10 distinct solvers** — once 10 different users each have a graded solution, a 24-hour closing countdown begins automatically.
2. **Countdown visible to all** — every solver can see the remaining time on the challenge page.
3. **Challenge closes** — no new submissions are accepted. The leaderboard is finalized.
4. **Payouts processed** — payouts are calculated for solvers, the challenge creator, and rubric contributors.

### Challenge Checklist

- [ ] Problem description is complete and unambiguous
- [ ] An agent could solve this without asking clarifying questions
- [ ] Evaluation metric is clearly defined (with formula if non-standard)
- [ ] Submission format is precisely specified
- [ ] `prepare.py` produces correct public/private splits and is deterministic
- [ ] `grade.py` correctly scores valid submissions and handles edge cases gracefully
- [ ] 5+ initial rubrics, majority REQUIRED or RECOMMENDED
- [ ] Rubrics are specific to this task, not generic

---

## Solutions

A solution is a Jupyter notebook (`solution.ipynb`) that solves a challenge end-to-end. As a challenge creator, your solution proves solvability and validates the grader. As a solver, you compete on the leaderboard for a share of the $1,150–$1,250 prize pool. Solutions run in a Docker environment with Kaggle Docker libraries (pandas, numpy, scikit-learn, xgboost, lightgbm, tensorflow, pytorch, etc.).

### Requirements

- **Self-contained** — must run end-to-end without manual intervention
- **Reads from `./dataset/public/`** — use relative paths
- **Writes only to `./working/`** — outputs `submission.csv`
- **Uses only standard Kaggle Docker libraries**
- **Completes in under 30 minutes** (64GB RAM, NVIDIA A10G, 24GB VRAM)

### Iterative Submissions

We want to see your progression — not just a single final answer. Submit multiple solutions as you improve your approach. Each submission should be a solid attempt, not a rough draft.

1. **First Submission — Solid Starting Point**
   Data pipeline is correct, model runs end-to-end, and you're getting reasonable results.
2. **Middle Submissions — Incremental Improvements**
   Try feature engineering, hyperparameter tuning, or different model architectures. Each submission should show measurable progress from the previous one.
3. **Final Submission — Best Effort**
   Your most refined submission with the most optimal model.

> We recommend 3–5 solutions showing incremental improvement. Each should be a complete, working solution — not rough drafts, but genuine attempts at solving the challenge.

### Submission Credits

Each challenge uses a credit system that limits how often you can submit solutions.

| Parameter | Value |
|-----------|-------|
| Max credits per problem | 6 |
| Refill rate | 1 every 4 hours |
| Full refill time | 24 hours |
| Scope | Per problem (independent pools) |

> Credits are consumed only on submit — creating drafts, running checks, or editing does not cost credits. Make each submission count.

### Prize Structure

Each challenge has a $1,150–$1,250 prize pool split between the challenge creator, solvers, and rubric authors.

| Pool | Amount | Who receives it |
|------|--------|-----------------|
| Creator | $400–$500 | Challenge author (scales with novelty) |
| Leaderboard | $500 | 1st: $250 · 2nd: $150 · 3rd: $100 |
| Solution | $150 | Split by merit among solvers who beat the AI baseline |
| Rubric | $100 | Best accepted rubric submission |

> You must beat the AI baseline to earn a payout. When a challenge is created, an AI agent attempts it to establish a baseline score. Only solutions that outperform this baseline are eligible for solver payouts.

### Payout Lifecycle

After a challenge closes, payouts move through three statuses before reaching your Stripe account.

- **Pending** — Payout computed and recorded, but not yet sent to Stripe.
- **Processing** — Stripe transfer created and in transit. Typically 1–2 business days.
- **Paid** — Transfer complete. Funds are in your Stripe account.

> Stripe account required. Payouts remain Pending until you have a connected Stripe account. Your earnings are safe — they'll be processed once connected. Go to shipd.ai/payouts to set up your account.

### Solution Checklist

- [ ] Solution script runs end-to-end without errors
- [ ] Uses only Kaggle Docker environment libraries
- [ ] Reads from `./dataset/public/`, writes to `./working/submission.csv`
- [ ] Achieves a reasonable score (validates the task is solvable)
- [ ] Comments explain your reasoning at each step
- [ ] Clear progression from baseline to optimized (if submitting multiple)

---

## Rubrics

Rubrics are the core of what makes a challenge valuable. A final score tells you whether an agent got the right answer, but nothing about how it got there. Did it handle missing values intelligently, or just drop every row? Did it validate on a proper holdout, or leak test data? Rubrics capture this — they define what "good ML engineering" looks like for a specific challenge.

For each rubric item, you will define the criteria, or condition that must be met by the agent. You will also categorize the rubric by type and importance, and provide a rationale explaining why this criterion is important for good performance on the challenge.

### Rubric Types

| Type | What It Covers |
|------|----------------|
| DATA_HANDLING | Data loading, cleaning, missing values, outliers, preprocessing |
| MODELING | Model selection, architecture choices, hyperparameters |
| FEATURE_ENGINEERING | Deriving features, selection, reduction |
| TRAINING | Training process, validation strategy, avoiding leakage |
| CODE_QUALITY | Code correctness, efficiency, no crashes |
| AGENT_BEHAVIOR | Exploration patterns, iteration, debugging approach |
| COMMUNICATION | Explanations, reasoning, documentation in agent output |

### Importance Levels

| Level | When to Use | Example |
|-------|-------------|---------|
| REQUIRED | Failure likely means invalid/broken solution | "Must resize and normalize pixel values before training" |
| RECOMMENDED | Recommended for solving optimally | "Use gradient boosting as it's effective for tabular data" |
| UNIVERSAL | Universally applicable optimizations | "Features should be scaled and normalized" |

### Writing Good Rubrics

| Property | What It Means | Example |
|----------|---------------|---------|
| Quantity | Aim for 5+ rubrics. Focus on quality over quantity. | — |
| Specificity | Ground rubrics in YOUR dataset and task, not generic ML advice. | Good: "Handles the 23% class imbalance using oversampling, undersampling, or class weights" <br> Bad: "Handles class imbalance appropriately" |
| Balance | Include both SHOULD (positive) and SHOULD NOT (negative) criteria. | SHOULD: "Validates on a holdout set" <br> SHOULD NOT: "Must not train on test data" |
| Approach-neutral | Don't lock rubrics to a specific technique. Any valid approach should satisfy the criterion. | Good: "Achieves AUC > 0.75 on the test set" <br> Bad: "Uses XGBoost with max_depth=6" |
| Discrimination | Rubrics should separate good attempts from poor ones — if every submission passes, the rubric isn't useful. | Good: "Uses cross-validation with k≥5" <br> Bad: "Attempts to solve the problem" |

### Anti-Patterns

| Anti-Pattern | Example | Why It Fails |
|--------------|---------|--------------|
| Too vague | "Explores data well" | Can't be objectively verified — what counts as "well"? |
| Tool-specific | "Uses pandas profiling library" | Penalizes valid approaches that use different tools |
| Impossible to fail | "Attempts to solve the problem" | Every submission passes, so the rubric adds no signal |
| Solution-specific | "Uses the same feature engineering as the reference" | Only your exact approach passes — other valid methods are penalized |
| Mostly UNIVERSAL | 8 out of 10 rubrics are generic best practices | Doesn't distinguish this task from any other — reviewers will reject |

### Rubric Submissions

Challenge creators define the initial rubric when creating a challenge. After a challenge is accepted, the community can propose improvements through rubric submissions. An accepted rubric submission gets merged into the challenge's canonical rubric and earns a share of the prize pool.

**Who Can Submit**
- **Challenge creator** — you defined the initial rubric and can refine it
- **Solvers** — you've submitted a solution and have first-hand insight into what matters
- **Reviewers** — platform reviewers can also contribute rubric improvements

**Submission Flow**

1. **Create a Rubric Draft** — Click "Create Rubric" on the challenge page. This forks the current canonical rubric into a new draft you can edit. Your draft starts with all existing criteria so you can add, modify, or remove items.
2. **Edit Criteria** — Add new criteria, update existing ones, or remove items that aren't useful. Changes auto-save as you type. You can also add notes explaining your overall rationale for the changes.
3. **Run Validation Checks** — Click "Run Checks" to validate your rubric. All checks must pass before you can submit.
4. **Submit for Review** — Once checks pass, click "Submit for Review". Your current version becomes locked (no more edits) and the submission enters the review queue. A reviewer will evaluate your changes and approve, request revisions, or reject.

> If a reviewer requests changes, your submission moves back to an editable state. You can create a new version based on any previous version, make the requested changes, and resubmit.

**What Happens on Approval**

When your rubric submission is approved, your changes are merged into the canonical rubric. The system computes a diff between the canonical rubric at the time you started your draft and your submitted changes, then applies added, modified, and removed items to the current canonical. This means multiple submissions can be worked on and approved concurrently without conflicts.

---

## Reference

### Status Lifecycle

Every submission — datasets, challenges, solutions, and rubrics — goes through the same lifecycle:

- **Draft** — Work in progress. You can edit freely. Not visible to reviewers.
- **Pending** — Submitted for review. Automated checks run first, then a human reviewer evaluates.
- **Changes** — Reviewer requested changes. Review the feedback, make updates, and resubmit.
- **Accepted** — Approved and finalized. Your contribution is live and you earn your reward.
- **Rejected** — Does not meet quality standards. Review feedback for details.

### Exemplary Submissions

Explore complete, high-quality examples across different domains. Each includes a dataset, challenge, rubrics, and a reference solution — annotated to explain what makes them effective.

| Domain | Difficulty | Title | Description | Tags |
|--------|------------|-------|-------------|------|
| Tabular | Medium | Notebook Upvote Prediction | Predict Kaggle notebook upvote counts from metadata including author stats, code metrics, and text features. A tabular regression problem with diverse feature types. | regression, tabular, feature-engineering, text |
| Computer Vision | Medium | Aneurysm Volume Prediction | Predict intracranial aneurysm volume from 3D MRI patches using segmentation. A medical imaging challenge with only 49 training cases. | segmentation, 3d-medical, regression, small-data |
| Time Series | Medium | Intraday Liquidity Forecasting | Forecast required liquidity buffers at 5-minute intervals using market microstructure, regulatory metrics, and cross-asset signals. A time-series regression challenge with regime shifts. | regression, time-series, finance, forecasting |

---

> **Questions?** Ask in Discord before starting — it's easier to clarify upfront than to redo work.
