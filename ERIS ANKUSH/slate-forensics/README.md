# Slate Forensics: Auditing and Repairing a Broken Product Recommender

An Eris challenge built on the genuine **Online Retail II** dataset (UCI id
502, CC BY 4.0, DOI 10.24432/C5CG6D). A hidden but realistic "You may also
like" recommendation policy was replayed over the real purchase log; a seeded
minority of the emitted slates was corrupted by one of three production
failure modes. Solvers must **audit** each logged request (clean or
corrupted?), **diagnose** the failure mode, and **repair** the slate by
reconstructing what the healthy policy would have served — three coupled,
heterogeneous outputs scored by a composite metric with hard consistency
constraints.

- Domain: **Recommendation** (patterns: recommendation-policy repair,
  counterfactual recommendation audit)
- Solver-facing statement: [problem_description.md](problem_description.md)
- Data provenance and schema: [dataset_description.md](dataset_description.md)
- Evaluation rubrics: [rubrics.md](rubrics.md)

## Quickstart (challenge maintainer)

```bash
python download_data.py     # fetch UCI source + verify license live (network)
python prepare.py           # deterministic build of dataset/public + private
python probe/test_grade.py  # grader unit tests (format, penalties, edge cases)
python probe/probe.py       # naive baselines stay below the 0.45 floor
jupyter nbconvert --to notebook --execute --inplace solution.ipynb
python grade.py working/submission.csv
```

## Score landscape (measured)

| Strategy | Composite |
| --- | --- |
| Random guessing | 0.165 |
| All-corrupted + bestseller repairs | 0.252 |
| Copy emitted slates, declare all clean (= sample submission) | 0.331 |
| Best handcrafted single rule (price heuristic) | 0.370 |
| **Reference solution (`solution.ipynb`, ~1 min)** | **0.900** |
| Perfect oracle | 1.000 |

The reference pipeline: policy-imitation ranker (multi-window co-purchase,
price and popularity features; LightGBM) → slate-level forensic diagnosis
(4-class) → consistency-aware assembly with the threshold tuned against the
official grader on customer-disjoint validation. Remaining headroom is
mostly in repairing corrupted slates (per-bucket RBO ≈ 0.74–0.75), which
requires modelling the policy's set-level structure.

## Repository layout

```text
download_data.py     # genuine-source fetch + live license verification
prepare.py           # deterministic hidden-policy replay + corruption overlay
grade.py             # composite scorer (repair / audit / mode / consistency)
solution.ipynb       # executed reference solution
config.yaml          # challenge metadata
problem_description.md dataset_description.md rubrics.md
DESIGN_CANDIDATES.md SELF_AUDIT.md          # reviewer-facing design records
data/                # raw source + provenance (source_metadata.json)
dataset/public/      # solver files     dataset/private/  # answers.csv
probe/               # test_grade.py, probe.py, probe_results.json
```

Reviewer note: `prepare.py`, `DESIGN_CANDIDATES.md` and `SELF_AUDIT.md`
document the hidden generation process and must not be shared with solvers.
