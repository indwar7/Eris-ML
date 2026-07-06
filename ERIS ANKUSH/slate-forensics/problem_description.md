# Slate Forensics: Auditing and Repairing a Broken Product Recommender

## Overview

A UK-based online giftware retailer ran a "You may also like" module that
served each active customer a personalised slate of 10 product
recommendations. After a maintenance window, the on-call team discovered that
a stale configuration flag had been silently routing a fraction of requests
through **broken variants of the recommendation service**. The emitted slates
were logged; the flag that recorded which code path served each request was
not.

You are the forensic auditor. Working from the real transaction log and the
logged slates, you must — for every request in the evaluation set —

1. **Audit**: decide whether the emitted slate came from the healthy service
   or a broken variant (`corrupted` ∈ {0, 1});
2. **Diagnose**: name the failure mode that produced it;
3. **Repair**: reconstruct the slate the *healthy* service would have emitted
   for that customer at that moment (for a clean request, that is exactly the
   emitted slate).

This is a reconstruction-and-audit problem, not a next-purchase prediction
problem: the ground truth is the behaviour of the hidden recommendation
policy, which you must learn to imitate from the labeled examples.

## Incident report: the three failure modes

The engineering post-mortem identified three broken code paths. Their exact
parameters are unknown — that is what your models must infer — but their
symptoms were described as follows:

- `popularity_fallback` — personalisation dies entirely and the service
  falls back to generic bestsellers, ignoring everything known about the
  customer (including what they just bought).
- `price_band_shift` — a misconfigured upsell experiment skews
  recommendations toward items priced well above the customer's usual
  spending band. The strength of the skew varied between requests.
- `stale_index` — the service reads from a stale index and serves
  recommendations computed from months-old purchase statistics, out of step
  with the current season.

Exactly one failure mode applies to each corrupted request. A substantial
minority of the evaluation requests is corrupted; most are clean.

## Dataset

Public files (train only on these; write predictions to
`./working/submission.csv`):

```text
./dataset/public/transactions.csv         real purchase log (see below)
./dataset/public/catalog.csv              per-item reference statistics
./dataset/public/slates_train.csv         emitted slates, labeled requests
./dataset/public/slate_labels_train.csv   corrupted flag + failure mode
./dataset/public/healthy_slates_train.csv healthy slate for every train request
./dataset/public/slates_test.csv          emitted slates to audit (unlabeled)
./dataset/public/sample_submission.csv    required output format
```

`transactions.csv` is the genuine Online Retail II purchase log (UCI Machine
Learning Repository, CC BY 4.0) — invoices, stock codes, descriptions,
quantities, timestamps, unit prices, customer ids and countries. The export
ends at the evaluation anchor date, so it contains everything the healthy
service could have known and nothing it could not.

| Column | Type | File | Description |
| --- | --- | --- | --- |
| slate_id | str | slates_* | Opaque request identifier |
| customer_id | str | slates_* | Joins to transactions.csv |
| anchor_date | date | slates_* | When the request was served |
| position | int | slates_*, healthy_* | Rank 1..10 within the slate |
| stock_code | str | slates_*, healthy_* | Recommended item |
| corrupted | int | slate_labels_train | 1 if a broken path served it |
| mode | str | slate_labels_train | Failure mode, `none` if clean |

Important notes:

- Train requests were served at six anchor dates; every evaluation request
  was served at the final anchor. Train and evaluation customers are
  disjoint — you must learn the *policy*, not the customers.
- A request could only have used information available strictly before its
  `anchor_date`. Respect that boundary when engineering features for train
  requests; the evaluation anchor has no post-anchor data at all.
- The transaction log is real and messy: cancelled invoices start with `C`,
  ~23% of rows have no `customer_id`, service codes (postage, manuals) do not
  look like product codes, and demand is strongly seasonal.
- `healthy_slates_train.csv` gives the healthy slate for **every** train
  request — for clean requests it equals the emitted slate; for corrupted
  requests it is what should have been served. This is your supervision for
  learning the healthy policy.

## Evaluation

Submissions are scored with a composite forensic score. **Higher is better**;
range [0, 1]. With `B = {none, popularity_fallback, price_band_shift,
stale_index}` the true condition of each request:

- **S_repair** — for each request, truncated rank-biased overlap between your
  slate and the hidden healthy slate,
  `RBO@10(pred, truth) = Σ_{d=1..10} 0.9^(d-1) · |pred[:d] ∩ truth[:d]| / d`,
  normalised by `Σ_d 0.9^(d-1)`; averaged within each bucket of `B`, then
  averaged over the four buckets with equal weight. Farming the majority
  clean bucket cannot carry this component.
- **S_audit** — `0.5 · mean_m(recall of corrupted=1 over each true failure
  mode m)` + `0.5 · (specificity on truly clean requests)`.
- **S_mode** — macro recall of your `mode` over the three corrupted
  conditions (predicting `none` on a corrupted request scores zero for it).
- **S_consistency** — fraction of requests with none of these violations:
  flagged clean but slate ≠ emitted; flagged corrupted but slate = emitted;
  flagged clean but mode ≠ `none`; flagged corrupted but mode = `none`.

```text
final = (0.50·S_repair + 0.30·S_audit + 0.20·S_mode) · (0.7 + 0.3·S_consistency)
```

The exact implementation ships with the challenge in `grade.py` — read it;
it is the metric.

## Submission format

One CSV, `./working/submission.csv`, with exactly 10 rows per evaluation
`slate_id` (7,570 rows total) and a header:

| Column | Type | Description |
| --- | --- | --- |
| slate_id | str | From `slates_test.csv` |
| corrupted | int | 0 or 1, identical on all 10 rows of the slate |
| mode | str | `none`, `popularity_fallback`, `price_band_shift`, or `stale_index`; identical on all 10 rows |
| position | int | 1..10, each exactly once per slate |
| stock_code | str | Your reconstruction of the healthy slate, no duplicates within a slate |

Malformed submissions (wrong ids, missing positions, duplicate items,
non-constant flags, unknown modes) are rejected, not scored. Fabricated stock
codes earn nothing and displace items that could have matched.

## Why this matters

Logged-but-unlabeled recommender incidents are a real operational problem:
serving bugs silently degrade revenue and trust, and post-hoc audits must
reconstruct what a healthy system *would have done* from behavioural data
alone. The skills tested here — imitating a production policy from its logs,
characterising failure signatures, and coupling several outputs into one
consistent verdict — transfer directly to recommender-system reliability
work.

## Rules

- Train only on files under `./dataset/public/`; write only under
  `./working/`.
- No external data, no pretrained models fetched at runtime, no LLM-generated
  content.
- Standard Kaggle Python Docker image libraries; the full pipeline must
  finish within 30 minutes.
