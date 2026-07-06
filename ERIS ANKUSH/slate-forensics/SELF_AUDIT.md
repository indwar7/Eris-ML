# Self-Audit — Slate Forensics (reviewer-facing; do not ship to solvers)

## Selection

- **Open domain**: Recommendation — spec patterns *recommendation-policy
  repair* and *counterfactual recommendation audit* (with structured
  multi-output reasoning).
- **Genuine dataset**: Online Retail II, UCI Machine Learning Repository
  (dataset id 502) — 1,067,371 real transactions of a UK online giftware
  retailer, 2009-12-01 → 2011-12-09.
- **Source URL**: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **License URL**: https://creativecommons.org/licenses/by/4.0/
- **DOI / citation**: 10.24432/C5CG6D — *Chen, D. (2012). Online Retail II
  [Dataset]. UCI Machine Learning Repository.*
- **Why redistribution/use is permitted**: CC BY 4.0 explicitly allows
  copying, adaptation and redistribution for any purpose, including
  commercial, with attribution. Attribution is given in
  `dataset_description.md`, `data/raw/source_license.txt` and
  `data/source_metadata.json`. The license string and DOI were confirmed
  **live from the UCI source at download time** (2026-07-05) by
  `download_data.py`, which aborts rather than writing unverified provenance;
  the verbatim API response is preserved as evidence
  (`data/raw/uci_api_metadata.json`).

## Transformations applied (deterministic, seed 20260705)

1. A hidden healthy recommendation policy replayed at six anchor dates over
   each eligible customer's real history: recency-weighted co-purchase
   cosine affinity over a fresh reference window, price-band kernel around
   the customer's median paid price, popularity tilt, repeat-purchase
   exclusion, and a greedy MMR-style diversity re-rank (λ=0.9) that makes the
   slate a set-level object rather than a pointwise top-k.
2. Corruption of a seeded ~35% of requests by exactly one realistic failure
   mode: `popularity_fallback` (personalisation loss, exclusion bypassed),
   `price_band_shift` (price anchor multiplied by a per-request factor from
   {1.9, 2.6, 3.3}), `stale_index` (statistics read from a window two
   quarters old). Corruptions indistinguishable from healthy output are
   relabeled clean, so every positive label is detectable in principle.
3. Packaging with anti-leakage measures (below).

**Why realistic**: each mode reproduces a documented class of production
recommender incident (fallback-on-error serving, misconfigured upsell
re-ranking, stale feature/index reads), and the healthy policy mirrors how
lightweight production co-purchase recommenders with diversity re-ranking
actually work. The overlay preserves full contact with the genuine data:
every behavioural statistic a solver models is computed from real purchases.

## Why this is not a standard benchmark

- Not classification/regression: the primary deliverable is a **ranked
  10-item reconstruction over a ~4,700-item catalog** for every request,
  coupled to an audit flag and a diagnosis label with hard consistency
  constraints. Repair carries 50% of the composite.
- Not standard recommendation ranking: ground truth is **the hidden policy's
  output**, not user relevance — it is inverse/imitation learning from logged
  slates, plus anomaly forensics. No public leaderboard task matches
  "logged-slate corruption audit + policy repair" on this data (checked
  against known Online Retail II uses: RFM/CLV, market-basket, churn).
- Not solvable by feature engineering or GBM alone: measured floors — copy
  emitted 0.331, best handcrafted rule 0.370, bestseller repairs 0.252,
  random 0.165 — versus 0.900 for the full learned pipeline. The repair head
  requires candidate generation + learned ranking; a GBM without a
  policy-imitation architecture has no way to emit calibrated slates.
- Why modeling is necessary: the policy's parameters (windows, decay, kernel,
  diversity strength) are hidden; only ~5.2k labeled examples define it.
  Reconstruction demands representation of co-purchase structure and
  learning-to-rank; audit/diagnosis demand learned separation of three
  signature families, one of which varies its strength per request.

## Novelty (target ≥ 4/5)

Inverse-recommendation forensics — imitate a production policy from its logs
to detect, name and undo serving corruption — is not a Kaggle archetype, not
a benchmark-clone, and composes three heterogeneous targets with logical
coupling. The substrate dataset is well known but its standard uses (basket
analysis, CLV) share nothing with this task formulation. Recorded ideation of
15+ alternatives with elimination rationale: `DESIGN_CANDIDATES.md`.

## Difficulty calibration (measured on the private test set)

| Strategy | Composite |
| --- | --- |
| random | 0.165 |
| all-corrupted + bestsellers | 0.252 |
| copy-emitted (sample submission) | 0.331 |
| best single handcrafted rule | 0.370 |
| reference solution (full pipeline) | 0.900 |
| oracle | 1.000 |

Gradient within the reference: audit/diagnosis of `popularity_fallback` is
deliberately easy (real fallbacks are obvious to real auditors), the variable
`price_band_shift` is intermediate, `stale_index` is hard; repairing
corrupted slates (bucket RBO ≈ 0.74–0.75) is the open headroom and requires
set-aware (diversity-respecting) slate construction that pointwise rankers
cannot fully capture.

## Hidden structure the model must learn

The healthy policy itself (candidate universe, affinity computation, price
matching, diversity re-rank), the corruption assignment (which requests
deviate from that policy), and the per-mode failure signatures.

**What breaks naive baselines**: copying emitted slates zeroes audit recall
and earns repair credit only in the clean bucket (bucket-balancing caps it);
popularity lists earn near-zero repair everywhere; single-feature rules
cannot cover three heterogeneous signatures at once; ignoring consistency
coupling costs up to 30% multiplicatively.

## Leakage checks performed (all automated, all passing)

1. Train/test customers disjoint; train/test slate ids disjoint.
2. Public transaction log truncated strictly before the evaluation anchor →
   no post-anchor information exists for any test request.
3. Slate-id order vs hidden labels: id permutation is rejection-sampled at
   build time until |corr(id, flag)| < 0.02 (train) and < 0.04 (test);
   final measured values −0.005 / −0.012, per-mode |corr| ≤ 0.018.
4. No label columns in any public test file; sample submission carries only
   defaults; clean slates satisfy emitted == healthy exactly and corrupted
   slates differ (verified over all 6,001 requests).
5. Every healthy/emitted code exists in the public catalog (no private-only
   vocabulary); per-slate structure validated (10 unique positions, 10
   unique items).
6. prepare.py reruns are byte-identical (sha256-verified), so no
   timestamp/order artifacts encode labels.
7. probe/probe.py confirms no naive strategy exceeds 0.45.

## Licensing / provenance checks performed

- License string ("CC BY 4.0") and DOI (10.24432/C5CG6D) extracted from the
  live UCI page/API at download time; script fails hard if unconfirmed.
- Citation text copied verbatim from the UCI citation block (APA, year 2012).
- SHA-256 checksums of all raw downloads recorded in
  `data/raw/checksums.txt` and `data/source_metadata.json`.
- No credentials, payment or manual approval involved; single static HTTPS
  URL on the first-party archive.
- No LLM-generated content anywhere in data or solution (guide core rule).

## Deviations / notes for the maintainer

- The task prompt referenced both `Eris/New-Projects/<name>` and
  `Eris/new/<name>`; per the operator's explicit instruction the package is
  saved under `ERIS ANKUSH/slate-forensics`.
- `answers.csv` includes the emitted slate alongside the healthy slate so the
  grader's consistency checks work when the platform passes only
  (submission, answers) DataFrames.
- The reference notebook tunes its decision threshold against `grade.py` on
  held-out customers — metric-aware but using public information only.
