# Self-Audit — Unfluke: Skill-vs-Luck Forensics for Systematic Trading Records

## Selected open domain

**From Scratch.** Solvers must build the entire pipeline from raw CSVs:
reconstruct strategy behavior from trade logs and market bars, learn
sequence-derived representations, train supervised models under
arena-grouped validation, calibrate probabilities, and make a
constrained portfolio decision. No pretrained model, foundation model,
or external corpus is applicable or allowed.

## Genuine dataset

- **Dataset:** Binance cryptocurrencies historical daily data
  (uploader: Serhii Kanyhin) — real daily OHLCV market history for 289
  cryptocurrency trading pairs, 2021-06-01 to 2023-06-30.
- **Source URL:** https://zenodo.org/records/8187872
- **DOI:** 10.5281/zenodo.8187872
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **License URL:** https://creativecommons.org/licenses/by/4.0/legalcode
- **Citation:** Serhii Kanyhin (2023). *Binance cryptocurrencies
  historical daily data* [Data set]. Zenodo.
  https://doi.org/10.5281/zenodo.8187872
- **Access:** credential-free direct download, verified live on
  2026-07-07 (archive SHA-256
  `ce9a73eb817d0d9c111f296f7dfdea1694d3bc8149d36375e6da25ce36fbed1f`).

## Why redistribution and use are permitted

The Zenodo record carries an explicit CC BY 4.0 license, which permits
commercial use, modification, and redistribution with attribution.
Attribution is preserved in `dataset_description.md`,
`dataset/raw/source_license.txt`, and `data/source_metadata.json`.
Residual risk considered: the uploader collected the data from a
cryptocurrency exchange's public market feed. Exchange OHLCV bars are
factual market observations (prices and volumes), the classic
fact-not-expression case, and the specific compilation redistributed
here is the uploader's own CC-BY-licensed deposit on a reputable
research archive with a DOI. This was judged materially safer than
sources with no license at all (e.g. exchange bulk endpoints) and is
disclosed rather than hidden. Alternatives with unclear or
non-commercial licenses were rejected during screening
(see `candidate_ideas.md`).

## Transformations applied (deterministic, seed 20260707)

1. 160 assets with ≥600 contiguous clean daily bars → one 590-day
   window each ("arena") at a seeded offset; steps 0–449 public,
   450–589 hidden.
2. Anonymization: per-window price/volume normalization, per-day
   multiplicative jitter (±0.2% prices, ±2% volume), asset identities,
   dates, and offsets dropped, arena ids order-shuffled. All
   simulation runs on the jittered series, so published numbers are
   internally consistent.
3. Strategy population per arena: 15 "informed" strategies with a
   persistent trade-level edge (losing base trades vetoed, well-timed
   multi-day holds inserted — both concentrated on high-volatility
   days, applied identically in the hidden window so the edge
   persists); 15 "lucky" pure-noise strategies chosen per family as the
   nearest unused candidates in z-normalized (in-sample Sharpe,
   in-sample win rate) space from a 150-candidate pool; 8 unmatched
   background strategies. Four rule families (trend, breakout,
   mean-reversion, volatility-timing) with seeded parameters and
   minimum-activity acceptance.
4. Labels: hidden `skill` flag; `oos_sharpe` = annualized Sharpe of the
   strategy's daily returns over the hidden 140-day continuation.
5. Split: 96 train / 64 test arenas; strategies inherit the split, so
   train labels cannot reveal any test arena's continuation regime.

## Why the transformations are realistic

Vetoing bad entries and catching favorable moves early is exactly what
genuine informational edge looks like in a trade record; concentrating
it in high-volatility states mirrors how real alpha tends to be
regime-conditional. Retaining lucky strategies whose in-sample Sharpe
and win rate match the skilled cohort reproduces the multiple-testing
selection bias that makes real backtest evaluation hard (the
deflated-Sharpe problem). The simulation layer is disclosed as
semi-synthetic in `dataset_description.md`; the market data underneath
is real, so volatility clustering, fat tails, and regime persistence
are genuine.

## Why this is not a standard benchmark clone

There is no public leaderboard, Kaggle competition, or textbook task
whose object of prediction is *the strategy track record itself*:
detect injected-but-realistic skill among Sharpe-matched lucky
look-alikes, rank future risk-adjusted performance within markets, and
commit to an exact-size portfolio under logical-consistency penalties.
The underlying price data is public, but the task's targets are
functions of undisclosed strategy mechanics, not of the price data
alone.

## Why training/representation learning is necessary

- The skill flag is defined by hidden generation mechanics whose only
  observable footprint is distributional; it must be *learned* from the
  3,648 labeled training strategies and transferred across disjoint
  arenas.
- Headline statistics are neutralized by construction (matching), so
  solvers must build sequence-derived, market-conditional
  representations of each track record — a representation-learning and
  structured-reasoning problem, not a lookup or rule.
- The R component requires modeling relative continuation performance
  within unseen markets; the P component requires calibrated
  decision-making under an exact-K constraint.

## Hidden structure the model must learn

Skill expresses itself as (a) trade quality that is *conditional on the
market's volatility state at entry*, (b) *temporal consistency* of edge
across the in-sample window, and (c) entry/exit timing that aligns with
subsequent returns beyond what the strategy's own rule family explains.
Luck expresses itself as streaks: concentrated in one regime, a few
outlier trades, or one sub-period.

## Why the naive baseline fails

`probe/probe.py` ranks by in-sample total return and a trade-level
Sharpe proxy — precisely the statistics the matching neutralizes.
Validated results: naive baseline scores **0.385** (detection D = 0.50;
among high-in-sample-Sharpe strategies its detection AUC collapses to
0.54, i.e. near chance inside the matched cohort). The reference
solution using conditional/temporal representations scores **0.762**
(D = 0.87, within-arena R = 0.26, portfolio precision 1.00). The
placeholder sample submission scores **0.027**; a perfect submission
scores **0.996**.

## Why novelty should be ≥ 4/5

The task imports a real, unsolved evaluation problem from quantitative
finance (backtest-overfitting forensics / deflated Sharpe) into an ML
benchmark with exact ground truth, which observational data can never
provide. The combination — heterogeneous targets, adversarially matched
negatives, disjoint-market transfer, composite scoring with logical
consistency penalties — does not correspond to any standard task
template; fifteen candidate designs were screened and the strongest
selected (`candidate_ideas.md`).

## Why difficulty should pass Eris gates

- Naive/simple-baseline gap: 0.385 vs 0.762 reference, 0.996 ceiling —
  large headroom both below and above the reference.
- Plain gradient boosting *on the provided columns* is impossible (the
  public tables contain almost no features); GBM becomes useful only
  after nontrivial sequence-feature construction, which is the intended
  skill. Detection inside the Sharpe-matched cohort is the hard core
  (naive ≈ chance there).
- The three graded outputs plus the exact-240 constraint punish
  single-metric shortcuts.

## Leakage checks performed

- Public files contain no hidden-window bars: `arenas.csv` stops at
  step 449; verified programmatically.
- Train/test arena disjointness verified; `oos_sharpe` labels exist
  only for training arenas.
- Strategy ids globally shuffled before assignment (id order carries no
  label information); arena ids order-shuffled.
- No file exposes group membership (informed / lucky / background),
  edge parameters, rule parameters, or asset identity.
- Trade logs are truncated at step 449 (`open_at_cutoff` flag) so no
  trade reveals continuation prices.
- External reconstruction: arenas are normalized, jittered windows of
  unnamed assets at unnamed offsets; even a solver who correctly
  re-identified an asset and time span would gain only the continuation
  *prices* — the targets additionally depend on undisclosed strategy
  mechanics (hidden positions), so labels are not externally
  recoverable. The jitter also prevents byte-exact matching against
  public archives.
- `prepare.py` verified byte-identical across repeated runs, including
  via the platform-style call
  `prepare(Path("dataset"), Path("tmp_public"), Path("tmp_private"))`.

## Licensing / provenance checks performed

- Zenodo record page and API fetched live; license field confirmed
  `cc-by-4.0`; DOI, uploader, publication date, and file inventory
  recorded in `data/source_metadata.json` (machine-readable).
- Archive downloaded credential-free; SHA-256 recorded and re-verified
  on every `download_data.py` run; failure modes are explicit errors,
  never fabricated data.
- Source name and URL are confined to `dataset_description.md`,
  `data/source_metadata.json`, `dataset/raw/source_license.txt`,
  `candidate_ideas.md`, and this audit — they appear in no
  solver-facing file (`problem_description.md`, `config.yaml`,
  `solution.ipynb`, probe scripts, public CSVs), verified by grep.

## Validation summary (all run locally, exit status clean)

1. `python download_data.py` — live download + checksum verification.
2. `python prepare.py` — deterministic build; rerun and platform-style
   call produce byte-identical outputs.
3. `python grade.py dataset/public/sample_submission.csv` → 0.027.
4. `python probe/test_grade.py` — 14/14 grader-behavior tests pass
   (perfect/sample/malformed/duplicate/missing/out-of-range/penalty/
   constant-prediction/determinism).
5. `python probe/probe.py` → 0.385 (naive baseline, no leakage).
6. `solution.ipynb` executed end-to-end (~30 s CPU) → valid
   `./working/submission.csv` → 0.762.

## Known limitations (disclosed)

- Daily bars, long/flat positions, no fees/slippage: the simulation is
  an evaluation environment, not a trading simulator; this is stated in
  `dataset_description.md`.
- `oos_sharpe` labels carry genuine market noise over 140 days; the
  composite metric was designed (family-stratified AUC, within-arena
  rank correlation, precision) to remain stable under it.
- The skill mechanism, while realistic in its statistical footprint, is
  synthetic; this is unavoidable for exact ground truth and is
  disclosed.
