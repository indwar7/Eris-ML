# Dataset Description

## Overview

This dataset supports **"Unfluke: Skill-vs-Luck Forensics for Systematic
Trading Records"**, a From Scratch task in which solvers must decide
which simulated systematic trading strategies possess a genuine,
persistent edge and which only look good in sample by luck, predict each
strategy's realized out-of-sample Sharpe ratio, and pick a fixed-size
portfolio.

The market data underneath the task is **genuine public cryptocurrency
market history** (see Source and License). The strategy layer on top of
it is **deterministic and semi-synthetic**: rule-based long/flat trading
strategies simulated on anonymized 590-day windows of the real price
series, with a hidden subset of strategies given a persistent
trade-level edge and a matched cohort of pure-noise strategies retained
because their in-sample statistics resemble the skilled ones.

Each public row therefore represents either one daily bar of a real,
anonymized market ("arena") or one trade / one strategy from the
simulated evaluation population. The ML skills tested are sequence
representation learning over trade records, leakage-aware grouped
validation, calibration, and decision-making under a composite metric —
not price forecasting.

## Source and License

- **Source dataset:** Binance cryptocurrencies historical daily data
  (uploader: Serhii Kanyhin), Zenodo record 8187872.
- **Source URL:** https://zenodo.org/records/8187872
- **Download URL (credential-free):**
  https://zenodo.org/records/8187872/files/historical%20data%20and%20indicators.zip?download=1
- **DOI:** 10.5281/zenodo.8187872
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **License URL:** https://creativecommons.org/licenses/by/4.0/legalcode
- **Citation:** Serhii Kanyhin (2023). *Binance cryptocurrencies
  historical daily data* [Data set]. Zenodo.
  https://doi.org/10.5281/zenodo.8187872
- **Access date:** 2026-07-07 (archive download; `data/source_metadata.json` records the latest verified rebuild date)
- **Archive SHA-256:**
  `ce9a73eb817d0d9c111f296f7dfdea1694d3bc8149d36375e6da25ce36fbed1f`
- **Commercial use allowed:** yes (CC BY 4.0)
- **Redistribution allowed:** yes, with attribution (CC BY 4.0);
  attribution is preserved here, in `dataset/raw/source_license.txt`,
  and in `data/source_metadata.json`.

The upstream record distributes real daily OHLCV market history
(open/high/low/close, quote volume in BUSD) for 289 cryptocurrency
trading pairs from 2021-06-01 to 2023-06-30, collected from a major
exchange. Only the raw OHLCV files are used; the record's precomputed
technical-indicator files are discarded.

**Data secrecy note.** The source is deliberately *not* named in any
solver-facing file (`problem_description.md`, `config.yaml`, the
reference notebook, or the public CSVs). Arenas are anonymized,
rescaled, and jittered (see Transformations) so that solver-side
identification of the underlying assets is impractical and, by design,
unrewarding: all hidden targets depend on undisclosed strategy
mechanics, not just on hidden future prices.

## Raw data

`dataset/raw/prices.csv` (18.1 MB, 195,588 rows) is the canonical raw
input for `prepare.py`: the 289 per-asset CSVs concatenated unchanged.

| Column | Type | Description |
| --- | --- | --- |
| `symbol` | str | Trading-pair ticker as published upstream. |
| `open_time` | date | Trading day (UTC), `YYYY-MM-DD`. |
| `close_time` | datetime | End of the trading day (UTC). |
| `open` | float | Daily open price. |
| `high` | float | Daily high price. |
| `low` | float | Daily low price. |
| `close` | float | Daily close price. |
| `volume_busd` | float | Daily traded quote volume (BUSD). |

## Processed data (what `prepare.py` builds)

1. **Arena construction.** 160 assets with at least 600 contiguous,
   clean daily bars are selected deterministically; one 590-day window
   is carved per asset at a seeded offset. Steps 0–449 form the public
   in-sample segment; steps 450–589 form the hidden continuation
   segment (never published).
2. **Anonymization.** Prices are divided by the window's first close,
   volume by the median in-sample volume; every day's OHLC is
   multiplied by a deterministic jitter factor in [0.998, 1.002]
   (volume: [0.98, 1.02]); asset identity, calendar dates, and window
   offsets are dropped; arena ids are assigned in shuffled order.
   All simulation runs on the jittered series, so every published
   number is internally consistent.
3. **Strategy population.** Per arena: 15 "informed" strategies with a
   persistent trade-level edge concentrated in specific market states;
   150 pure-noise rule strategies of the same four families
   (`trend`, `breakout`, `meanrev`, `voltimer`), of which 15 are
   retained as "lucky" look-alikes — the nearest unused candidates to
   the informed cohort in z-normalized (in-sample Sharpe, in-sample
   win-rate) space — plus 8 unmatched "background" strategies.
   38 published strategies per arena; 6,080 total.
4. **Labels.** `skill` = 1 for informed strategies, else 0.
   `oos_sharpe` = annualized Sharpe (mean/std × √365, population std,
   0 if the strategy never traded, clipped to [-10, 10], rounded to 4
   decimals) of the strategy's daily returns over the hidden 140-day
   continuation, where the same strategy mechanics keep running.
5. **Split.** 96 arenas (3,648 strategies) are training data with full
   labels; the other 64 arenas (2,432 strategies) are the test set.
   Train and test strategies never share an arena.

The exact generation parameters (edge mechanics, matching metric, seeds)
are documented in `prepare.py` itself; they are hidden from solvers.

## File structure

```text
unfluke/
  README.md
  candidate_ideas.md
  self_audit.md
  dataset_description.md
  problem_description.md
  rubrics.md
  config.yaml
  download_data.py
  prepare.py
  grade.py
  solution.ipynb

  data/
    source_metadata.json          # machine-readable provenance (JSON)
    raw/
      source_daily_ohlcv.zip     # upstream archive (111 MB; recreated by
                                  #   download_data.py; omitted from the
                                  #   upload archive, checksum recorded)
      historical_data/            # 289 per-asset OHLCV CSVs extracted from
                                  #   the archive (recreated by
                                  #   download_data.py; omitted from the
                                  #   upload archive)
      checksums.sha256            # SHA-256 of the archive and prices.csv

  dataset/
    raw/
      prices.csv                  # canonical raw input (195,588 rows)
      source_license.txt          # upstream license + citation notice
    public/
      arenas.csv                  # 72,000 rows: in-sample bars, 160 arenas
      trades.csv                  # 70,653 rows: in-sample trade logs
      train.csv                   # 3,648 rows: labeled train strategies
      test.csv                    # 2,432 rows: unlabeled test strategies
      sample_submission.csv       # 2,432 rows: required output format
    private/
      answers.csv                 # 2,432 rows: hidden answer key

  probe/
    probe.py                      # naive baseline (shows naive is poor)
    test_grade.py                 # grader behavior tests
```

## Columns

### `dataset/public/arenas.csv`

| Column | Type | Description |
| --- | --- | --- |
| `arena_id` | str | Arena identifier `A001`–`A160`. |
| `step` | int | Trading-day index, 0–449. |
| `open` | float | Normalized, jittered daily open. |
| `high` | float | Normalized, jittered daily high. |
| `low` | float | Normalized, jittered daily low. |
| `close` | float | Normalized, jittered daily close. |
| `volume` | float | Normalized, jittered daily volume. |

### `dataset/public/trades.csv`

| Column | Type | Description |
| --- | --- | --- |
| `strategy_id` | str | Strategy identifier `S00001`–`S06080`. |
| `entry_step` | int | Day the long position was opened. |
| `exit_step` | int | Day the position was closed (or 449 if truncated). |
| `trade_return` | float | `close[exit]/close[entry] - 1`, rounded to 8 dp. |
| `open_at_cutoff` | int | 1 if the trade was truncated at step 449. |

### `dataset/public/train.csv`

| Column | Type | Description |
| --- | --- | --- |
| `strategy_id` | str | Strategy identifier. |
| `arena_id` | str | Home arena (one of the 96 training arenas). |
| `family` | str | `trend`, `breakout`, `meanrev`, or `voltimer`. |
| `skill` | int | 1 = genuine persistent edge, 0 = no edge. |
| `oos_sharpe` | float | Realized annualized Sharpe over the hidden 140-day continuation window (4 dp). |

### `dataset/public/test.csv`

| Column | Type | Description |
| --- | --- | --- |
| `strategy_id` | str | Strategy identifier. |
| `arena_id` | str | Home arena (one of the 64 test arenas). |
| `family` | str | `trend`, `breakout`, `meanrev`, or `voltimer`. |

### `dataset/public/sample_submission.csv`

| Column | Type | Description |
| --- | --- | --- |
| `strategy_id` | str | Test strategy identifier. |
| `p_skill` | float | Placeholder 0.5. |
| `oos_sharpe` | float | Placeholder 0.0. |
| `select` | int | Placeholder valid portfolio (exactly 240 ones). |

### `dataset/private/answers.csv`

| Column | Type | Description |
| --- | --- | --- |
| `strategy_id` | str | Test strategy identifier. |
| `skill` | int | Hidden skill flag. |
| `oos_sharpe` | float | Realized out-of-sample Sharpe (4 dp). |
| `family` | str | Rule family (used for stratified detection scoring). |
| `arena_id` | str | Home arena (used for within-arena ranking scoring). |
| `is_sharpe_tercile` | str | `low`/`mid`/`high` in-sample-Sharpe tercile (diagnostic bucket reported by the grader). |

## Known quirks

- Arena bars are real market history: fat tails, volatility clusters,
  trends, and occasional extreme daily moves are genuine, not bugs.
- `trade_return` reconciles with `arenas.csv` closes to within the 8-dp
  rounding of both files.
- Strategies with a position still open at step 449 have that trade
  truncated and flagged `open_at_cutoff = 1`.
- The `skill`/no-skill populations deliberately overlap in headline
  statistics (in-sample Sharpe, win rate); separating them requires
  conditional / temporal structure, which is the point of the task.
- `oos_sharpe` labels include real market noise: a skilled strategy can
  still lose over 140 days, and a lucky one can win again. The
  composite metric (family-stratified AUC + within-arena rank
  correlation + portfolio precision) is designed to be stable under
  this noise.

## Reproducibility

```bash
python download_data.py   # fetch + verify the public archive, build dataset/raw/prices.csv
python prepare.py         # deterministically rebuild dataset/public and dataset/private
```

Both scripts are seeded and idempotent: rerunning `prepare.py` produces
byte-identical public/private files. `prepare.py` also supports the
platform call `prepare(dataset_dir, public_dir, private_dir)`, searches
several standard locations for `prices.csv`, falls back to copying
already-prepared outputs, and as a last resort re-runs the documented
public download.

## Limitations and assumptions

- The upstream uploader collected exchange market data and published it
  on Zenodo under CC BY 4.0; exchange OHLCV data is factual market
  history, and the permissive license plus preserved attribution are
  relied on for redistribution (see `self_audit.md`).
- Daily bars limit the realism of the strategy simulation (no intraday
  fills or fees); this is acceptable because the task evaluates
  forensic inference on the records, not trading realism.
- The skill mechanism is synthetic by necessity — real ground truth for
  "genuine edge" does not exist in observational data. It was designed
  so that its statistical footprint (persistent, regime-conditional
  trade quality) mirrors what real allocators look for.
