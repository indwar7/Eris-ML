# Unfluke: Skill-vs-Luck Forensics for Systematic Trading Records

> **Domain: From Scratch.** Everything you need is in the public files.
> You build the entire pipeline — representations of trade records,
> models, calibration, and a final portfolio decision — from raw CSVs.
> No pretrained models, no LLMs, and no external data are needed or
> allowed; the anonymized markets cannot be meaningfully matched to any
> outside source.

## Overview

An evaluation desk reviews the records of **6,080 systematic trading
strategies** that were run on **160 anonymized markets** ("arenas").
Every strategy submitted a full in-sample track record: the exact trades
it took over 450 trading days, together with the daily bars of its
arena. Some of these strategies possess a **genuine, persistent edge**;
the rest are ordinary technical rules that merely *look* profitable in
sample — the kind of lucky survivors that multiple testing always
produces. Nothing in a strategy's headline statistics announces which is
which: among the high-performing strategies, the lucky ones were
deliberately retained *because* their in-sample Sharpe ratios and win
rates match those of the genuinely skilled ones.

After the in-sample period ends, every strategy kept trading for a
**hidden 140-day continuation window** on its arena. Your job, for each
of the 2,432 strategies on the 64 held-out test arenas, is to judge the
track record the way a forensic allocator would:

1. **`p_skill`** — the probability that the strategy has a genuine edge
   (rather than a lucky in-sample run);
2. **`oos_sharpe`** — the annualized Sharpe ratio the strategy realized
   over the hidden continuation window;
3. **`select`** — a portfolio decision: pick **exactly 240** of the
   2,432 test strategies for the desk's book.

This is not a price-forecasting task and not a standard tabular
benchmark: train and test strategies live on **disjoint arenas**, the
useful signal lives in the *joint structure* of each strategy's trade
sequence and its market context, and headline summary statistics are
deliberately uninformative about the skill flag. Solutions that win are
the ones that learn *how* skill expresses itself in trade records —
when the winning trades happen, how consistent the edge is across time
and market states — rather than *how much* profit a record shows.

## Data files

All public files are in `./dataset/public/`.

### `arenas.csv` — 72,000 rows (160 arenas × 450 steps)

Daily bars for the in-sample segment of every arena, train and test
alike. Prices are normalized so that each arena's first close is 1.0;
volume is normalized by the arena's median in-sample volume. Arenas are
real markets, anonymized and rescaled.

| Column | Type | Description |
| --- | --- | --- |
| `arena_id` | str | Arena identifier, `A001`–`A160`. |
| `step` | int | Trading day index within the arena, 0–449. |
| `open` | float | Normalized daily open. |
| `high` | float | Normalized daily high. |
| `low` | float | Normalized daily low. |
| `close` | float | Normalized daily close. |
| `volume` | float | Normalized daily traded volume. |

### `trades.csv` — 70,653 rows

The complete in-sample trade log of every strategy (train and test).
All strategies are long/flat: a trade opens at the close of
`entry_step` and closes at the close of `exit_step`.

| Column | Type | Description |
| --- | --- | --- |
| `strategy_id` | str | Strategy identifier, `S00001`–`S06080`. |
| `entry_step` | int | Day the position was opened (0–449). |
| `exit_step` | int | Day the position was closed (0–449). |
| `trade_return` | float | `close[exit_step] / close[entry_step] - 1` on the strategy's arena. |
| `open_at_cutoff` | int | 1 if the position was still open at step 449 and the trade was truncated there, else 0. |

### `train.csv` — 3,648 rows (strategies on the 96 training arenas)

| Column | Type | Description |
| --- | --- | --- |
| `strategy_id` | str | Strategy identifier. |
| `arena_id` | str | Arena the strategy traded. |
| `family` | str | Rule family: `trend`, `breakout`, `meanrev`, or `voltimer`. |
| `skill` | int | 1 if the strategy has a genuine persistent edge, else 0. |
| `oos_sharpe` | float | Annualized Sharpe ratio the strategy realized over its arena's hidden 140-day continuation window. |

### `test.csv` — 2,432 rows (strategies on the 64 test arenas)

| Column | Type | Description |
| --- | --- | --- |
| `strategy_id` | str | Strategy identifier. |
| `arena_id` | str | Arena the strategy traded. |
| `family` | str | Rule family: `trend`, `breakout`, `meanrev`, or `voltimer`. |

### `sample_submission.csv` — 2,432 rows

The required output format, filled with placeholder values
(`p_skill = 0.5`, `oos_sharpe = 0.0`, and an arbitrary valid set of 240
`select = 1` rows).

## Important notes

1. **Arena split.** Train and test strategies never share an arena, so
   per-arena memorization does not transfer; what transfers is what you
   learn about *strategy behavior*.
2. **Skilled strategies exist in both splits** and in every family. The
   proportion of skilled strategies in the test set is close to the
   proportion you can observe in `train.csv`.
3. **The hidden continuation window** is the 140 trading days
   immediately following step 449 of each arena. Its bars are never
   published. A strategy's `oos_sharpe` label is
   `mean(daily_returns) / std(daily_returns) * sqrt(365)` over those
   140 days (population standard deviation; the value is 0 when the
   strategy never traded), clipped to `[-10, 10]`. Daily returns are
   position-times-market-return; positions are the strategy's own.
4. **Summary statistics are a trap by construction.** For every
   genuinely skilled strategy there is a lucky look-alike whose
   in-sample Sharpe ratio *and* win rate match. Models built only on
   headline performance plateau far below the achievable score.
5. **Everything is reproducible from the public files.** Trade returns
   in `trades.csv` are consistent with `arenas.csv` closes to within
   rounding (8 decimal places).

## Evaluation

The composite score combines three components and a consistency
penalty. **Higher is better**; the score is bounded in `[0, 1]`.

### Component D — skill detection (weight 0.40)

For each of the 4 families `f` (over test strategies of that family
only), compute the ROC-AUC of your `p_skill` against the hidden `skill`
flag using the Mann-Whitney formulation with average ranks for ties:

```python
def auc(labels, scores):          # labels in {0,1}
    ranks = rank(scores, ties="average")
    n1, n0 = (labels == 1).sum(), (labels == 0).sum()
    return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

D = mean(max(0.0, 2.0 * auc_f - 1.0) for f in families)
```

Random guessing gives `auc_f = 0.5`, hence `D = 0`. The family
stratification means you cannot score by solving only the easiest
family.

### Component R — within-market forward ranking (weight 0.25)

For each of the 64 test arenas `a`, compute the Spearman rank
correlation `rho_a` between your predicted `oos_sharpe` and the true
`oos_sharpe` across the strategies of that arena (`rho_a = 0` if either
vector is constant within the arena). Then:

```python
R = max(0.0, mean(rho_a for a in test_arenas))
```

Because the correlation is computed *within* each arena, market-regime
guesses cancel out; R rewards ranking strategies against their peers on
the same market.

### Component P — portfolio precision (weight 0.35)

Among your 240 selected strategies (`select == 1`), let `prec` be the
fraction whose hidden `skill` flag is 1, and let `q` be the base rate
of skilled strategies in the full test set:

```python
P = clip((prec - q) / (1 - q), 0.0, 1.0)
```

Selecting at random gives `P = 0`; selecting only genuinely skilled
strategies gives `P = 1`.

### Consistency penalty

Your three outputs must tell one coherent story. Each test row is
checked for two contradictions:

1. `select == 1` while `p_skill < 0.2` (you picked a strategy you
   yourself consider almost certainly lucky);
2. `select == 1` while your predicted `oos_sharpe < 0` (you picked a
   strategy you yourself expect to lose money).

Each contradiction counts once per rule per row:

```python
penalty = min(0.001 * n_contradictions, 0.10)
```

### Final aggregation

```python
score = clip(0.40 * D + 0.25 * R + 0.35 * P - penalty, 0.0, 1.0)
```

The placeholder `sample_submission.csv` scores approximately 0.03. A
strong solution exceeds 0.60.

## Submission format

Write `./working/submission.csv` with **exactly 2,432 rows** (one per
`strategy_id` in `test.csv`, any order) plus a header row.

| Column | Type | Description |
| --- | --- | --- |
| `strategy_id` | str | A strategy id from `test.csv`. |
| `p_skill` | float | Probability in `[0, 1]` that the strategy has a genuine edge. |
| `oos_sharpe` | float | Predicted annualized out-of-sample Sharpe, in `[-10, 10]`. |
| `select` | int | Portfolio pick, 0 or 1. Exactly 240 rows must be 1. |

Requirements:

- exactly 2,432 rows, one per test `strategy_id`;
- ids must match `test.csv` exactly — no unknown, missing, or duplicate
  ids;
- no missing, non-numeric, or non-finite values;
- `p_skill` in `[0, 1]`; `oos_sharpe` in `[-10, 10]`; `select` in
  `{0, 1}` with `sum(select) == 240`;
- submissions violating any of these are rejected (not scored).

Example of a valid file:

```csv
strategy_id,p_skill,oos_sharpe,select
S00002,0.81,1.42,1
S00005,0.12,-0.30,0
S00007,0.55,0.65,0
S00011,0.97,2.10,1
```

## What a good solution looks like

- Reconstructs each strategy's daily position and P&L series from
  `trades.csv` and `arenas.csv` rather than trusting aggregate stats.
- Learns from `train.csv` labels *which behavioral signatures* of a
  trade record generalize across arenas — e.g. how the edge is
  distributed over time and market conditions — and encodes them as
  sequence-derived representations.
- Validates with grouped splits (by arena) to mimic the train/test
  structure, and calibrates `p_skill` before thresholding into the
  240-pick portfolio.
- Keeps the three outputs consistent by construction.
