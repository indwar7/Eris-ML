# Rubrics

Task-specific criteria for "Unfluke: Skill-vs-Luck Forensics for
Systematic Trading Records". The task rewards forensic evaluation
modeling: reconstructing behavior from trade records, learning which
behavioral signatures generalize across markets, and making calibrated,
internally consistent decisions.

## Required

1. **Reconstructs strategy behavior from the raw records**
   - Type: `DATA_HANDLING`
   - Importance: `REQUIRED`
   - Criterion: The solution must derive per-strategy behavioral
     information by joining `trades.csv` with `arenas.csv` (e.g. daily
     position/P&L series, trade timing relative to market conditions)
     rather than modeling only per-strategy aggregate statistics or the
     metadata columns of `train.csv`/`test.csv`.

2. **Uses arena-grouped validation**
   - Type: `TRAINING`
   - Importance: `REQUIRED`
   - Criterion: Any validation split used for model selection or early
     stopping must group by `arena_id` (train and test strategies never
     share an arena), not split strategies of the same arena across
     folds.

3. **Avoids data leakage**
   - Type: `TRAINING`
   - Importance: `REQUIRED`
   - Criterion: The solution must not use `dataset/private/`, external
     market data, or any attempt to de-anonymize arenas; it must train
     only on files in `./dataset/public/`.

4. **Produces a valid, complete submission**
   - Type: `CODE_QUALITY`
   - Importance: `REQUIRED`
   - Criterion: The notebook must write `./working/submission.csv` with
     exactly 2,432 rows, the four required columns, `p_skill` in
     [0, 1], `oos_sharpe` in [-10, 10], and exactly 240 rows with
     `select == 1`, matching the test ids exactly.

5. **Keeps the three outputs logically consistent**
   - Type: `MODELING`
   - Importance: `REQUIRED`
   - Criterion: The portfolio must be constructed from the model's own
     beliefs (no row with `select == 1` and `p_skill < 0.2`, and no row
     with `select == 1` and predicted `oos_sharpe < 0`), so the
     grader's consistency penalty is zero or negligible.

## Recommended

6. **Learns conditional / temporal edge signatures, not just headline stats**
   - Type: `FEATURE_ENGINEERING`
   - Importance: `RECOMMENDED`
   - Criterion: The solution should build representations that capture
     *when* and *how consistently* a strategy wins (e.g. trade outcomes
     conditioned on market state, stability of edge across sub-periods)
     and should demonstrate — via validation or ablation — that these
     beat summary statistics such as in-sample Sharpe or win rate,
     which the task deliberately makes uninformative.

7. **Targets each score component deliberately**
   - Type: `MODELING`
   - Importance: `RECOMMENDED`
   - Criterion: The solution should address all three graded outputs
     (calibrated skill probabilities; within-arena ranking of
     out-of-sample Sharpe; a 240-pick portfolio chosen to maximize
     expected precision), not optimize a single component and fill the
     rest with placeholders.

8. **Handles the ranking target's noise structure**
   - Type: `MODELING`
   - Importance: `RECOMMENDED`
   - Criterion: Because `oos_sharpe` is graded by *within-arena* rank
     correlation, the solution should model relative ordering (e.g.
     within-arena rank targets, arena-normalized labels, or blending
     skill probability into the ranking) rather than regressing raw
     Sharpe values dominated by cross-arena regime noise.

9. **Calibrates probabilities before the portfolio decision**
   - Type: `MODELING`
   - Importance: `RECOMMENDED`
   - Criterion: The solution should check (and if needed correct) the
     calibration of `p_skill` on held-out arenas before thresholding
     into the exact-240 selection.

10. **Explains the forensic reasoning**
    - Type: `COMMUNICATION`
    - Importance: `RECOMMENDED`
    - Criterion: The notebook should briefly explain which behavioral
      signatures separate genuine edge from luck in this data and how
      each modeling choice serves one of the three score components.

11. **Runs deterministically within budget**
    - Type: `CODE_QUALITY`
    - Importance: `RECOMMENDED`
    - Criterion: The notebook should fix random seeds, run end-to-end
      in under 30 minutes on CPU using only standard Kaggle-image
      libraries, and produce the same submission on rerun.
