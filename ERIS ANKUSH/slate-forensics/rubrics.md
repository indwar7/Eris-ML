# Rubrics

## Required

1. **Respects anchor-time causality**
   - Type: `TRAINING`
   - Importance: `REQUIRED`
   - Criterion: Features for a request must be computed only from transactions
     strictly before that request's `anchor_date`. The solution must not use
     post-anchor transactions when learning from or predicting train-anchor
     requests (the evaluation anchor has no post-anchor data, so violating
     this creates train/test skew and constitutes leakage).

2. **Matches the required submission format exactly**
   - Type: `CODE_QUALITY`
   - Importance: `REQUIRED`
   - Criterion: Writes `./working/submission.csv` with columns
     `slate_id, corrupted, mode, position, stock_code`; exactly 10 rows per
     evaluation slate (7,570 total); positions 1..10 once each; no duplicate
     items within a slate; flag and mode constant within each slate; modes
     drawn from the allowed vocabulary.

3. **Learns the healthy policy rather than hand-coding repairs**
   - Type: `MODELING`
   - Importance: `REQUIRED`
   - Criterion: Repaired slates come from a trained model of the healthy
     service (e.g., candidate generation plus a learned ranker fit on
     `healthy_slates_train.csv`), not from copying emitted slates, global
     bestseller lists, or other fixed heuristics alone.

4. **Uses customer-disjoint validation**
   - Type: `TRAINING`
   - Importance: `REQUIRED`
   - Criterion: Because evaluation customers never appear in training, any
     validation split used for model or threshold selection must hold out
     customers (not just slates), so estimates reflect the true
     policy-generalisation setting.

5. **Couples the three outputs consistently**
   - Type: `MODELING`
   - Importance: `REQUIRED`
   - Criterion: The submitted flag, mode and slate must be assembled jointly:
     clean requests resubmit the emitted slate with mode `none`; corrupted
     requests submit a changed slate with a non-`none` mode. Submissions that
     ignore the coupling (and rely on the grader's consistency multiplier to
     catch it) do not meet this criterion.

## Recommended

6. **Builds forensic signatures for all three failure modes**
   - Type: `FEATURE_ENGINEERING`
   - Importance: `RECOMMENDED`
   - Criterion: The audit/diagnosis models should include features targeting
     each documented symptom family — personalisation loss (popularity
     concentration, affinity to the customer's history), price-band
     displacement relative to the customer's spending, and staleness
     (old-window vs recent-window demand) — rather than a single generic
     feature set.

7. **Handles the real-data quirks deliberately**
   - Type: `DATA_HANDLING`
   - Importance: `RECOMMENDED`
   - Criterion: Cancellation invoices, rows with missing `customer_id`,
     zero-price rows and non-product service codes are filtered or treated
     with an explicit, stated rationale before behavioural statistics are
     computed.

8. **Validates against the official composite**
   - Type: `TRAINING`
   - Importance: `RECOMMENDED`
   - Criterion: Component models and decision thresholds are selected using
     the actual composite score (e.g., by scoring pseudo-submissions on
     held-out customers with `grade.py`), not accuracy or another proxy that
     ignores the bucket weighting and the consistency multiplier.

9. **Explains the diagnosis reasoning**
   - Type: `COMMUNICATION`
   - Importance: `RECOMMENDED`
   - Criterion: The notebook briefly explains which signals separate each
     failure mode and reports per-mode validation quality, not just a single
     aggregate number.
