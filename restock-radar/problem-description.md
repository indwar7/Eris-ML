# Restock Radar: Next-Week Grocery Purchase Ranking

> **Domain: Recommendation.** You build a personalized ranking system from
> raw interaction logs — no pretrained models, no LLMs, and no external
> data. Everything you need is in the public files; the whole pipeline
> (load → features → model → ranked lists) must run end-to-end in your
> notebook.

## Task

An online grocer logs every product view, add-to-cart, and purchase. For
each of **1,600 target users**, rank the **20 items** they are most likely
to **purchase during the prediction week (2025-03-31 to 2025-04-06)** —
the week immediately after the public logs end. Rank 1 is your strongest
recommendation. Submissions are scored with **NDCG@20** (see Metric), so
the *order* of your 20 items matters.

The predictive signal lives in each shopper's habits: what they buy on
repeat and how often it is due for restock, which categories they favour,
what their recent sessions show, and which items the retailer will promote
next week. Global bestseller lists score near the floor on this task.

## Data files

All public files are in `./dataset/public/`.

### `train.csv` — 2,401,547 rows (weeks 0–11 of the event log)

| Column | Type | Role |
|---|---|---|
| `user_id` | str | Shopper (`U0001`–`U6000`). |
| `item_id` | str | Catalog item (`I0001`–`I2400`). |
| `event_type` | str | `view`, `add_to_cart`, or `purchase`. |
| `timestamp` | str, `YYYY-MM-DD HH:MM:SS` | Event time (UTC). Weeks run Monday–Sunday starting 2025-01-06. |

### `items.csv` — 2,400 rows

`item_id`, `category` (24 grocery categories), `brand` (192 brands),
`unit_price` (float, dollars), `launch_date` (date the item first became
available; items generate no events before it).

### `users.csv` — 6,000 rows

`user_id`, `signup_date` (users generate no events before signup).

### `promotions.csv` — 1,560 rows

`week_start` (Monday), `item_id`: the retailer's banner-promotion calendar,
120 items per week. **Promotions are planned ahead, so the calendar
includes the prediction week (`week_start = 2025-03-31`)** — you may use
it.

### `test.csv` — 1,600 rows

`user_id` (the target users), `predict_week_start` (constant,
`2025-03-31`). Provide a ranked list for exactly these users.

### `sample_submission.csv`

The required output format (32,000 rows = 1,600 users × ranks 1–20),
filled with arbitrary items.

## Important data notes (read carefully)

1. **Log replays (duplicates):** ~1.5% of `train.csv` rows are exact
   byte-for-byte duplicates of another row, an artifact of the ingestion
   pipeline. Deduplicate before counting anything, or every frequency
   feature is inflated.
2. **Views are exposure-biased.** Promoted items collect roughly 2.5× the
   views of comparable weeks but only ~1.35× the purchases — banner traffic
   converts at about half the normal rate. Raw view counts therefore
   systematically overstate demand for promoted items; purchases are the
   reliable signal of intent.
3. **Repeat purchases dominate — but timing matters.** About two-thirds of
   a typical week's purchases are items the user has bought before, and
   each (user, item) pair has a fairly stable restock rhythm inherited from
   its category (produce ≈ weekly; laundry or oral-care ≈ every 5–6 weeks).
   An item bought last week may be far *less* likely this week than one
   bought five weeks ago — "recently purchased" and "due for restock" are
   different things.
4. **New items exist.** 119 items launched in the final two public weeks;
   they account for roughly 4–5% of prediction-week purchases. Pure
   co-occurrence methods cannot rank them — use the catalog metadata,
   launch dates, and the promo calendar.
5. **Users churn and signup mid-record.** Do not assume every user has a
   full 12-week history; `signup_date` bounds each user's stream. All
   target users, however, have recent purchase activity.
6. **The evaluation is forward-censored.** No week-12 events appear in any
   public file. Only the promo calendar legitimately looks into the
   prediction week.

## Metric

**Mean NDCG@20** over the 1,600 target users, with binary relevance
(rel = 1 iff the user purchased that item at least once during the
prediction week):

```python
def ndcg_at_20(ranked_items, purchased_set):        # one user
    dcg = sum(1.0 / np.log2(r + 1)
              for r, it in enumerate(ranked_items, start=1)
              if it in purchased_set)
    ideal = min(len(purchased_set), 20)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal + 1))
    return dcg / idcg
```

The final score is the unweighted mean across all target users. Range
[0, 1]; higher is better. Every target user has at least one prediction-
week purchase (mean ≈ 7). Reference points: a random submission scores
≈ 0.006 and the global purchase-popularity top-20 scores ≈ 0.03, while a
well-built personalized model exceeds 0.45.

## Submission format

Write a CSV file named **`submission.csv`** to `./working/` with exactly
these three columns:

```
user_id,rank,item_id
U0007,1,I0412
U0007,2,I1300
...
```

- Exactly **20 rows per target user** — one for each rank 1–20 — for
  exactly the 1,600 `user_id`s in `test.csv` (32,000 rows + header), in
  any row order.
- `rank` must be the integers 1–20, each used exactly once per user
  (1 = strongest recommendation).
- An `item_id` may appear **at most once per user**. Item ids not in the
  catalog are scored as not-purchased. Missing/extra users, missing or
  duplicated ranks, or duplicated items within a user make the submission
  invalid.

## Rules & environment

- **Build from the provided data only** — no pretrained or foundation
  models, no LLMs, and no external data.
- Use only libraries available in the **Kaggle Python Docker image**
  (pandas, numpy, scikit-learn, xgboost, lightgbm, tensorflow, pytorch, …).
- **No LLM-generated outputs** may be used anywhere in your solution.
- Your notebook must run **end-to-end, top to bottom**: load the public
  data → build features → model → write `./working/submission.csv`.
- Seed everything; your run should be reproducible.
