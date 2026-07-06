# Synthetic Online-Grocery Interaction Logs — Views, Carts, Purchases & Promo Calendar for Next-Week Purchase Ranking

Thirteen weeks of event-level interaction logs (views, add-to-carts,
purchases) for 6,000 shoppers across a 2,400-item grocery catalog, with the
retailer's planned promotion calendar. Built for the **"Restock Radar:
Next-Week Grocery Purchase Ranking"** task: ranking the items each user is
most likely to purchase in the following week.

---

## 1. Provenance & license

- **Type:** fully synthetic. Every value is produced by the mechanistic
  behavioural simulation in [`generate_dataset.py`](generate_dataset.py)
  (numpy RNG draws + deterministic preference/habit equations). **No LLM
  output is used anywhere in the data** (brand names are procedural syllable
  combinations), and no Kaggle competition or external dataset was copied
  or used.
- **Reproducibility:** the generator is fully deterministic (global seed
  `20260704`; each user is simulated with a child RNG seeded
  `(20260704, 100000 + user_index)` so output does not depend on iteration
  order). Running `python generate_dataset.py` twice produces
  **byte-identical** CSVs. Requires only `numpy` and `pandas` (Kaggle Python
  Docker image).
- **License:** CC0 1.0 (public domain dedication) — synthetic data created
  for this task; commercial use permitted.

## 2. Store story (what the data represents)

An online grocer logs every catalog interaction. Shoppers have stable
category preferences and **restock habits**: once a household adopts an
item, it is repurchased on a roughly periodic cycle (produce ≈ weekly;
laundry or oral-care ≈ every 5–6 weeks). Each week the retailer also runs a
**promotion cycle** (120 banner placements): promoted items collect far more
views from low-intent banner traffic, but only a modest real sales lift —
so raw view counts systematically overstate promo-week demand.

| Piece | Span / size |
|---|---|
| Interaction log | 2025-01-06 → 2025-04-06 (weeks 0–12; hourly-resolution timestamps) |
| Weeks 0–11 | "history" period — 2,401,547 log rows |
| Week 12 (2025-03-31 →) | prediction week — logged in `interactions.csv` for completeness of the raw upload; the downstream task withholds it (see §6) |
| Users | 6,000 (5,953 appear in the log; signups 2024-04-01 → 2025-03-10) |
| Items | 2,400 across 24 categories, 192 procedural brands |

## 3. Files & schema

This dataset upload contains exactly five files, all at the top level: the
four CSVs documented below plus `generate_dataset.py` (the reproducible
generation script).

### `interactions.csv` — 2,641,518 rows (one per logged event)

| Column | Type | Description |
|---|---|---|
| `user_id` | string | Shopper identifier (`U0001`–`U6000`). |
| `item_id` | string | Catalog item (`I0001`–`I2400`). |
| `event_type` | string | `view`, `add_to_cart`, or `purchase`. Funnel events share the session's timestamp neighbourhood (view → cart +5 s → purchase at checkout). |
| `timestamp` | datetime `YYYY-MM-DD HH:MM:SS` | Event time (UTC). |

**Log replays:** ~1.5% of rows (39,053) are **exact duplicates** of another
row — the ingestion pipeline occasionally replays events. They are byte-for-
byte identical rows and must be deduplicated before any counting.

### `items.csv` — 2,400 rows

| Column | Type | Description |
|---|---|---|
| `item_id` | string | Catalog item. |
| `category` | string | One of 24 grocery categories (`produce`, `dairy-eggs`, …, `hair-care`). |
| `brand` | string | One of 192 brands (8 per category). |
| `unit_price` | float | Price in dollars (0.97–35.00). |
| `launch_date` | date | First day the item is on the shelf. Items cannot be interacted with before launch. 119 items launch in weeks 10–11 (recent range refresh). |

### `users.csv` — 6,000 rows

| Column | Type | Description |
|---|---|---|
| `user_id` | string | Shopper. |
| `signup_date` | date | Account creation; users generate no events before it. |

### `promotions.csv` — 1,560 rows (120 items × 13 weeks)

| Column | Type | Description |
|---|---|---|
| `week_start` | date | Monday of the promo week (weeks 0–12). |
| `item_id` | string | Promoted (banner-placed) item that week. |

Promotions are **planned in advance** by the retailer, so the calendar
legitimately includes week 12 — the prediction week.

## 4. Generation process (mechanistic model)

Full details and exact constants are in `generate_dataset.py`; summary:

1. **Catalog.** Each category has a typical restock period (1–6 weeks) and
   price band; item prices are lognormal within band; each item draws a
   quality level and a base popularity weight.
2. **Shoppers.** Each user draws 3–6 "home" categories with Dirichlet
   affinities, a weekly activity rate, a price sensitivity, an exploration
   propensity, and a signup week; 8% of users churn during weeks 8–11.
3. **Sessions.** Per user-week, a Poisson number of sessions (weekend-
   weighted days, 08:00–22:00). A session views ~13 items drawn from three
   pools: the user's own repeat items (weighted by how *overdue* they are),
   home-category items (weighted by popularity × seasonality × promo/launch
   exposure boosts), and exploration across the catalog.
4. **Funnel.** Each viewed item is added to cart with a probability driven
   by habit due-ness, category affinity, quality, price fit relative to the
   category median, and seasonality; carted items convert to purchase with
   p = 0.72 at checkout.
5. **Restock habits.** A first purchase creates a per-(user, item) habit
   with period = category period × lognormal jitter. Repurchase probability
   rises as elapsed time approaches the period — recently bought items are
   *not* due, long-overdue items are.
6. **Promotions.** 120 items per week get ~4× exposure in the browsing
   pools, but banner traffic is low-intent: cart conversion is multiplied by
   0.42 for non-habit viewers (1.15 for existing habit shoppers). Measured
   effect per item-week: **views ×2.55, purchases ×1.35, conversion ×0.53**.
7. **Seasonality & trends.** Category demand follows a 26-week sinusoid;
   two categories trend upward and one declines across the record.
8. **New launches.** Items launched mid-record get a 2-week ~3× exposure
   push; 119 "cold" items launch in weeks 10–11 with little history.
9. **Log replays.** A deterministic 1.5% sample of rows is duplicated
   verbatim, mimicking ingestion replays.

## 5. Controlled complexity (planted, documented)

| Property | Where | Skill it tests |
|---|---|---|
| Views ×2.55 vs purchases ×1.35 on promo weeks | `interactions.csv` + `promotions.csv` | event weighting / exposure-bias handling: purchases, not views, measure demand |
| 1.5% exact-duplicate rows | `interactions.csv` | data cleaning before aggregation |
| ~68% of next-week purchases are repeats with per-item periodicity | purchase streams | habit modeling: "overdue-ness" beats raw recency (a 5-week laundry item bought last week is *not* due) |
| 119 items launched in weeks 10–11; ≈4.6% of prediction-week purchases | `items.csv` launch dates | cold-start coverage via content/promo features, not just co-occurrence |
| Dense, autocorrelated per-user event streams | all of `interactions.csv` | temporal validation (random row splits leak habit rows across folds) |
| Week-12 promo calendar published in advance | `promotions.csv` | using legitimately available future information |
| Global popularity is a weak signal (top-20 list scores ≈0.03 NDCG@20 vs ≈0.38 for a simple personalized heuristic) | evaluation design | personalization over popularity |

## 6. Intended downstream splits (context only — not part of this upload)

A separate task submission provides a `prepare.py` that transforms the raw
files above into public/private splits during task creation on the platform.
The split files (`train.csv`, `test.csv`, `sample_submission.csv`,
`answers.csv`) are **not included in this dataset upload**; they are
documented here so reviewers can see the intended use.

- **Public `train.csv`**: `interactions.csv` rows from weeks 0–11 only
  (replay duplicates left in, as documented). Week-12 events are withheld.
- **Public `items.csv` / `users.csv` / `promotions.csv`**: as uploaded
  (promotions include week 12 — the calendar is planned ahead).
- **Public `test.csv`** (1,600 rows): the target `user_id`s — users with ≥3
  distinct recent purchases (weeks 8–11) and ≥1 week-12 purchase, sampled
  deterministically — plus the prediction week start date (2025-03-31).
- **Public `sample_submission.csv`** (32,000 rows): required format,
  `user_id,rank,item_id` with ranks 1–20 per user (arbitrary items).
- **Private `answers.csv`** (11,528 rows): the distinct
  `(user_id, item_id)` purchases made by target users during week 12
  (mean 7.2 items per user). Week-12 rows never appear in any public file,
  so the answers cannot be recovered from the public split.

## 7. Sample-size summary

| Piece | Count |
|---|---|
| Raw log rows | 2,641,518 (incl. 39,053 replay duplicates) |
| Public-period rows (weeks 0–11) | 2,401,547 — 1,762,281 views / 372,008 carts / 267,258 purchases |
| Users / items / categories / brands | 6,000 / 2,400 / 24 / 192 |
| Prediction-week purchasers | 4,344 (of whom 1,600 are sampled as targets) |
| Answer pairs (target users) | 11,528 (mean 7.21 items/user) |

## 8. Regenerating

```bash
python generate_dataset.py --out-dir .   # regenerates the four CSVs next to the script
```

Byte-identical on every run. (Without `--out-dir`, the script writes into a
`raw/` subfolder next to itself.)
