#!/usr/bin/env python3
"""generate_dataset.py - synthetic online-grocery interaction log generator
for the "Restock Radar: Next-Week Grocery Purchase Ranking" task.

Simulates 6,000 shoppers browsing and buying from a 2,400-item catalog over
13 weeks (weeks 0-11 form the public log period; week 12 is the held-out
evaluation week). Every value is produced by the mechanistic simulation
below (numpy RNG draws + deterministic behavioural equations). No LLM
output is used anywhere, and no external dataset was copied or used.

Determinism: global seed 20260704. Each user is simulated with a child RNG
seeded (SEED, 100000 + user_index), so output does not depend on iteration
order. Running the script twice produces byte-identical CSVs. Requires only
numpy and pandas (Kaggle Python Docker image).

Outputs (to --out-dir, default ./raw next to this script):
    interactions.csv   one row per logged event (view / add_to_cart / purchase)
    items.csv          catalog with category, brand, price, launch date
    users.csv          user ids and signup dates
    promotions.csv     the promo calendar, one row per (week_start, item_id)

Usage:
    python generate_dataset.py [--out-dir raw]
"""

import argparse
import os

import numpy as np
import pandas as pd

SEED = 20260704
N_USERS = 6000
ITEMS_PER_CAT = 100
PUBLIC_WEEKS = 12          # weeks 0..11 = public log period
EVAL_WEEK = 12             # week 12 = held-out evaluation week
N_WEEKS = 13
WEEK0 = np.datetime64("2025-01-06T00:00:00")   # Monday, start of week 0
PROMO_PER_WEEK = 120       # 5% of the catalog promoted each week
DUP_FRAC = 0.015           # fraction of log rows duplicated (log replays)

# (name, typical restock period in weeks, price_lo, price_hi)
CATEGORIES = [
    ("produce",            1.0,  0.8,   6.0),
    ("dairy-eggs",         1.0,  1.2,   8.0),
    ("bakery",             1.0,  1.5,   9.0),
    ("meat-seafood",       1.5,  4.0,  25.0),
    ("deli",               1.0,  2.5,  14.0),
    ("frozen",             2.0,  2.0,  12.0),
    ("beverages",          1.5,  1.0,  10.0),
    ("snacks",             1.5,  1.0,   8.0),
    ("breakfast-cereal",   2.5,  2.5,   9.0),
    ("pantry-staples",     3.0,  1.0,  12.0),
    ("condiments-sauces",  4.0,  1.5,   9.0),
    ("baking",             5.0,  1.5,  11.0),
    ("coffee-tea",         3.0,  3.0,  18.0),
    ("desserts",           2.0,  2.0,  10.0),
    ("international",      3.0,  1.5,  12.0),
    ("health-wellness",    4.0,  3.0,  30.0),
    ("baby",               2.0,  4.0,  35.0),
    ("pet-supplies",       3.0,  2.0,  40.0),
    ("household-cleaning", 4.0,  2.0,  15.0),
    ("laundry",            5.0,  4.0,  20.0),
    ("paper-goods",        3.0,  3.0,  22.0),
    ("personal-care",      4.0,  2.0,  18.0),
    ("oral-care",          6.0,  2.0,  12.0),
    ("hair-care",          6.0,  3.0,  16.0),
]
N_CATS = len(CATEGORIES)
N_ITEMS = N_CATS * ITEMS_PER_CAT

# procedural brand-name syllables (synthetic; not LLM output)
SYL_A = ["Al", "Bel", "Cor", "Dan", "El", "Fen", "Gar", "Hal", "Jov", "Kel",
         "Lum", "Mar", "Nor", "Or", "Pel", "Quin", "Ras", "Sol", "Tam", "Ur",
         "Vel", "Wes", "Yor", "Zan"]
SYL_B = ["ba", "da", "fi", "go", "ka", "li", "mo", "na", "po", "ri", "sa",
         "ti", "vo", "wa", "ze"]
BRANDS_PER_CAT = 8


def build_catalog(rng):
    """Item catalog: category, brand, price, quality, base weight, launch."""
    combos = [a + b for a in SYL_A for b in SYL_B]
    order = rng.permutation(len(combos))
    brand_names = [combos[i] for i in order[:N_CATS * BRANDS_PER_CAT]]

    cat_idx = np.repeat(np.arange(N_CATS), ITEMS_PER_CAT)
    brand_local = rng.integers(0, BRANDS_PER_CAT, N_ITEMS)
    brands = [brand_names[c * BRANDS_PER_CAT + b]
              for c, b in zip(cat_idx, brand_local)]

    lo = np.array([CATEGORIES[c][2] for c in cat_idx])
    hi = np.array([CATEGORIES[c][3] for c in cat_idx])
    mid = np.sqrt(lo * hi)
    price = np.clip(mid * np.exp(rng.normal(0.0, 0.35, N_ITEMS)), lo, hi)
    price = np.round(price, 2)

    quality = rng.beta(5.0, 2.0, N_ITEMS)
    base_w = quality ** 1.5 * np.exp(rng.normal(0.0, 0.6, N_ITEMS))

    # launch weeks: 82% established (before week 0), 13% mid-history,
    # 5% cold launches in weeks 10-11 (retailer's autumn range refresh)
    u = rng.random(N_ITEMS)
    launch_week = np.where(
        u < 0.82, rng.integers(-20, 1, N_ITEMS),
        np.where(u < 0.95, rng.integers(1, 10, N_ITEMS),
                 rng.integers(10, 12, N_ITEMS)))
    return cat_idx, brands, price, quality, base_w, launch_week


def build_users(rng):
    """User latents. Only user_id + signup_date are exported."""
    u = rng.random(N_USERS)
    signup_week = np.where(u < 0.60, rng.integers(-40, 1, N_USERS),
                           rng.integers(1, 10, N_USERS))
    activity = np.clip(rng.gamma(2.4, 1.0, N_USERS), 0.3, 6.0)
    price_sens = np.exp(rng.normal(0.0, 0.25, N_USERS))
    explore = rng.beta(2.0, 9.0, N_USERS) + 0.03
    churn_week = np.where(rng.random(N_USERS) < 0.08,
                          rng.integers(8, 12, N_USERS), 99)
    n_home = rng.integers(3, 7, N_USERS)
    return signup_week, activity, price_sens, explore, churn_week, n_home


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=os.path.join(here, "raw"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rng_cat = np.random.default_rng([SEED, 1])
    rng_usr = np.random.default_rng([SEED, 2])
    rng_promo = np.random.default_rng([SEED, 3])
    rng_dup = np.random.default_rng([SEED, 4])

    cat_idx, brands, price, quality, base_w, launch_week = build_catalog(rng_cat)
    cat_period = np.array([CATEGORIES[c][1] for c in cat_idx])
    cat_median_price = np.array(
        [np.median(price[cat_idx == c]) for c in range(N_CATS)])[cat_idx]

    (signup_week, activity, price_sens, explore,
     churn_week, n_home) = build_users(rng_usr)

    # global category popularity (used to pick users' home categories)
    cat_pop = rng_usr.dirichlet(np.full(N_CATS, 3.0))

    # per-user home categories + affinities
    home_cats, home_aff = [], []
    for uix in range(N_USERS):
        r = np.random.default_rng([SEED, 50000 + uix])
        cats = r.choice(N_CATS, size=n_home[uix], replace=False, p=cat_pop)
        aff = r.dirichlet(np.full(len(cats), 1.2))
        home_cats.append(cats)
        home_aff.append(aff)

    # promo calendar: PROMO_PER_WEEK items per week, weeks 0..12.
    promo_sets = [rng_promo.choice(N_ITEMS, PROMO_PER_WEEK, replace=False)
                  for _ in range(N_WEEKS)]
    promo_mask = np.zeros((N_WEEKS, N_ITEMS), dtype=bool)
    for w, s in enumerate(promo_sets):
        promo_mask[w, s] = True

    # category seasonality (26-week cycle) + two trending / one declining cat
    phase = rng_cat.random(N_CATS) * 26.0
    trend = np.zeros(N_CATS)
    trending = rng_cat.choice(N_CATS, 3, replace=False)
    trend[trending[0]] = 0.020
    trend[trending[1]] = 0.015
    trend[trending[2]] = -0.015
    week_idx = np.arange(N_WEEKS)[:, None]
    season = (1.0 + 0.18 * np.sin(2 * np.pi * (week_idx + phase[None, :]) / 26.0)
              ) * (1.0 + trend[None, :] * week_idx)
    season_item = season[:, cat_idx]                       # (weeks, items)

    # per-week exposure weight per item
    active = launch_week[None, :] <= week_idx              # item on shelf
    fresh = (week_idx - launch_week[None, :] >= 0) & \
            (week_idx - launch_week[None, :] < 2)          # launch push
    expo = (base_w[None, :] * season_item
            * np.where(promo_mask, 4.0, 1.0)
            * np.where(fresh, 3.0, 1.0)
            * active)

    # normalized within-category and global exploration weights, per week
    cat_items = [np.where(cat_idx == c)[0] for c in range(N_CATS)]
    cat_probs = []                                         # [week][cat] -> p
    glob_probs = []
    for w in range(N_WEEKS):
        row = expo[w]
        cat_probs.append([row[ci] / row[ci].sum() if row[ci].sum() > 0 else
                          np.full(len(ci), 1.0 / len(ci)) for ci in cat_items])
        g = row ** 0.7
        glob_probs.append(g / g.sum())

    day_p = np.array([0.11, 0.10, 0.11, 0.12, 0.17, 0.22, 0.17])  # Mon..Sun

    ev_user, ev_item, ev_type, ev_ts = [], [], [], []       # int arrays
    eval_purchases = set()                                  # (uix, iix)

    for uix in range(N_USERS):
        r = np.random.default_rng([SEED, 100000 + uix])
        habits = {}                                         # iix -> [last_t, period]
        aff_of = dict(zip(home_cats[uix], home_aff[uix]))
        first_w = max(0, int(signup_week[uix]))
        for w in range(first_w, N_WEEKS):
            if w >= churn_week[uix]:
                break
            n_sess = r.poisson(activity[uix])
            for _ in range(n_sess):
                day = r.choice(7, p=day_p)
                hour = int(r.integers(8, 23))
                minute = int(r.integers(0, 60))
                sess_ts = (w * 7 + day) * 86400 + hour * 3600 + minute * 60
                t_frac = w + (day + hour / 24.0) / 7.0

                n_views = 6 + r.poisson(7)
                # --- assemble the session's viewed items ---------------
                viewed = []
                h_items = np.array(list(habits.keys()), dtype=int)
                if len(h_items):
                    due = np.array([(t_frac - habits[i][0]) / habits[i][1]
                                    for i in h_items])
                    wgt = np.clip(due, 0.15, 2.2) ** 2
                    k_rep = r.binomial(n_views, 0.42)
                    k_rep = min(k_rep, len(h_items))
                    if k_rep:
                        p = wgt / wgt.sum()
                        viewed.extend(r.choice(h_items, k_rep, replace=False,
                                               p=p).tolist())
                k_exp = r.binomial(n_views - len(viewed), explore[uix])
                k_aff = n_views - len(viewed) - k_exp
                for _ in range(k_aff):
                    c = r.choice(home_cats[uix], p=home_aff[uix])
                    viewed.append(int(r.choice(cat_items[c],
                                               p=cat_probs[w][c])))
                if k_exp:
                    viewed.extend(r.choice(N_ITEMS, k_exp, replace=False,
                                           p=glob_probs[w]).tolist())
                # de-dup within session, keep first occurrence
                seen, sess_items = set(), []
                for i in viewed:
                    if i not in seen and launch_week[i] <= w:
                        seen.add(i)
                        sess_items.append(i)

                # --- events -------------------------------------------
                checkout = sess_ts + len(sess_items) * 37 + 120
                n_bought = 0
                for pos, i in enumerate(sess_items):
                    vts = sess_ts + pos * 37
                    ev_user.append(uix); ev_item.append(i)
                    ev_type.append(0); ev_ts.append(vts)

                    if i in habits:
                        due = (t_frac - habits[i][0]) / habits[i][1]
                        p_cart = 0.34 * min(max(due, 0.05), 1.6)
                    else:
                        p_cart = 0.05
                        a = aff_of.get(cat_idx[i], 0.0)
                        p_cart += 0.10 * a * len(home_cats[uix])
                    p_cart += 0.10 * (quality[i] - 0.5)
                    rel = price[i] / cat_median_price[i] / price_sens[uix]
                    p_cart *= 0.6 + 0.4 * np.exp(-2.0 * (rel - 1.0) ** 2)
                    p_cart *= season_item[w, i] ** 0.5
                    if promo_mask[w, i]:
                        # banner placements are low-intent traffic: conversion
                        # collapses for non-habit viewers; habit shoppers
                        # stock up mildly. Net: views inflate far more than
                        # purchases, so raw view counts overstate promo demand.
                        p_cart *= 1.15 if i in habits else 0.42
                    p_cart = min(max(p_cart, 0.004), 0.9)

                    if r.random() < p_cart:
                        ev_user.append(uix); ev_item.append(i)
                        ev_type.append(1); ev_ts.append(vts + 5)
                        if r.random() < 0.72:
                            pts = checkout + n_bought * 7
                            n_bought += 1
                            ev_user.append(uix); ev_item.append(i)
                            ev_type.append(2); ev_ts.append(pts)
                            if i in habits:
                                habits[i][0] = t_frac
                            else:
                                per = cat_period[i] * float(
                                    np.exp(r.normal(0.0, 0.3)))
                                habits[i] = [t_frac, min(max(per, 0.7), 10.0)]
                            if w == EVAL_WEEK:
                                eval_purchases.add((uix, i))

    ev = pd.DataFrame({
        "uix": np.array(ev_user, dtype=np.int32),
        "iix": np.array(ev_item, dtype=np.int32),
        "etype": np.array(ev_type, dtype=np.int8),
        "ts": np.array(ev_ts, dtype=np.int64),
    })

    # log replays: duplicate a deterministic sample of rows verbatim
    n_dup = int(round(len(ev) * DUP_FRAC))
    dup_rows = ev.iloc[np.sort(rng_dup.choice(len(ev), n_dup, replace=False))]
    ev = pd.concat([ev, dup_rows], ignore_index=True)
    ev = ev.sort_values(["ts", "uix", "etype", "iix"],
                        kind="mergesort").reset_index(drop=True)

    ts = WEEK0 + ev["ts"].to_numpy().astype("timedelta64[s]")
    etype_names = np.array(["view", "add_to_cart", "purchase"])
    interactions = pd.DataFrame({
        "user_id": [f"U{u + 1:04d}" for u in ev["uix"]],
        "item_id": [f"I{i + 1:04d}" for i in ev["iix"]],
        "event_type": etype_names[ev["etype"]],
        "timestamp": pd.Series(ts).dt.strftime("%Y-%m-%d %H:%M:%S"),
    })

    items = pd.DataFrame({
        "item_id": [f"I{i + 1:04d}" for i in range(N_ITEMS)],
        "category": [CATEGORIES[c][0] for c in cat_idx],
        "brand": brands,
        "unit_price": price,
        "launch_date": pd.Series(
            WEEK0 + (launch_week * 7).astype("timedelta64[D]")
        ).dt.strftime("%Y-%m-%d"),
    })

    users = pd.DataFrame({
        "user_id": [f"U{u + 1:04d}" for u in range(N_USERS)],
        "signup_date": pd.Series(
            WEEK0 + (signup_week * 7).astype("timedelta64[D]")
        ).dt.strftime("%Y-%m-%d"),
    })

    promos = pd.DataFrame({
        "week_start": np.concatenate([
            np.full(PROMO_PER_WEEK,
                    str((WEEK0 + np.timedelta64(w * 7, "D")).astype(
                        "datetime64[D]")))
            for w in range(N_WEEKS)]),
        "item_id": [f"I{i + 1:04d}"
                    for w in range(N_WEEKS) for i in np.sort(promo_sets[w])],
    })

    interactions.to_csv(os.path.join(args.out_dir, "interactions.csv"),
                        index=False)
    items.to_csv(os.path.join(args.out_dir, "items.csv"), index=False)
    users.to_csv(os.path.join(args.out_dir, "users.csv"), index=False)
    promos.to_csv(os.path.join(args.out_dir, "promotions.csv"), index=False)

    n_eval = ev["ts"] >= EVAL_WEEK * 7 * 86400
    print(f"interactions: {len(interactions):,} rows "
          f"({n_dup:,} replay duplicates)")
    print(f"  public rows: {(~n_eval).sum():,}   eval-week rows: "
          f"{n_eval.sum():,}")
    print(f"eval-week distinct (user,item) purchases: "
          f"{len(eval_purchases):,}")


if __name__ == "__main__":
    main()
