#!/usr/bin/env python3
"""prepare.py — build the Slate Forensics challenge from the genuine
Online Retail II data (UCI id 502, CC BY 4.0; see data/source_metadata.json).

REVIEWER-FACING FILE. This script documents the full hidden generation
process. It is not shipped to solvers; problem_description.md deliberately
does not reveal the internals below.

What is real and what is synthetic
----------------------------------
REAL (untouched): every transaction row, invoice id, stock code, product
description, quantity, timestamp, unit price, customer id and country that
solvers see comes verbatim from the source workbook.

SYNTHETIC OVERLAY (deterministic, seed 20260705): a hidden "You may also
like" production policy is replayed at six anchor dates against each
eligible customer's real purchase history, emitting a 10-item slate per
(customer, anchor). A seeded ~35% of slates is then corrupted by exactly
one of three realistic production failure modes. The observable emitted
slate is the healthy slate for clean requests and the buggy slate for
corrupted requests.

The hidden healthy policy P(customer c, anchor A)
-------------------------------------------------
Inputs: only policy-valid rows (quantity>0, price>0, non-cancelled invoice,
stock code matching ^\\d{5}[A-Z]*$) strictly before A.

  reference window  W  = [A-90d, A)
  candidate pool       = items with invoice-support >= 8 in W
  co-purchase affinity = cosine similarity over W invoices:
                         cos(i,j) = #inv(i&j) / sqrt(#inv(i) * #inv(j))
  history H_c          = the <=40 most recently purchased distinct items in
                         [A-365d, A), item j weighted w_j = 0.5^(age_j/60d)
                         (age_j = days since last purchase of j)
  price affinity       = exp(-0.5 * ((ln p_i - ln m_c)/0.7)^2), where p_i is
                         the item's median unit price in [A-365d, A) and m_c
                         the customer's median paid unit price in the lookback
  popularity tilt      = 1 + 0.05 * ln(1 + support_W(i))
  exclusion            = items bought by c within [A-90d, A) are removed
  score(i)             = [sum_j w_j * cos(i,j)] * price_aff(i) * pop_tilt(i)
  diversity re-rank    = the slate is built GREEDILY (MMR style): each next
                         position takes argmax of score(i) - 0.9 * max cosine
                         between i and the items already selected (ties by
                         support desc, stock_code asc). Production slates
                         avoid near-duplicate variants; this set-level step is
                         also what makes the policy hard to imitate with a
                         pointwise ranker.
  healthy slate        = 10 greedy picks from the positive-score candidates;
                         customers with <10 positive-score candidates are
                         dropped from the slate universe.

The three failure modes (exactly one per corrupted slate)
---------------------------------------------------------
  popularity_fallback : personalization dies; the slate is the top-10 of W
                        by (support desc, stock_code asc) — no history, no
                        price kernel, and NO recent-purchase exclusion (the
                        fallback may re-recommend items just bought).
  price_band_shift    : an upsell misconfiguration multiplies the customer's
                        price anchor m_c by a per-request factor drawn
                        deterministically from {1.9, 2.6, 3.3}; everything
                        else (including the diversity re-rank) unchanged.
  stale_index         : the service reads a stale index: support, cosine and
                        candidate pool are computed on W_stale=[A-270d,A-180d)
                        (wrong season), history/price/exclusions unchanged.

If a bug slate happens to equal the healthy slate item-for-item, the request
is relabeled clean (an in-principle-undetectable corruption must not exist).

Anchors, split and anti-leakage measures
----------------------------------------
  anchors (train)     : 2010-09-01, 2010-12-01, 2011-03-01, 2011-06-01,
                        2011-09-01, 2011-11-15
  anchor  (test)      : 2011-11-15 only
  customer split      : the eligible-customer universe is split 55% train /
                        45% test by seeded permutation; a customer appears on
                        one side only. Train slates: all (anchor, customer)
                        pairs for train customers. Test slates: test customers
                        at the final anchor only.
  transaction cutoff  : public transactions.csv ends strictly before the test
                        anchor, so NO post-anchor information exists for any
                        test slate (no predict-the-future shortcut).
  opaque slate ids    : ids S00001.. are assigned by a seeded permutation over
                        the union of train+test slates, so id order carries no
                        information about split, anchor, customer or label.
  file ordering       : every output is sorted by (slate_id, position) or by
                        natural transaction order — never by label.
  eligibility         : >=3 invoices, >=10 distinct items in [A-365d, A),
                        last purchase within [A-150d, A).

Outputs
-------
  dataset/public/transactions.csv        real log, truncated at test anchor
  dataset/public/catalog.csv             real per-item stats (pre-anchor only)
  dataset/public/slates_train.csv        emitted train slates
  dataset/public/slate_labels_train.csv  train flags + failure modes
  dataset/public/healthy_slates_train.csv healthy reference for ALL train slates
  dataset/public/slates_test.csv         emitted test slates (unlabeled)
  dataset/public/sample_submission.csv   format example (copy-emitted, all clean)
  dataset/private/answers.csv            test flag+mode+healthy+emitted slates

Deterministic: a fixed seed drives every random choice; rerunning produces
byte-identical outputs (verified in the build log).

Usage:
    python prepare.py [--raw data/raw] [--public dataset/public]
                      [--private dataset/private]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

SEED = 20260705
RNG_SPLIT, RNG_CORRUPT, RNG_IDS, RNG_PRICE = 11, 23, 37, 41  # stream tags

TRAIN_ANCHORS = ["2010-09-01", "2010-12-01", "2011-03-01",
                 "2011-06-01", "2011-09-01", "2011-11-15"]
TEST_ANCHOR = "2011-11-15"

TOP_K = 10
LOOKBACK_D, WINDOW_D = 365, 90
STALE_LO_D, STALE_HI_D = 270, 180          # W_stale = [A-270d, A-180d)
MIN_SUPPORT = 8
HIST_MAX_ITEMS, HIST_HALFLIFE_D = 40, 60.0
PRICE_BW, POP_TILT = 0.7, 0.05
LAMBDA_DIV = 0.9                           # MMR diversity strength
PRICE_MULTS = (1.9, 2.6, 3.3)              # price_band_shift factors
EXCLUDE_D = 90
ELIG_MIN_INV, ELIG_MIN_ITEMS, ELIG_RECENCY_D = 3, 10, 150
CORRUPT_P = 0.35
MODES = ["popularity_fallback", "price_band_shift", "stale_index"]
TRAIN_FRAC = 0.55


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_raw(raw: Path) -> pd.DataFrame:
    """Load the raw workbook. The canonical platform upload is the CSV export
    (identical rows/columns to the official xlsx, produced by
    download_data.py); the xlsx is accepted as a fallback so the pipeline
    also runs directly on the original UCI archive."""
    # Locate the raw file wherever the platform placed it (uploads may be
    # nested); prefer the canonical CSV names, then any single CSV, then xlsx.
    def find(patterns: list[str]) -> Path | None:
        for pat in patterns:
            hits = sorted(raw.rglob(pat))
            if hits:
                return hits[0]
        return None

    csv_path = find(["retail_transactions_2009_2011.csv",
                     "online_retail_ii.csv"])
    if csv_path is None:
        all_csv = [p for p in sorted(raw.rglob("*.csv"))
                   if "sample" not in p.name.lower()]
        if len(all_csv) == 1:
            csv_path = all_csv[0]
    if csv_path is not None:
        df = pd.read_csv(csv_path,
                         dtype={"Invoice": str, "StockCode": str,
                                "Description": str, "Country": str},
                         parse_dates=["InvoiceDate"])
    else:
        xlsx_path = find(["online_retail_ii.xlsx", "*.xlsx"])
        if xlsx_path is None:
            listing = "\n  ".join(str(p.relative_to(raw))
                                  for p in sorted(raw.rglob("*"))[:50])
            raise FileNotFoundError(
                "Could not find retail_transactions_2009_2011.csv (or the "
                "online_retail_ii xlsx workbook) anywhere under "
                f"{raw}. Contents:\n  {listing}")
        xl = pd.ExcelFile(xlsx_path)
        frames = [pd.read_excel(xl, sheet_name=s) for s in xl.sheet_names]
        df = pd.concat(frames, ignore_index=True)
    df.columns = ["invoice", "stock_code", "description", "quantity",
                  "invoice_date", "unit_price", "customer_id", "country"]
    df["invoice"] = df["invoice"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df["description"] = df["description"].astype(str).str.strip()
    df.loc[df["description"].isin(["nan", "None"]), "description"] = ""
    # customer ids as clean strings ("13085"), empty when unknown
    cid = pd.to_numeric(df["customer_id"], errors="coerce")
    df["customer_id"] = cid.astype("Int64").astype(str).replace("<NA>", "")
    df = df.sort_values(["invoice_date", "invoice", "stock_code"],
                        kind="mergesort").reset_index(drop=True)
    return df


def policy_valid(df: pd.DataFrame) -> pd.DataFrame:
    m = ((df["quantity"] > 0) & (df["unit_price"] > 0)
         & ~df["invoice"].str.startswith("C")
         & df["stock_code"].str.match(r"^\d{5}[A-Z]*$"))
    return df.loc[m]


# --------------------------------------------------------------------------
# per-window statistics
# --------------------------------------------------------------------------
class WindowStats:
    """Invoice-support and item-item cosine co-occurrence for one window."""

    def __init__(self, rows: pd.DataFrame):
        pairs = rows[["invoice", "stock_code"]].drop_duplicates()
        self.items = np.array(sorted(pairs["stock_code"].unique()))
        self.index = {c: i for i, c in enumerate(self.items)}
        inv_codes, inv_idx = pd.factorize(pairs["invoice"], sort=True)
        item_idx = pairs["stock_code"].map(self.index).to_numpy()
        B = sparse.csr_matrix(
            (np.ones(len(pairs), dtype=np.float32), (inv_codes, item_idx)),
            shape=(len(inv_idx), len(self.items)))
        self.support = np.asarray(B.sum(axis=0)).ravel()          # invoices/item
        C = (B.T @ B).toarray().astype(np.float32)
        np.fill_diagonal(C, 0.0)
        d = np.sqrt(np.maximum(self.support, 1.0))
        self.cosine = C / d[None, :] / d[:, None]                 # dense, ~item² floats

    def candidates(self, min_support: int) -> np.ndarray:
        return np.where(self.support >= min_support)[0]


def greedy_diverse_k(scores: np.ndarray, support: np.ndarray,
                     items: np.ndarray, idx: np.ndarray,
                     cosine: np.ndarray, k: int) -> list[str] | None:
    """MMR-style greedy slate: each pick maximizes
    score(i) - LAMBDA_DIV * max_cos(i, selected), ties broken by
    (support desc, stock_code asc). Requires k positive-score candidates."""
    pool = idx[scores[idx] > 0]
    if len(pool) < k:
        return None
    max_cos = np.zeros(len(pool), dtype=np.float64)
    taken = np.zeros(len(pool), dtype=bool)
    picks: list[int] = []
    for _ in range(k):
        adj = scores[pool] - LAMBDA_DIV * max_cos
        adj[taken] = -np.inf
        j = np.lexsort((items[pool], -support[pool], -adj))[0]
        picks.append(pool[j])
        taken[j] = True
        max_cos = np.maximum(max_cos, cosine[pool[j], pool].astype(np.float64))
    return items[np.array(picks)].tolist()


# --------------------------------------------------------------------------
# the hidden policy and its failure modes
# --------------------------------------------------------------------------
def slates_for_anchor(valid: pd.DataFrame, anchor: pd.Timestamp) -> pd.DataFrame:
    """Emit healthy + all three bug slates for every eligible customer."""
    look = valid[(valid["invoice_date"] < anchor)
                 & (valid["invoice_date"] >= anchor - pd.Timedelta(days=LOOKBACK_D))
                 & (valid["customer_id"] != "")]
    fresh = WindowStats(look[look["invoice_date"]
                             >= anchor - pd.Timedelta(days=WINDOW_D)])
    stale_rows = valid[(valid["invoice_date"] < anchor - pd.Timedelta(days=STALE_HI_D))
                       & (valid["invoice_date"] >= anchor - pd.Timedelta(days=STALE_LO_D))]
    stale = WindowStats(stale_rows)
    cand_f = fresh.candidates(MIN_SUPPORT)
    cand_s = stale.candidates(MIN_SUPPORT)

    # median unit price per item over the lookback (price affinity input)
    item_price = look.groupby("stock_code")["unit_price"].median()

    # eligibility
    g = look.groupby("customer_id").agg(n_inv=("invoice", "nunique"),
                                        n_item=("stock_code", "nunique"),
                                        last_ts=("invoice_date", "max"),
                                        m_price=("unit_price", "median"))
    elig = g[(g["n_inv"] >= ELIG_MIN_INV) & (g["n_item"] >= ELIG_MIN_ITEMS)
             & (g["last_ts"] >= anchor - pd.Timedelta(days=ELIG_RECENCY_D))]

    # per-item last-purchase recency per customer (history weights)
    lastbuy = (look.groupby(["customer_id", "stock_code"])["invoice_date"]
               .max().reset_index())
    lastbuy["age_d"] = (anchor - lastbuy["invoice_date"]).dt.total_seconds() / 86400.0
    lastbuy["w"] = 0.5 ** (lastbuy["age_d"] / HIST_HALFLIFE_D)
    hist_by_cust = dict(tuple(lastbuy.groupby("customer_id")))

    def price_kernel(stats: WindowStats, m_c: float) -> np.ndarray:
        p = item_price.reindex(stats.items).to_numpy(dtype=np.float64)
        p = np.where(np.isfinite(p) & (p > 0), p, np.nan)
        z = (np.log(p) - np.log(m_c)) / PRICE_BW
        k = np.exp(-0.5 * z * z)
        return np.where(np.isfinite(k), k, 0.0)

    def personalized(stats: WindowStats, cand: np.ndarray, hist: pd.DataFrame,
                     m_c: float, exclude: set[str]) -> list[str] | None:
        # the <=40 most recently purchased distinct items (ties by code)
        h = hist.sort_values(["age_d", "stock_code"],
                             kind="mergesort").head(HIST_MAX_ITEMS)
        cols, w = [], []
        for code, wt in zip(h["stock_code"], h["w"]):
            j = stats.index.get(code)
            if j is not None:
                cols.append(j); w.append(wt)
        if not cols:
            return None
        base = stats.cosine[:, cols] @ np.asarray(w, dtype=np.float32)
        score = (base.astype(np.float64) * price_kernel(stats, m_c)
                 * (1.0 + POP_TILT * np.log1p(stats.support)))
        keep = cand[[stats.items[i] not in exclude for i in cand]]
        return greedy_diverse_k(score, stats.support, stats.items, keep,
                                stats.cosine, TOP_K)

    # popularity fallback slate is identical for every customer at this anchor
    order = np.lexsort((fresh.items[cand_f], -fresh.support[cand_f]))
    fallback_slate = fresh.items[cand_f[order[:TOP_K]]].tolist()

    recent_cut = anchor - pd.Timedelta(days=EXCLUDE_D)
    recent = look[look["invoice_date"] >= recent_cut]
    recent_by_cust = recent.groupby("customer_id")["stock_code"].agg(set).to_dict()

    # deterministic per-customer price-shift factor for this anchor
    elig_sorted = sorted(elig.index)
    r_price = np.random.default_rng([SEED, RNG_PRICE, int(anchor.value)])
    mult_of = dict(zip(elig_sorted,
                       r_price.choice(PRICE_MULTS, size=len(elig_sorted))))

    out = []
    for cust in elig_sorted:
        hist = hist_by_cust.get(cust)
        if hist is None:
            continue
        m_c = float(elig.loc[cust, "m_price"])
        excl = recent_by_cust.get(cust, set())
        healthy = personalized(fresh, cand_f, hist, m_c, excl)
        if healthy is None:
            continue
        price_bug = personalized(fresh, cand_f, hist, m_c * mult_of[cust], excl)
        stale_bug = personalized(stale, cand_s, hist, m_c, excl)
        if price_bug is None or stale_bug is None:
            continue
        out.append({"customer_id": cust, "anchor": anchor,
                    "healthy": healthy,
                    "popularity_fallback": fallback_slate,
                    "price_band_shift": price_bug,
                    "stale_index": stale_bug})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def prepare(raw: Path, public: Path, private: Path) -> None:
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    df = load_raw(raw)
    valid = policy_valid(df)
    test_anchor = pd.Timestamp(TEST_ANCHOR)

    # ---- hidden policy replay at every anchor -----------------------------
    per_anchor = []
    for a_str in TRAIN_ANCHORS:
        sl = slates_for_anchor(valid, pd.Timestamp(a_str))
        per_anchor.append(sl)
        print(f"[anchor {a_str}] slates emitted: {len(sl)}")
    slates = pd.concat(per_anchor, ignore_index=True)

    # ---- customer-disjoint split ------------------------------------------
    universe = np.array(sorted(slates["customer_id"].unique()))
    rng = np.random.default_rng([SEED, RNG_SPLIT])
    perm = rng.permutation(len(universe))
    n_train = int(round(TRAIN_FRAC * len(universe)))
    train_cust = set(universe[perm[:n_train]])
    test_cust = set(universe[perm[n_train:]])

    train = slates[slates["customer_id"].isin(train_cust)].copy()
    test = slates[(slates["customer_id"].isin(test_cust))
                  & (slates["anchor"] == test_anchor)].copy()

    # ---- corruption assignment --------------------------------------------
    def assign(frame: pd.DataFrame, tag: int) -> pd.DataFrame:
        frame = frame.sort_values(["anchor", "customer_id"],
                                  kind="mergesort").reset_index(drop=True)
        r = np.random.default_rng([SEED, RNG_CORRUPT, tag])
        corrupted = r.random(len(frame)) < CORRUPT_P
        mode_pick = r.integers(0, len(MODES), size=len(frame))
        modes, emitted = [], []
        for i, row in frame.iterrows():
            if corrupted[i]:
                mode = MODES[mode_pick[i]]
                slate = row[mode]
                if slate == row["healthy"]:      # undetectable in principle
                    mode, slate = "none", row["healthy"]
            else:
                mode, slate = "none", row["healthy"]
            modes.append(mode)
            emitted.append(slate)
        frame["mode"] = modes
        frame["corrupted"] = (frame["mode"] != "none").astype(int)
        frame["emitted"] = emitted
        return frame

    train = assign(train, 1)
    test = assign(test, 2)

    # ---- opaque ids over the union ----------------------------------------
    # Ids are assigned by a seeded permutation, and the permutation is
    # deterministically rejection-sampled (sub-seed k = 0, 1, 2, ...) until id
    # order is measurably uncorrelated with the hidden labels on BOTH splits —
    # so even an unlucky draw can never leak a usable id-order signal.
    both = pd.concat([train.assign(_split="train"), test.assign(_split="test")],
                     ignore_index=True)

    def id_label_corr(frame: pd.DataFrame) -> float:
        idnum = frame["slate_id"].str[1:].astype(int).to_numpy(dtype=float)
        flag = frame["corrupted"].to_numpy(dtype=float)
        if flag.std() == 0 or idnum.std() == 0:
            return 0.0
        return float(np.corrcoef(idnum, flag)[0, 1])

    for k in range(1000):
        r_ids = np.random.default_rng([SEED, RNG_IDS, k])
        cand = both.iloc[r_ids.permutation(len(both))].reset_index(drop=True)
        cand["slate_id"] = [f"S{i + 1:05d}" for i in range(len(cand))]
        r_tr = id_label_corr(cand[cand["_split"] == "train"])
        r_te = id_label_corr(cand[cand["_split"] == "test"])
        if abs(r_tr) < 0.02 and abs(r_te) < 0.04:
            both = cand
            print(f"[ids] permutation sub-seed k={k} accepted "
                  f"(|corr| train={r_tr:+.4f}, test={r_te:+.4f})")
            break
    else:  # pragma: no cover — statistically unreachable
        raise RuntimeError("no id permutation passed the independence gate")
    train = both[both["_split"] == "train"].copy()
    test = both[both["_split"] == "test"].copy()

    # ---- public: real transaction log & catalog (pre-anchor only) ---------
    pub_tx = df[df["invoice_date"] < test_anchor]
    pub_tx.to_csv(public / "transactions.csv", index=False,
                  date_format="%Y-%m-%d %H:%M:%S")

    v_pub = policy_valid(pub_tx)
    catalog = (v_pub.groupby("stock_code")
               .agg(description=("description",
                                 lambda s: s.value_counts().index[0]),
                    median_unit_price=("unit_price", "median"),
                    n_invoices=("invoice", "nunique"),
                    first_seen=("invoice_date", "min"),
                    last_seen=("invoice_date", "max"))
               .reset_index()
               .sort_values("stock_code", kind="mergesort"))
    catalog["first_seen"] = catalog["first_seen"].dt.strftime("%Y-%m-%d")
    catalog["last_seen"] = catalog["last_seen"].dt.strftime("%Y-%m-%d")
    catalog.to_csv(public / "catalog.csv", index=False)

    # ---- public/private slate files ----------------------------------------
    def long_format(frame: pd.DataFrame, col: str,
                    keep_meta: bool) -> pd.DataFrame:
        rows = []
        for _, row in frame.iterrows():
            for pos, code in enumerate(row[col], start=1):
                rec = {"slate_id": row["slate_id"], "position": pos,
                       "stock_code": code}
                if keep_meta:
                    rec.update(customer_id=row["customer_id"],
                               anchor_date=row["anchor"].strftime("%Y-%m-%d"))
                rows.append(rec)
        out = pd.DataFrame(rows).sort_values(["slate_id", "position"],
                                             kind="mergesort")
        cols = (["slate_id", "customer_id", "anchor_date", "position",
                 "stock_code"] if keep_meta
                else ["slate_id", "position", "stock_code"])
        return out[cols].reset_index(drop=True)

    long_format(train, "emitted", True).to_csv(
        public / "slates_train.csv", index=False)
    long_format(train, "healthy", False).to_csv(
        public / "healthy_slates_train.csv", index=False)
    (train[["slate_id", "corrupted", "mode"]]
     .sort_values("slate_id", kind="mergesort")
     .to_csv(public / "slate_labels_train.csv", index=False))

    long_format(test, "emitted", True).to_csv(
        public / "slates_test.csv", index=False)

    sample = long_format(test, "emitted", False)
    sample.insert(1, "corrupted", 0)
    sample.insert(2, "mode", "none")
    sample.to_csv(public / "sample_submission.csv", index=False)

    healthy_l = long_format(test, "healthy", False).rename(
        columns={"stock_code": "healthy_code"})
    emitted_l = long_format(test, "emitted", False).rename(
        columns={"stock_code": "emitted_code"})
    answers = healthy_l.merge(emitted_l, on=["slate_id", "position"])
    answers = answers.merge(test[["slate_id", "corrupted", "mode"]], on="slate_id")
    answers = answers[["slate_id", "corrupted", "mode", "position",
                       "healthy_code", "emitted_code"]]
    answers.to_csv(private / "answers.csv", index=False)

    # ---- leakage guards -----------------------------------------------------
    assert not (set(train["slate_id"]) & set(test["slate_id"]))
    assert not (set(train["customer_id"]) & set(test["customer_id"]))
    assert set(answers["slate_id"]) == set(test["slate_id"])
    test_cols = pd.read_csv(public / "slates_test.csv", nrows=0).columns
    assert "healthy_code" not in test_cols
    assert "mode" not in test_cols and "corrupted" not in test_cols
    assert (pub_tx["invoice_date"] < test_anchor).all()

    # ---- diagnostics ---------------------------------------------------------
    def overlap(a: list[str], b: list[str]) -> float:
        return len(set(a) & set(b)) / TOP_K

    print("\n=== diagnostics ===")
    for split_name, frame in [("train", train), ("test", test)]:
        n = len(frame)
        print(f"{split_name}: {n} slates | corrupted "
              f"{frame['corrupted'].mean():.3f}")
        for mode in ["none"] + MODES:
            sub = frame[frame["mode"] == mode]
            if not len(sub):
                continue
            ov = np.mean([overlap(r["emitted"], r["healthy"])
                          for _, r in sub.iterrows()])
            print(f"  {mode:20s} n={len(sub):5d}  "
                  f"mean emitted∩healthy overlap={ov:.3f}")
    print(f"public transactions rows: {len(pub_tx):,} "
          f"(cutoff {TEST_ANCHOR}); catalog items: {len(catalog):,}")


def main() -> None:
    here = Path(__file__).resolve().parent
    # canonical Eris layout is dataset/raw; the build repo also keeps a copy
    # under data/raw — accept whichever exists
    default_raw = next((p for p in (here / "dataset" / "raw",
                                    here / "data" / "raw") if p.exists()),
                       here / "dataset" / "raw")
    parser = argparse.ArgumentParser(
        description="Build Slate Forensics public/private data.")
    parser.add_argument("--raw", default=default_raw, type=Path)
    parser.add_argument("--public", default=here / "dataset" / "public", type=Path)
    parser.add_argument("--private", default=here / "dataset" / "private", type=Path)
    args = parser.parse_args()
    prepare(args.raw, args.public, args.private)


if __name__ == "__main__":
    main()
