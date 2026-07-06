# Dataset Description

## Overview

The substrate is the **genuine Online Retail II dataset**: 1,067,371 real
transactions of a UK-based, registered non-store online retailer selling
all-occasion giftware, 2009-12-01 to 2011-12-09. Every transaction field a
solver sees — invoice ids, stock codes, product descriptions, quantities,
timestamps, unit prices, customer ids, countries — is verbatim from the
source.

On top of the real log, `prepare.py` adds a **deterministic synthetic
overlay** that turns it into a forensic challenge: a hidden "You may also
like" recommendation policy is replayed at fixed anchor dates against each
eligible customer's real purchase history, and a seeded minority of the
emitted 10-item slates is corrupted by one of three realistic production
failure modes (`popularity_fallback`, `price_band_shift`, `stale_index`).
Solvers must audit the emitted slates, diagnose the failure mode, and
reconstruct the healthy slate. The skill tested is policy imitation from
logs, failure-signature modeling, and multi-output consistency — not
next-purchase prediction.

## Source and License

- Genuine source dataset: **Online Retail II**
- Source URL: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Direct download: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
- License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
- License URL: https://creativecommons.org/licenses/by/4.0/
- Commercial use allowed: **yes** (CC BY 4.0)
- Redistribution allowed: **yes, with attribution** (CC BY 4.0)
- DOI / accession: **10.24432/C5CG6D** (UCI dataset id 502)
- Citation: *Chen, D. (2012). Online Retail II [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C5CG6D.*
- Access date: **2026-07-05** (license string and DOI confirmed live from the
  UCI page/API at download time; see `data/source_metadata.json` and
  `data/raw/uci_api_metadata.json` for the recorded evidence)

Machine-readable provenance, including SHA-256 checksums of the downloaded
files, is written by `download_data.py` to `data/source_metadata.json`.

## File Structure

```text
data/
  raw/                          # created by download_data.py
    online_retail_ii.zip        # official UCI archive (45.6 MB)
    online_retail_ii.xlsx       # extracted workbook (2 sheets)
    retail_transactions_2009_2011.csv  # lossless CSV export of the workbook —
                                #   canonical raw file read by prepare.py
                                #   (byte-identical outputs vs the xlsx path)
    uci_api_metadata.json       # live UCI metadata (license/DOI evidence)
    checksums.txt               # sha256 of raw files
    source_license.txt          # license + attribution statement
  source_metadata.json          # machine-readable provenance

dataset/
  public/
    transactions.csv            # real log, truncated at the test anchor
    catalog.csv                 # real per-item reference statistics
    slates_train.csv            # emitted train slates (labeled requests)
    slate_labels_train.csv      # corrupted flag + failure mode per request
    healthy_slates_train.csv    # healthy slate for every train request
    slates_test.csv             # emitted evaluation slates (unlabeled)
    sample_submission.csv       # required format (copy-emitted, all clean)
  private/
    answers.csv                 # flag, mode, healthy + emitted slates
```

## Raw data description

One xlsx workbook with two sheets (`Year 2009-2010`, `Year 2010-2011`),
identical schema: Invoice, StockCode, Description, Quantity, InvoiceDate,
Price, Customer ID, Country. 1,067,371 rows, 5,852 identified customers,
~4,700 catalogued products.

## Processed data description

| Column | Type | File(s) | Description |
| --- | --- | --- | --- |
| invoice | str | transactions.csv | Invoice number; a leading `C` marks a cancellation |
| stock_code | str | transactions, catalog, slates | Item code; service rows (postage, manuals…) do not match `^\d{5}[A-Z]*$` |
| description | str | transactions, catalog | Product description (raw, occasionally blank) |
| quantity | int | transactions.csv | Units per row; negative for cancellations |
| invoice_date | datetime | transactions.csv | Transaction timestamp |
| unit_price | float | transactions.csv | Sterling unit price; occasionally 0 |
| customer_id | str | transactions, slates | Customer key; empty on ~23% of transaction rows |
| country | str | transactions.csv | Customer country (mostly United Kingdom) |
| median_unit_price / n_invoices / first_seen / last_seen | mixed | catalog.csv | Real per-item statistics computed strictly before the test anchor |
| slate_id | str | slate files | Opaque request id (independence of id order from labels is verified at build time) |
| anchor_date | date | slates_train/test | When the request was served |
| position | int | slate files | Rank 1..10 |
| corrupted | int | slate_labels_train | 1 if a broken code path served the request |
| mode | str | slate_labels_train | `none`, `popularity_fallback`, `price_band_shift`, `stale_index` |

Sizes: 993,283 public transaction rows (cutoff at the evaluation anchor,
2011-11-15); 4,686 catalog items; 5,244 train requests at six anchor dates
(52,440 slate rows, 35.5% corrupted); 757 evaluation requests at the final
anchor (7,570 rows); `answers.csv` additionally stores the emitted slate so
the grader's consistency checks are self-contained.

## Transformation summary

All transformations live in `prepare.py` (single seed 20260705) and are
deterministic — rerunning produces byte-identical files:

1. Clean typing/normalisation of the raw workbook (codes uppercased, customer
   ids stringified; every retained value is genuine).
2. A hidden healthy recommendation policy (personalised, history-, price- and
   popularity-aware, with a set-level diversity re-rank) is replayed at six
   anchor dates for every eligible customer, producing healthy 10-item slates.
   Its exact form and parameters are intentionally undisclosed here — learning
   them from the labeled slates is the challenge.
3. A seeded ~35% of requests is corrupted by exactly one of the three failure
   modes (parameters likewise undisclosed; one mode varies its strength per
   request). If a corrupted slate coincides with the healthy one it is
   relabeled clean, so no in-principle-undetectable corruption exists.
4. Anti-leakage measures: customer-disjoint train/test split; all evaluation
   requests at the final anchor; public transaction log truncated at that
   anchor (no post-anchor information exists); opaque slate ids assigned by a
   permutation that is rejection-sampled until id order is measurably
   uncorrelated with the hidden labels on both splits; all files sorted by
   keys, never by labels.

## Known quirks

- Cancellations (`C…` invoices, negative quantities) and ~23% missing
  `customer_id` are genuine properties of the source data and are preserved.
- Demand is strongly seasonal (Christmas peak) — relevant to one failure mode.
- A handful of stock codes are services (POST, DOT, M…), not products.
- The same (invoice, stock_code) pair can appear on multiple rows (real
  duplicates in the source system).
- Train anchors span five quarters; the transaction file extends past the
  earlier anchors, so features built for train requests must respect each
  request's `anchor_date` (see rubrics).

## Reproducibility

```bash
python download_data.py    # fetch + verify the genuine UCI source (network)
python prepare.py          # deterministic rebuild of public/private files
python probe/test_grade.py # grader unit tests
python probe/probe.py      # naive-baseline floor checks
```

## Redistribution notes

CC BY 4.0 permits adaptation and redistribution (including commercial use)
with attribution. This package redistributes a cleaned/truncated form of the
transactions plus derived synthetic slate files, with attribution given above
and in `data/raw/source_license.txt`. No LLM-generated content is included
anywhere in the data or the reference solution.

## Limitations and assumptions

- The recommendation slates are a synthetic overlay: the retailer did not
  actually run this recommender. The overlay is documented as synthetic and
  never presented as historical fact.
- Ground-truth "healthy" behaviour is defined by the hidden policy itself,
  so repair is well-posed by construction (every answer is exactly
  reproducible from public data plus the hidden policy).
- Customer ids are the original pseudonymous keys from the UCI release; no
  re-identification is attempted or enabled beyond the public source.
