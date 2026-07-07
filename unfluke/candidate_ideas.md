# Candidate Screening — Crypto-Themed Eris Task

Fifteen candidate concepts were generated. Every candidate is grounded in a
**genuine public dataset** whose landing page and license were verified by
direct HTTP fetch on 2026-07-07 (no URL, DOI, or license below is invented).
Candidates were screened against the Eris gates: clear license, novelty,
resistance to rule-based / plain gradient-boosting shortcuts, external
reconstruction risk, and feasibility of deterministic preparation + grading.

**Selected candidate: #1 — Unfluke (skill-vs-luck forensics for systematic
trading records).**

---

## 1. Unfluke: Skill-vs-Luck Forensics for Systematic Trading Records — SELECTED

- **Open domain:** From Scratch (sequence representation learning and
  evaluation modeling built entirely from the provided files; no pretrained
  models applicable).
- **Source dataset:** Binance cryptocurrencies historical daily data
  (S. Kanyhin), Zenodo.
- **Source URL:** https://zenodo.org/records/8187872
- **License:** CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/
- **DOI:** 10.5281/zenodo.8187872
- **Proposed transformation:** Deterministically simulate thousands of
  parameterized rule-based trading strategies on real daily OHLCV segments
  ("arenas"). A hidden subset of strategies carries a small persistent,
  regime-conditional edge; the rest are pure-noise rules kept only when
  their in-sample Sharpe *by luck* matches the skilled Sharpe distribution
  (rejection-sampled, so marginal summary statistics are uninformative).
  Public files expose in-sample trade logs + normalized arena bars; private
  answers hold the skill flag and realized out-of-sample Sharpe computed on
  hidden continuation windows.
- **Hidden structure to infer:** temporal *consistency* and market-state
  *conditionality* of per-trade edge — skill is spread evenly across time
  and concentrated in specific volatility states; luck clusters in streaks
  and single regimes.
- **Heterogeneous targets:** skill probability (detection), out-of-sample
  Sharpe (regression), fixed-size portfolio selection (decision), with
  cross-field logical consistency rules.
- **Expected novelty:** 5/5 — backtest-overfitting forensics (deflated
  Sharpe, multiple-testing luck) is a real research problem that has no
  public leaderboard equivalent; the evaluation object is *the strategy*,
  not the price series.
- **Likely difficulty:** hard — summary statistics are deliberately
  uninformative; solvers must build trade-sequence representations.
- **Major rejection risks & mitigations:** (a) plain GBM on naive stats —
  neutralized by Sharpe-distribution matching; (b) external reconstruction —
  strategies' parameters and rules are never disclosed, so even a solver who
  identifies the underlying assets cannot simulate the hidden out-of-sample
  trades; arenas are anonymized and per-window normalized; (c) grader
  complexity — controlled by exact, documented component formulas.

## 2. Manufactured-Move Detection (injected pump localization)

- **Domain:** From Scratch. **Source:** same Zenodo 8187872 (CC BY 4.0), or
  market data of digital asset exchange (binance, kucoin, upbit), Zenodo,
  https://zenodo.org/records/19091779, CC BY 4.0, DOI 10.5281/zenodo.19091779.
- **Transformation:** inject deterministic pump-shaped price/volume episodes
  into a subset of real windows; detect, localize onset, estimate magnitude.
- **Hidden structure:** injected price-volume coupling signature.
- **Novelty:** 4/5. **Difficulty:** medium-hard.
- **Rejected:** clean (non-injected) windows remain byte-derivable from
  public exchange history, so a determined solver could unmask labels by
  matching windows against external archives — weaker leakage story than #1.

## 3. Ransomware Wallet Campaign Forensics

- **Domain:** From Scratch. **Source:** BitcoinHeist Ransomware Address
  Dataset, UCI ML Repository, https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset,
  CC BY 4.0, DOI 10.24432/C5BG8V (2.9M rows, verified download HTTP 200).
- **Transformation:** temporal campaign-linking and family prediction.
- **Novelty:** 2/5. **Rejected:** widely mirrored on Kaggle with public
  notebooks (benchmark-clone risk); core task collapses to tabular
  classification solvable by gradient boosting.

## 4. Cryptocurrency Ecosystem Survival Dynamics

- **Domain:** From Scratch. **Source:** Data from "Evolutionary dynamics of
  the cryptocurrency market" (ElBahrawy et al.), Zenodo/Dryad,
  https://zenodo.org/records/4932549, CC0 1.0, DOI 10.5061/dryad.38776.
- **Transformation:** predict coin survival and future market-share quantiles
  from weekly share trajectories (1,469 coins, 2013–2017).
- **Novelty:** 3/5. **Rejected:** weekly aggregate shares only — reduces to
  low-dimensional tabular survival prediction; data ends in 2017.

## 5. Cross-Exchange Arbitrage Window Prediction

- **Domain:** From Scratch. **Source:** market data of digital asset exchange
  (binance, kucoin, upbit), Zenodo, https://zenodo.org/records/19091779,
  CC BY 4.0, DOI 10.5281/zenodo.19091779 (~1.5 GB across three archives).
- **Transformation:** predict cross-exchange spread-window openings.
- **Novelty:** 3/5. **Rejected:** 1.5 GB raw download is hostile to
  reproducible preparation; task shape is standard short-horizon forecasting.

## 6. Multi-Path Arbitrage Consistency Modeling

- **Domain:** From Scratch. **Source:** XRP/ETH Multi-Path Arbitrage Dataset
  (Binance Exchange, 2025), Zenodo, https://zenodo.org/records/18133862,
  CC BY 4.0, DOI 10.5281/zenodo.18133862.
- **Transformation:** model no-arbitrage consistency across XRP/USDT,
  ETH/USDT, XRP/ETH and predict deviation episodes.
- **Novelty:** 3/5. **Rejected:** single currency triple — narrow evaluation
  set; deviations largely detectable by a fixed algebraic rule.

## 7. Crypto-Influencer Authorship & Market-Impact Attribution

- **Domain:** NLP. **Source:** Pre-processed Tweets by verified users, Elon
  Musk, Vitalik Buterin and CZ Binance, Zenodo,
  https://zenodo.org/records/5336611, CC BY 4.0, DOI 10.5281/zenodo.5336611.
- **Transformation:** authorship attribution + market-window alignment.
- **Novelty:** 3/5. **Rejected:** tweets of public figures are trivially
  searchable online — labels reconstructable externally; platform-ToS
  ambiguity for redistributed tweet text.

## 8. Sentiment–Price Divergence Regimes

- **Domain:** NLP. **Source:** Bitcoin Sentiment & Market Data (Merged:
  Kaggle + Binance), Zenodo, https://zenodo.org/records/17380941, CC BY 4.0,
  DOI 10.5281/zenodo.17380941.
- **Transformation:** predict divergence regimes between hourly sentiment and
  price. **Novelty:** 3/5. **Rejected:** provenance chain runs through an
  unnamed Kaggle corpus — upstream license cannot be confirmed.

## 9. ICO Outcome Reconstruction

- **Domain:** NLP/tabular. **Source:** Initial Coin Offering database,
  Zenodo, https://zenodo.org/records/4034258, CC BY 4.0,
  DOI 10.5281/zenodo.4034258.
- **Transformation:** predict ICO funding outcome from offering attributes.
- **Novelty:** 2/5. **Rejected:** one small spreadsheet; plain tabular
  prediction, immediately GBM-solvable.

## 10. On-Chain Metric Intervention Forecasting

- **Domain:** Sequence-to-Sequence. **Source:** Bitcoin Dataset without
  Missing Values, Zenodo, https://zenodo.org/records/5122101, CC BY 4.0,
  DOI 10.5281/zenodo.5122101 (18 daily on-chain/attention series).
- **Transformation:** forecast masked spans of on-chain series under hidden
  intervention masks. **Novelty:** 3/5. **Rejected:** single asset, daily
  granularity, small (~2k rows/series); upstream scraped from bitinfocharts
  with LOCF imputation — weak raw material for a hard benchmark.

## 11. Crypto/NFT Dataset-Catalog RAG

- **Domain:** RAG. **Source:** Dataset of cryptocurrency & NFT datasets,
  Zenodo, https://zenodo.org/records/6967048, CC BY 4.0,
  DOI 10.5281/zenodo.6967048.
- **Transformation:** retrieval-grounded question answering over a corpus of
  dataset documentation. **Novelty:** 3/5. **Rejected:** authoring the Q/A
  supervision at scale would require LLM generation, which the Eris rules
  forbid inside the dataset; hand-authoring enough items is infeasible.

## 12. Event-Window Microstructure Translation

- **Domain:** Sequence-to-Sequence. **Source:** Bitcoin price, volume, search
  interest, and SEC media coverage around the January 9, 2024 SEC X account
  compromise, Zenodo, https://zenodo.org/records/20464457, CC BY 4.0,
  DOI 10.5281/zenodo.20464457 (includes coinbase_1min_jan2024.csv and a
  LICENSE file).
- **Transformation:** translate 1-minute OHLCV into attention/coverage
  intensity sequences. **Novelty:** 3/5. **Rejected:** one month around one
  event — far too few independent samples for stable grading.

## 13. Bitcoin Price-Factor Error-Correction Reproduction

- **Domain:** From Scratch. **Source:** Dataset used in "What drives the
  Bitcoin price?", Zenodo, https://zenodo.org/records/2562278, CC BY 4.0,
  DOI 10.5281/zenodo.2562278.
- **Transformation:** factor-augmented error-correction modeling.
- **Novelty:** 1/5. **Rejected:** reproducing a published econometric study
  is the definition of a benchmark clone; data ships as `.rar`.

## 14. Cross-Coin Transfer-Forecast Robustness Audit

- **Domain:** From Scratch. **Source:** Cryptocurrency Transfer Learning
  Forecasting Dataset, Zenodo, https://zenodo.org/records/20572501,
  CC BY 4.0, DOI 10.5281/zenodo.20572501.
- **Transformation:** audit forecast robustness across coins under transfer.
- **Novelty:** 3/5. **Rejected:** raw material collected via `yfinance`;
  Yahoo Finance terms do not clearly permit redistribution, so the Zenodo
  CC-BY declaration cannot cure the upstream ambiguity.

## 15. Ransomware Campaign Emergence Detection (streaming)

- **Domain:** From Scratch. **Source:** BitcoinHeist (as #3), UCI,
  https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset,
  CC BY 4.0, DOI 10.24432/C5BG8V.
- **Transformation:** detect the emergence day of unseen ransomware campaigns
  in a temporal stream. **Novelty:** 3/5. **Rejected:** same
  external-familiarity problem as #3 — the dataset and its labels are fully
  public and widely re-hosted, so campaign labels are reconstructable.

---

## Rejection-log summary of non-Zenodo sources considered

- **Binance public dumps (data.binance.vision):** no explicit license for the
  data — rejected (license-unclear).
- **Coin Metrics Community Data:** community license is non-commercial —
  rejected.
- **CoinGecko / CryptoDataDownload / Bitstamp API:** no clear redistribution
  grant — rejected.
- **Kaggle-hosted crypto datasets:** require credentialed (API-key) download —
  rejected under the no-credentials rule.
- **Google BigQuery public blockchain datasets (CC BY 4.0):** license is
  clear but access requires a GCP account — rejected (credentialed).

## Why #1 wins

Candidate #1 is the only concept that simultaneously (a) sits on a verified
permissive license with a credential-free download, (b) makes external
reconstruction of the answers structurally impossible rather than merely
inconvenient (the hidden out-of-sample simulation depends on undisclosed
strategy rules, not just undisclosed data), (c) defeats
summary-statistic/GBM shortcuts *by construction* (luck strategies are
rejection-sampled to match the skilled in-sample Sharpe distribution), and
(d) yields naturally heterogeneous targets with meaningful logical
consistency constraints and stratified buckets (strategy family × edge
regime), supporting a composite score that measures evaluation-modeling
ability rather than curve-fitting.
