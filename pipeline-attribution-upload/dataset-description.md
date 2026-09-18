# Dataset Description — Newspaper OCR Grouped by Scanning Batch

## Overview

Real historical newspaper text, paired with the identity of the digitization
pipeline that produced it.

Each record is one OCR'd text region — roughly one article or column — lifted
from a real page of a real newspaper digitized under a **national,
multi-decade newspaper-digitization program**, together with the **real
batch identifier** under which that page was processed, and the OCR engine and
contracted vendor that batch's own archival metadata records.

Nothing here is written, paraphrased, annotated or model-generated. The text is
the OCR engine's own uncorrected output, concatenated from the `<String>`
elements of the page's ALTO XML. The label is the archive's own batch
identifier, taken from the directory the page was published under. The vendor
fields are the values the digitizing institution itself wrote into that batch's
ALTO metadata. No human read a snippet and decided anything about it.

What makes the label learnable is that each batch was processed by its own
digitization run — its own engine and engine version, its own vendor settings,
its own scanning era and source microfilm — across several decades of the
program. Each such run left a systematic character-error fingerprint in its
output: its own habits of substitution, its own junk characters, its own way of
breaking word boundaries. That fingerprint, not the words, is the signal.

### Dataset at a glance

- **Batches (label classes)** — 35
- **Text snippets** — 8,417
- **Distinct institution prefixes** — 34
- **OCR engine families** — 3 — ABBYY (19 batches), iArchives (15), Prime (1)
- **Issue-date span** — roughly 1831–1960
- **Snippets per batch** — 113 min / 241 median / 356 max
- **Distinct issue dates per batch** — 13 min / 18 median / 26 max
- **Minimum snippet length** — 40 characters
- **Source** — The program's public archive of digitized historical newspapers. The dataset's Source URL (platform metadata, required for licensing and verified by an automated fetch) points to the archive's public data root — the only page of the program that answers automated requests; the rest of the site returns 403. Which batches were used, the batch naming scheme, and the candidate-selection rule are deliberately not given anywhere solver-visible (see *Label leakage*). Full citation in the raw-side `ATTRIBUTION.txt`
- **License** — Public Domain Mark — per the archive's own rights statement
- **Fetch date** — 2026-08-12
- **Human annotation** — None — every field is the archive's own recorded value

## Source

Everything in this dataset — both the text and its label — comes from a single
public archive. There is no second source and no join across providers.

- **PRIMARY** — Source: A national newspaper-digitization program's public archive (data root in the dataset's Source URL; full citation in the raw-side `ATTRIBUTION.txt`) · Endpoint: Static per-batch directories; the batches used are not identified · What is taken from it: Everything: OCR text, batch identity, vendor and OCR-engine metadata

Each batch directory publishes a manifest of the issues that batch contains.
Each issue's metadata file points at its pages; each page's OCR XML (ALTO)
carries both the text and the batch's own software/vendor metadata. Every
field in this dataset is read directly out of those files.

### Fetch method, step by step

- **1** — Step: Fetch manifest · Detail: Retrieve the batch manifest for the candidate `batch_id`
- **2** — Step: Sample issues · Detail: Evenly sample up to 18 distinct issue dates across the batch's full run
- **3** — Step: Locate page · Detail: Fetch each sampled issue's metadata to find its first page's OCR (ALTO) file
- **4** — Step: Extract regions · Detail: Keep every `TextBlock` with ≥ 40 characters of OCR text
- **5** — Step: Cap per page · Detail: At most 14 TextBlocks per page, so no single page dominates its batch

### Candidate selection and outcome

Candidate batches were drawn from across the archive's institution prefixes
and kept if the manifest held at least 4 distinct issue dates and the batch
yielded at least 60 snippets once fetched. **35 batches** were kept, yielding
**8,417 snippets**. The exact candidate-selection rule and the list of
attempted-but-skipped batch ids are deliberately not published here: together
with the public archive they would let a solver enumerate the corpus's batches
and match shipped text back to its source (see *Label leakage* below). They
remain in the raw-side `meta.json`, which is not part of the solver-visible
data directory.

## File Structure

- **`raw/snippets.jsonl`** — Format: JSON Lines · Contents: One line per OCR'd text region: the text plus its batch label
- **`raw/batches_meta.csv`** — Format: CSV · Contents: One row per batch: vendor, OCR engine, per-batch counts
- **`raw/meta.json`** — Format: JSON · Contents: Fetch provenance, selection parameters, coverage statistics
- **`raw/ATTRIBUTION.txt`** — Format: Text · Contents: Citation and the full licensing caveat
- **`generator/build_raw.py`** — Format: Python · Contents: The resumable fetcher that produced all of the above
- **`generator/README.md`** — Format: Markdown · Contents: Fetch method, rate-limit behaviour, known gotchas

`prepare.py` consumes `raw/snippets.jsonl` and derives the public and private
splits from it — the bagged items solvers read, the training labels, and the
hidden answer key. Those derived files are products of the prepare step. Two
things happen on the way from raw to public, both to close a measured
source-lookup path (see *Label leakage*): every real `batch_id` is replaced by
an opaque per-batch label (`batch_00` …) in both `train.csv` and the private
answer key, and every shipped snippet is a contiguous character window of its
source region (80–97% of it, bounds from a private seed) rather than the
region verbatim. Every test row is a distinct source snippet.

## Features

### `snippets.jsonl` — one JSON object per line, one per OCR'd text region

- **`snippet_id`** — Type: `str` · Nullable: No · Description: Unique ID, formed as `<batch_id>_<lccn>_<issue_date>_<seq>_<block_idx>`. **Contains the label as a literal prefix** — must not reach a solver at prediction time. Example: `ak_albatross_ver01_sn84021930_1898-11-23_1_0`
- **`batch_id`** — Type: `str` · Nullable: No · Description: **The label.** The archive's real digitization batch identifier, read from its own directory structure. Example: `ak_albatross_ver01`
- **`lccn`** — Type: `str` · Nullable: No · Description: The archive's catalogue control number for the source newspaper title. Near batch-identifying on its own. Example: `sn84021930`
- **`issue_date`** — Type: `str` (`YYYY-MM-DD`) · Nullable: No · Description: Real publication date of the source issue. Example: `1898-11-23`
- **`text`** — Type: `str` · Nullable: No · Description: Raw, uncorrected OCR output for one article/column region, concatenated from ALTO `<String CONTENT="...">` values in reading order. ≥ 40 characters. Not cleaned, spell-corrected, or normalised

### `batches_meta.csv` — CSV with a header row, one row per batch

- **`batch_id`** — Type: `str` · Nullable: No · Populated: 35/35 · Description: Joins to `snippets.jsonl.batch_id`
- **`institution_prefix`** — Type: `str` · Nullable: No · Populated: 35/35 · Description: Contributing institution's batch-ID prefix. 34 distinct values. Example: `ak`, `ct`, `mnhi`
- **`n_snippets`** — Type: `int` · Nullable: No · Populated: 35/35 · Description: Snippets kept from this batch. Range 113–356
- **`n_distinct_issue_dates`** — Type: `int` · Nullable: No · Populated: 35/35 · Description: Distinct issue dates sampled. Range 13–26
- **`software_creator`** — Type: `str` · Nullable: No · Populated: 35/35 · Description: OCR engine's vendor per the batch's ALTO. Example: `ABBYY`, `ABBYY Software LLC`
- **`software_name`** — Type: `str` · Nullable: No · Populated: 35/35 · Description: OCR engine/product name. Example: `ABBYY FineReader`, `ABBYY Recognition Server`. Recorded verbatim from ALTO and occasionally inconsistent with `software_creator` — four batches list `iArchives` as creator alongside an ABBYY product name. Treated as a real property of the archive's metadata, not corrected
- **`processing_agency`** — Type: `str` · Nullable: No · Populated: 35/35 (17 real values, 18 sentinel) · Description: Contracted digitization vendor. Example: `Apex CoVantage`, `HTC Global Services`. **17 batches** carry a real vendor name; the other **18** carry the literal sentinel `not_recorded` because the batch's ALTO never logged it — treat `not_recorded` as missing metadata, **not** as a distinct vendor

### `meta.json` — a single JSON object

- **`source`** — Type: `str` · Description: Archive name
- **`source_url`** — Type: `str` · Description: Bulk-data mirror URL
- **`fetch_date`** — Type: `str` · Description: `2026-08-12`
- **`fetch_approach`** — Type: `str` · Description: Prose description of the 5-step fetch method above
- **`candidate_selection`** — Type: `str` · Description: Prose description of how candidate batches were drawn (raw-side only; not reproduced in public docs — see *Label leakage*)
- **`n_batches_kept`** — Type: `int` · Description: `35`
- **`total_snippets`** — Type: `int` · Description: `8417`
- **`snippets_per_batch`** — Type: `object` · Description: Nested `min` / `median` / `max` — `113` / `241` / `356`
- **`issue_dates_per_batch`** — Type: `object` · Description: Nested `min` / `median` / `max` — `13` / `18` / `26`
- **`n_distinct_institution_prefixes`** — Type: `int` · Description: `34`
- **`vendor_metadata_coverage`** — Type: `object` · Description: Per-field population counts, with a note on why `processing_agency` is partial
- **`rate_limiting`** — Type: `object` · Description: Observed 429 behaviour and the handling applied
- **`skipped_batches`** — Type: `object` · Description: `{batch_id: reason}` for the 20 candidates attempted and not kept — all `http 404 on the batch manifest` (listed in the archive's index but not actually present). Persisted here so re-running the fetcher does not spend rate-limited requests on known-bad candidates

## Notes

**What the signal actually is — and what it is not.** The label is a **batch**,
which is finer-grained than an OCR engine. Only three engine families appear
across the 35 batches (ABBYY in 19, iArchives in 15, Prime in 1), so two
batches frequently share an engine and are still separable — what differs is
the whole digitization run: engine version, vendor settings, scanning era,
source microfilm stock, and the contracted agency's own processing choices.
What is recovered is therefore a *batch* fingerprint, of which the engine is
one component among several, not a clean per-engine signature.

The signal is character-level and content-free. Two batches may cover
overlapping decades and similar regional papers and remain separable because
their pipelines garble text differently, not because they cover different
subjects. This was measured rather than assumed: word-level TF-IDF clustering —
an approach that reads what the text *says* — scores **0.079**, and character
n-gram TF-IDF **0.038**, both at the noise floor, against **0.342** for a
reference built on content-free OCR-noise features. Any approach that leans on topic, vocabulary,
or place names is reading the newspaper rather than the pipeline, and will not
generalise to held-out batches.

Snippet length is a separate, real confound and should not be mistaken for
content-blindness in the TF-IDF numbers above: mean snippet length varies
roughly 10x across batches (~290 to ~3,050 characters), and one held-out
batch is majority-Spanish. Clustering on character count alone scores
**0.089** per bag, above the word-TF-IDF baseline. It is still well below the
0.342 reference and carries no fingerprint information, but it is a measured
shortcut, not a theoretical one — the current windowing does not fully
remove it.

**The text is genuinely noisy, and that is the point.** Character
substitutions, junk characters and broken word boundaries throughout are the
real output of period-appropriate OCR engines run against period newspaper
print — degraded microfilm, tight columns, broken type. None of it is synthetic
corruption injected to make the task harder. A sample snippet reads in full:
`vcu , 0OUGLAS CVVY AND TRKADWKLL, ALASKA. NOVEMBER 23, 1898. NO. 1.`

**Nothing was annotated.** Every `batch_id` and every vendor field is the
archive's own recorded value. There is no human labelling step anywhere in the
build, so there is no annotator disagreement, no labelling guideline, and no
label noise beyond whatever the archive itself recorded.

**Label leakage to watch for.** Because the text is real and its archive is
public, the answer can leak through any channel that lets a solver find a
shipped row's source region. Four such channels were identified — two at
design time, two by review — and all are closed in `prepare.py`:

- **`snippet_id`** — How it leaks: Contains `batch_id` verbatim as a literal prefix · Closed by: Never shipped; public row ids are opaque
- **`lccn`** — How it leaks: Near batch-identifying — a given newspaper title was typically digitized by one institution · Closed by: Never shipped
- **Batch identities in public files** — How it leaks: A reviewer took the held-out batch ids from an earlier public stats file, fetched those batches from the archive, exact-matched every test row, and scored 1.000 with no modeling · Closed by: No batch identity is published anywhere — stats carry counts only, and all labels are opaque `batch_NN`
- **Verbatim text** — How it leaks: With the batch set known, a shipped region is byte-matchable against the archive · Closed by: Every shipped snippet is a windowed sub-span of its source; zero test rows are byte-identical to any archive region

- **The archive's data pointer** — How it leaks: naming the bulk-data directories, the batch naming scheme, or the selection rule in solver-visible text hands a solver the index for real, searchable OCR (review round 2) · Closed by: none of those appear in the data directory or the task text. The dataset's Source URL (platform licensing metadata, validated by fetch) necessarily points at the archive's data root; the path is closed independently of that URL — no batch identity is published (35 of hundreds of batches, unnamed), every label is opaque, no shipped row is byte-identical to any archive region (0 of 3,568), and lookup is prohibited by rule in an offline solver environment

`prepare.py` enforces batch-disjoint train/test splits and samples test
snippets without replacement, so every test row is a distinct source region.
The candidate-selection rule and the skipped-batch list are likewise kept out
of public docs, since they would let the batch set be re-enumerated. None of
this alters the OCR text itself: the noise is the archive's own, uncorrected.

**Sampling is spread deliberately.** Issue dates are sampled evenly across each
batch's full run rather than taken consecutively, and TextBlocks are capped at
14 per page, so a batch's snippets are not all drawn from one issue, one year,
or one dense page. This is what keeps the fingerprint batch-wide rather than
page-specific.

**Fetch was rate-limited.** The source archive returns HTTP 429 under load. The
fetcher holds a 1.1s pace (~55 requests/minute); on a 429 it sleeps 30s and
retries the same request, up to 3 attempts. `generator/build_raw.py` is
resumable and documents this, so the corpus can be extended later without
re-fetching what is already present.

**Licensing.** The archive's own rights statement is that the newspapers in
the collection "are in the public domain or have no known copyright
restrictions": newspapers published in the United States more than 95 years
ago are in the public domain in their entirety, and those published less than
95 years ago are also believed to be in the public domain but may contain
some copyrighted third-party material. The program admits only newspaper
titles and runs that the contributing institution has independently confirmed
rights-clear for digitization and public redistribution. This dataset is
therefore distributed under the **Public Domain Mark**, matching the source.
Per the archive's own advisory, users of the youngest snippets (`issue_date`
within the last 95 years) should be alert for modern third-party content that
may carry copyright; `issue_date` is recorded on every snippet so that can be
checked. The full citation lives in `raw/ATTRIBUTION.txt`, which is raw-side
and not part of the solver-visible data directory.
