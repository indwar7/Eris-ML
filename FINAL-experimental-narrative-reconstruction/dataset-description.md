# Dataset Description — PLOS Experimental Sequences: Real Results Sections in Author Order

## Overview

Real scientific results, with the authors' own narrative order as the label.

Each record is one PLOS research paper's **Results section**, cut into the
titled subsections the authors wrote, in the sequence they wrote them. That
sequence is the ground truth: a shuffled version is shipped, and the task is to
recover the original.

Nothing here is written, paraphrased, annotated or model-generated. The text is
the authors' own published prose, the subsection titles are theirs, and the
order is the one they chose when structuring their argument. No one read a
paper and decided anything about it for this dataset.

## What makes the label learnable

A Results section is a chain of experiments with a logical direction. Authors
order subsections so that each one builds on what came before:

| Position | Typical role |
|---|---|
| early | establish that the phenomenon or molecule exists; characterise it |
| middle | test a mechanism; identify interacting factors |
| late | rule out alternatives; show the mechanism is *required*; extend to a new system |

That structure is **presuppositional**: a subsection showing that X binds Y
cannot precede the one establishing X is present; a control ruling out an
artefact follows the result it protects. Recovering the order therefore
requires understanding what each experiment establishes and what it assumes.
There is no stylistic shortcut — every subsection is in the same register, from
the same authors, and reports statistics.

### Dataset at a glance

| Property | Value |
|---|---|
| Unit | one paper's Results section (4–5 titled subsections shipped; papers with more contribute their first 5) |
| Source | PLOS research articles, full JATS XML |
| Label | the authors' own subsection order |
| Feature | paper discipline (second level of PLOS's subject taxonomy) |
| Licence | **CC BY 4.0** |
| Fetch date | 2026-09-13 |
| Human annotation | None |

Counts are recorded in `raw/meta.json` and, for the prepared split, in
`public/dataset_stats.json`.

## Source

| Role | Source | Endpoint | What is taken from it |
|---|---|---|---|
| PRIMARY | PLOS research articles | PLOS Search API (DOI discovery); PLOS article XML endpoint (full JATS XML) | Everything: subsection titles, text, document order, subject area |

### Fetch method

1. Query the PLOS Search API for `doc_type:full AND article_type:"Research
   Article"`, paging through results; skip `/annotation/` records.
2. Fetch each article's full JATS XML.
3. Walk the `<sec>` tag stack to locate the top-level **Results** section and
   its **direct child** subsections, in document order. (Nested tag tracking
   matters: PLOS papers nest subsections, and a flat scan silently drops most
   of them.)
4. Keep papers with 4–8 such subsections in the raw corpus, each within a
   length window (the prepared split ships the first 5 of each). Fewer
   makes ordering trivial; more makes it intractable.
5. For each subsection, concatenate its paragraphs; strip tags, inline
   cross-references and bracketed citation markers.
6. **Mask the shortcuts**: figure and table numbers → `<FIG>` (they are
   monotone in document order); explicit sequencing words — *first*, *next*,
   *finally*, *having shown*, *as described above*, *we then* — → `<SEQ>`.

Requests are rate-limited with a delay between calls and retried with backoff.

## File Structure

- `raw/papers.jsonl` — one JSON object per paper
- `raw/paper_meta.csv` — DOI, discipline and subsection count per paper
- `raw/meta.json` — fetch provenance, filters and masking policy
- `raw/ATTRIBUTION.txt` — source, licence and citation

## Features

### `raw/papers.jsonl`

| Column | Type | Description |
|---|---|---|
| `doi` | str | Source article DOI |
| `field` | str | Discipline, published as a feature |
| `n_subsections` | int | Number of Results subsections in the raw corpus (4–8); the prepared split ships the first 5 |
| `subsections` | list | **In author order.** Each `{title, text}` |

The order of the `subsections` list is the label. The prepared split shuffles
it and assigns opaque slot letters after shuffling.

## Intended Use

**Primary use.** Benchmarking scientific narrative reasoning: recovering the
logical order of a sequence of experiments from their content. The dataset is
built so that position cues cannot substitute for understanding (figure numbers
and sequencing words are masked), so that style cannot substitute for logic
(all items share register and authorship), and so that memorisation cannot
substitute for generalisation (papers are disjoint across splits).

**Also suitable for.** Discourse-coherence and sentence-ordering research on
technical text, pairwise-preference and listwise ranking model evaluation,
studies of scientific argument structure, and probing whether language models
represent experimental presupposition.

**Not suitable for.** Fact extraction or claim verification (no factual
labels), scientific quality or novelty assessment, section-type classification
(every item is Results), or any use that treats the order as a quality
judgement — it is the authors' choice, not a correctness label.

## Limitations

These are properties of the shipped data, stated so results are not over-read.

**Authorial order is a choice, not a law.** The label is the sequence the
authors chose. Some experiments genuinely have no logical dependency and could
be presented in either order; some authors order for narrative effect rather
than dependency. Under position accuracy an adjacent swap costs two
positions of a paper, so a ceiling below 1.0 is inherent, not a modelling
failure — and it is part of why the metric leaves headroom.

**Only conventionally structured papers are included.** Papers whose Results
section has fewer than 4 or more than 8 titled subsections, or no titled
subsections at all, are excluded. The corpus is biased toward papers that
structure their Results as a titled sequence, which are likely the papers with
the clearest narrative — the easier cases.

**Titles do heavy lifting.** Subsection titles are shipped because they are the
authors' text, and they compress the claim. A substantial fraction of the
recoverable order may be inferable from titles alone. This is disclosed rather
than removed because titles are genuine content, not a leak.

**Source lookup is possible in principle and prohibited by rule.** The text is
verbatim published prose, and PLOS's full-text search is public: a single
shipped subsection title was checked against it and, for some papers, returns
exactly one article, whose XML carries the document order — i.e. the answer.
Nothing shipped names the article (no DOI, title, author, or journal;
`paper_id` is opaque), but the text itself is the search key, and no
signal-preserving transform of a unique passage can stop that. Masking or
paraphrasing the text would destroy the "authors' own words" property the
task depends on. The closure is therefore procedural: the problem statement
and a `[REQUIRED]` rubric prohibit external data and source lookup outright,
the solver environment is expected to be offline, and no exact query
endpoint is reproduced in solver-visible documentation. A score near 1.0
from a solver should be treated as evidence of lookup, not of ability — the
measured content-based ceiling is well below it.

**Masking is regex-based, not exhaustive.** Figure numbers and a list of
sequencing expressions are masked, but scientific prose has many ways to
signal order ("the above data", "this raised the question", "to confirm
this"). Some survive. A residual-cue check is recommended.

**Field coverage is skewed toward biomedicine.** PLOS's output is dominated by
biology and medicine. Experimental narratives in other fields follow different
conventions, and the corpus does not represent them evenly.

**One publisher's structuring conventions.** PLOS author guidelines shape how
Results are subdivided. The narrative patterns learned here may transfer
imperfectly to journals with different conventions.

**Real people are named in the text.** Author names appear in citations and
acknowledgements within subsection text where they survived cleaning. The data
is public and the task is not about the individuals.

## License

**Creative Commons Attribution 4.0 International (CC BY 4.0)**
https://creativecommons.org/licenses/by/4.0/

Every PLOS research article carries the Creative Commons Attribution licence in
its own `<license>` block:

> "This is an open-access article distributed under the terms of the Creative
> Commons Attribution License, which permits unrestricted use, distribution,
> and reproduction in any medium, provided the original author and source are
> credited."

Commercial use is permitted with attribution.

**Attribution:** each paper is identified by DOI in the raw corpus, so every
item can be traced to its authors.

Citation: PLOS (Public Library of Science). Article content retrieved via the
PLOS Search API and article XML endpoint, 2026-09-13.
