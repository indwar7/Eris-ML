# Rubrics — Recovering OCR Batch Origin from Character Noise

- **[REQUIRED] Uses only the shipped data directory — no source-archive
  lookup, no external data.** The text is real and its source archive is
  public. A submission that recovers grouping by matching test rows back to
  the archive (or to any external copy of it), or that uses any data not
  shipped in the data directory, is not solving the task and is invalid
  regardless of score. The shipped data is built to make this path
  non-trivial — no batch identities are published anywhere, labels are
  opaque, and no test row is byte-identical to any archive region — but the
  rule, not the obfuscation, is the boundary. The solver environment is
  expected to have no network access; a solution must run to completion from
  the data directory alone.

- **[REQUIRED] Generalizes to unseen batches, does not memorize train batch
  identities.** Train and test batches are disjoint by construction. A
  correct approach learns what makes one OCR pipeline's noise pattern differ
  from another's in general, then applies that notion to snippets from
  batches never seen during training. Any approach that only works because it
  secretly assumes test items belong to one of the 19 known train batches
  (e.g. a closed-set classifier applied to open-set data without adaptation)
  is not solving the stated task.

- **[REQUIRED] Produces a valid partition per bag without assuming a fixed
  group count.** The number of true groups varies from bag to bag and is
  never given. A submission that hardcodes a single cluster count across all
  bags, or that silently fails on bags smaller or larger than an assumed
  size, does not meet the task as specified.

- **[REQUIRED] Partitions each bag using only that bag's own test rows —
  no pooled or cross-bag clustering.** Only 16 batches underlie the 110 test
  bags, so ignoring bag boundaries and clustering the entire test corpus at
  once, then slicing the result back into bags, is a transductive shortcut
  that recovers corpus structure rather than a generalisable notion of
  pipeline similarity. It is prohibited by the task rules, not merely
  discouraged: a submission built this way is invalid regardless of score.
  (It was measured during construction at 0.094 ARI, below the
  0.342 reference; per-bag scoring does not by itself prevent it, which is why the
  rule exists.) Learning from `train.csv` and applying that learned notion
  independently within each bag is the intended path.

- **[RECOMMENDED] Moves beyond a single cheap heuristic scalar.** Thresholding
  or clustering on one hand-picked statistic (e.g. non-ASCII character rate
  alone) was measured to plateau well below a properly trained multi-feature
  or learned-embedding approach during this task's construction. Solutions
  that combine multiple noise signals, or learn a representation end-to-end,
  are expected to outperform single-feature heuristics.

- **[RECOMMENDED] Learns a representation rather than relying on a single
  hand-picked statistic.** A trained multi-feature model (e.g. a tree-based
  classifier over character/noise features) is the strongest approach
  measured during this task's construction (0.342). A character-level neural
  encoder trained with a similarity-style objective on the labeled train
  batches was also measured and generalizes to unseen test batches, but did
  not exceed the tree-based approach at this data scale (~0.13 alone). Either
  is a reasonable path; a single hand-picked statistic alone was measured to
  plateau below both.

- **[RECOMMENDED] Validates using held-out train batches, not test signal.**
  Since test batches are never labeled, any hyperparameter or threshold
  tuning (e.g. choosing a similarity threshold or an expected group-count
  heuristic) should be validated on a held-out slice of the labeled train
  batches, not on the test set itself.

- **[UNIVERSAL] Documents the group-count estimation strategy used.** Since
  the true number of groups per bag is never given, a clear, brief note on
  how the submission estimates or avoids needing that count (e.g. a distance
  threshold plus connected components, silhouette-based selection, or a
  learned stopping criterion) helps interpret the results.
