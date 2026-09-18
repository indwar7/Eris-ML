# Rubrics — Experimental Narrative Reconstruction

- **[REQUIRED] Uses only the shipped data directory — no source lookup, no
  external data.** The subsections are real published text and the publisher's
  full-text search is public. A submission that recovers the order by locating
  the source article (by title, phrase, or any other key) and reading its
  document order, or that uses any data not shipped in the data directory, is
  not solving the task and is invalid regardless of score. No DOI, article
  title, or author is shipped, and `paper_id` is opaque, but the text itself
  is searchable — so the rule, not obfuscation, is the boundary. The solver
  environment is expected to have no network access; a solution must run to
  completion from the data directory alone.

- **[REQUIRED] Returns a valid permutation per paper.** Each paper's
  `predicted_order` must contain exactly that paper's slot letters, each once.
  A submission that repeats a slot, drops one, or uses letters from a different
  paper scores 0 for that paper and misrepresents the model's ability. Papers vary from 4 to 5 slots; a method that hardcodes a length does
  not meet the task.

- **[REQUIRED] Models the logical dependency between experiments, not surface
  position cues.** Figure numbers and explicit sequencing words are masked
  precisely because they would trivialise the task. A solution whose ordering
  comes from residual artefacts — subsection length, residual numerals, title
  length — rather than from reading what each experiment establishes and what
  it presupposes, has not solved the stated problem, even if it edges above
  chance.

- **[REQUIRED] Respects paper-disjointness in validation.** Train and test
  papers are disjoint. Any held-out validation must also hold out *whole
  papers*; splitting subsections of one paper across train and validation lets
  a model memorise the paper's narrative and inflates the estimate.

- **[REQUIRED] Reports the metric as per-paper position accuracy against
  the 1/n chance level.** Exact-match accuracy is not the scored quantity and
  is near zero for everyone; Kendall's τ is not the scored quantity either and
  its 0.5 chance level flatters weak orderings. A solution that validates on
  the wrong metric has not measured what the challenge scores; a solution that
  reports accuracy without reference to the ≈0.22 chance baseline can make
  chance look like progress.

- **[RECOMMENDED] Decomposes ordering into pairwise or listwise comparisons.**
  The intended approach is a model that reads two subsections and predicts
  which comes first, trained on all ordered pairs from the training papers,
  then decoded into a permutation (by score sorting, or by a consistent
  tournament). Predicting absolute positions independently ignores the
  relational structure that defines the task.

- **[RECOMMENDED] Uses the available GPU for a cross-encoder over subsection
  pairs.** Judging presupposition requires reading both texts jointly; a
  bi-encoder that embeds each subsection alone cannot see whether one claim
  depends on another. A fine-tuned cross-encoder over pairs is the intended use
  of the compute budget and is expected to clearly outperform feature-based
  heuristics.

- **[RECOMMENDED] Handles the decode step deliberately.** Pairwise predictions
  may be inconsistent (A>B, B>C, C>A). How the solution resolves cycles into a
  single permutation — sorting by aggregate win-score, greedy insertion, or
  assigning positions directly — is a real design choice that affects the
  score and should be made explicitly rather than by default. Position
  accuracy gives no credit for a globally shifted order, so the decode has to
  commit each experiment to a slot, not just get the pairs right.

- **[RECOMMENDED] Uses the subsection titles.** Titles are shipped because they
  are part of what the authors wrote, and a title like *"X regulates
  endocytosis of Y"* is a compressed statement of the claim whose logical
  position must be judged. Ignoring them discards the highest-density signal
  in the item.

- **[UNIVERSAL] Documents the decode strategy and the validation split.** A
  brief note on how pairwise scores became a permutation, and confirmation
  that validation held out whole papers, makes the accuracy result
  interpretable.
