## Approach

**Architecture**: a causal Transformer decoder (bird/individual-conditioned) combined
with a smoothed variable-order Markov chain (modified Kneser-Ney), mixed via a
**learned per-example gate** rather than a fixed weight. The gate reads the
Transformer's pooled hidden state plus log(context length) and decides how much
to trust each branch per example. Rationale: Kneser-Ney is a strong, reliable
estimator for short/common contexts, while the Transformer can exploit
longer-range structure once it has enough data to learn it -- the two fail in
different regimes, so a context-dependent mixing weight beats a fixed one.

**Preprocessing**: symbols are vocab-encoded (unseen symbols, including the
`MASK` token that appears only in test contexts, fall back to `<UNK>`).
Training sequences are expanded into (left-context -> next-token) examples,
left-padded so the position to predict is always the last index. Contexts are
capped at 40 tokens, matching the test set's context-length distribution.
Train/val split is done at the sequence level (not example level) to avoid
leaking adjacent positions from the same sequence across the split.

**Key design decision -- gate warmup curriculum**: training the gate and the
neural branch jointly from step 1 causes a collapse failure mode. At
initialization the neural branch is pure noise, so the loss immediately
teaches the gate to trust only Kneser-Ney (gate -> 0), which then starves the
neural branch of gradient and it never recovers -- confirmed empirically
(validation accuracy got stuck exactly at the KN-only accuracy for many
epochs under naive joint training). The fix: the Transformer trains alone on
plain cross-entropy for the first `gate_warmup_epochs` epochs (forcing
KN-free pure-neural mode), and only then is the learned gate switched on for
the remaining epochs. This is the single most important lesson from building
this solution -- without it the neural branch contributes nothing to the
final ensemble.

**Training**: AdamW, cosine LR schedule with linear warmup, gradient clipping,
label smoothing (justified by the genuinely stochastic grammar -- several
next-symbols are frequently all valid at a position, so hard one-hot targets
would push false overconfidence), early stopping on validation top-1 (active
only once both branches are training jointly), mixed precision on GPU.

**Inference**: SWA-style checkpoint averaging over the top-K validation
checkpoints, a 1-D temperature-scaling grid search fit on the validation
split, and context-window test-time augmentation (predicting from 100%/75%/50%
of the available context and averaging the resulting probability
distributions) to guard against overweighting a spurious long-range
coincidence when the short-range signal is actually cleaner.

**What worked**: the gate warmup fix was the difference between the neural
branch contributing nothing and the ensemble beating the Kneser-Ney floor.
**What didn't work initially**: naive joint gate+neural training (silent
gate collapse -- looked like a working model but was secretly 100% Kneser-Ney).
