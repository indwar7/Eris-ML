# Recovering Hidden Phrases in Bird-Song Syntax

Next-phrase prediction over stochastic, individual-specific birdsong grammars.
Top-1 accuracy. Majority baseline ≈0.05, bigram ≈0.27, strong sequence model
≈0.53, leaderboard ≈0.56.

## Why not a plain Transformer/LSTM

- **Markov models saturate fast.** A fixed order-k table either underfits
  the real long-range structure (small k) or fragments into contexts too
  sparse to estimate a reliable mode (large k), and a fresh test context
  is very likely to be entirely unseen at high order. Kneser-Ney smoothing
  fixes the sparsity failure mode but is still capped at whatever
  `kn_max_order` you pick.
- **Long-range structure is real.** Birdsong bouts are built from
  motifs/phrase-groups; whether a phrase is licensed at position `i` can
  depend on something 10+ tokens back (e.g. "has phrase X already
  occurred in this bout"), which no fixed-order chain can represent but a
  full-context attention/recurrent model can.
- **The grammar is genuinely stochastic**, not just noisy — several
  next-phrases are simultaneously valid at many positions. This caps
  top-1 accuracy well below 1.0 no matter the model, and rewards a model
  that finds the true *mode* of the distribution rather than memorizing
  one observed continuation. That's why label smoothing and Bayesian
  (Kneser-Ney) smoothing both appear in this solution — they keep the
  model's confidence honest instead of overfitting to which branch a
  bout happened to take.
- **Individual/dialect differences mean the "same" context can have a
  different modal next-phrase for different birds.** A model that pools
  across individuals without conditioning on identity throws this signal
  away.

## Architecture

`PhraseTransformer` (`model.py`): causal Transformer encoder (used as a
decoder via a causal attention mask) over phrase-token embeddings +
sinusoidal position embeddings + a projected per-bird "dialect" embedding
added at every position. Final hidden state (most recent token) is
projected to vocab logits.

`KneserNeyBackoff` (`dataset.py`): a smoothed variable-order Markov chain
(modified Kneser-Ney, order configurable up to `kn_max_order`), fit once
globally and once per inferred bird/individual id, with bird→global
backoff when a bird's table is sparse.

**Interpolation gate**: rather than a fixed mixing weight, a small MLP
(`model.py: PhraseTransformer.gate`) looks at the pooled Transformer hidden
state plus `log(1 + context_length)` and predicts, per example, how much
to trust the neural distribution vs. the Kneser-Ney distribution, combined
in log-space via `logsumexp`. This is the single highest-leverage design
choice: the two components fail in different regimes (KN is reliable on
short/common contexts, the Transformer is better positioned to exploit
long-range structure once there's enough data to learn it), so a
context-dependent gate captures gains a fixed ensemble weight can't.

Why this over other candidates considered (MoE, hierarchical attention,
beam search) — see the "Architecture Comparison" analysis delivered
alongside this code: MoE and hierarchical attention add real complexity
without a clear mechanism advantage at this problem's scale, and beam
search doesn't apply because this is single-token prediction, not
generation.

### Gate warmup curriculum (`Config.gate_warmup_epochs`)

Training the gate and the neural branch jointly from step 1 causes a
collapse failure mode: at initialization the neural branch is pure noise,
so NLL loss immediately teaches the gate to trust only the Kneser-Ney
branch (`gate -> ~0`). Once the gate is near 0, the neural branch gets
almost no gradient (its output barely affects the loss), so it never
improves, and the gate has no reason to reopen — a self-reinforcing
cold start observed empirically on this dataset (val_top1 got stuck at
exactly the KN-only accuracy for many epochs).

The fix (`trainer.py: fit`): for the first `gate_warmup_epochs` epochs,
train the Transformer alone on plain cross-entropy (`kn_log_probs=None`
forces `model.forward` into pure-neural mode) so it reaches a real,
non-trivial accuracy before it's ever compared against KN. Only then is
the learned gate switched on for the remaining epochs. Verified on a data
subset: neural-only warmup plateaued at ~0.29 val top-1; the instant
gating switched on, accuracy jumped to a stable ~0.47 and held across
subsequent epochs — confirming the gate now blends both branches instead
of collapsing. Early stopping and checkpoint saving are disabled during
the warmup epochs (it's a fixed-length curriculum stage, not something to
cut short on a metric plateau).

## Files

| File | Responsibility |
|---|---|
| `utils.py` | `Config` dataclass, seeding, top-k accuracy, checkpoint averaging, early stopping |
| `dataset.py` | vocab building, bird-id inference, Kneser-Ney backoff model, torch `Dataset`/collate |
| `model.py` | `PhraseTransformer`: embeddings, causal+padding-masked encoder, interpolation gate |
| `trainer.py` | AMP training loop, cosine LR w/ warmup, label smoothing, early stopping, checkpointing |
| `predict.py` | temperature scaling, checkpoint averaging (SWA), probability-space ensembling, context-window TTA |
| `train.py` | end-to-end training entrypoint; persists vocab/bird-vocab/KN models/config for inference |
| `submission.py` | loads trained artifacts, predicts on `test.json`, writes the submission CSV |
| `make_synthetic_data.py` | generates a toy dataset with the same qualitative properties, for local smoke-testing only |

## Running

```bash
python train.py --train_path data/train.json --max_epochs 60
python submission.py --checkpoint_dir checkpoints --test_path data/test.json \
    --output outputs/submission.csv
```

Ensemble across independently trained seeds by passing a comma-separated
list of checkpoint dirs to `submission.py --checkpoint_dir`:

```bash
python train.py --seed 1 --checkpoint_dir checkpoints_seed1
python train.py --seed 2 --checkpoint_dir checkpoints_seed2
python submission.py --checkpoint_dir checkpoints_seed1,checkpoints_seed2 \
    --output outputs/submission.csv
```

## Submission format

This challenge's `sample_submission.csv` uses columns `item_id,symbol`
(see `data/metadata.json`). This is configured via
`Config.submission_id_column` / `Config.submission_prediction_column`
rather than hardcoded, since the column name is challenge-specific.

## Hyperparameter search ranges

Reasonable ranges for a random/Bayesian search, given a dataset this size
(hundreds of sequences, vocab ~130, contexts up to ~40 tokens):

| Parameter | Range | Notes |
|---|---|---|
| `d_model` | 96 – 256 | Larger risks overfitting given the small number of training bouts |
| `n_layers` | 2 – 6 | Diminishing returns past ~4 at this data scale |
| `n_heads` | 4 – 8 | Must divide `d_model` |
| `d_ff` | 2×–4× `d_model` | Standard Transformer ratio |
| `dropout` | 0.1 – 0.35 | Push higher than usual (small dataset, low sequence count) |
| `bird_embed_dim` | 16 – 64 | Small — this is a coarse dialect signal, not the main capacity |
| `lr` | 1e-4 – 1e-3 (log-scale) | AdamW; higher end needs more warmup |
| `weight_decay` | 0.0 – 0.1 | |
| `warmup_steps` | 2–10% of total steps | |
| `label_smoothing` | 0.0 – 0.15 | Tune against val NLL, not just accuracy |
| `kn_max_order` | 3 – 7 | Higher orders need more data per context to avoid pure backoff |
| `gate_warmup_epochs` | 20–40% of `max_epochs` | Must be long enough for neural-only val accuracy to approach the KN baseline before gating switches on, or the gate collapses to KN-only (see below) |
| `kn_discount` | 0.5 – 0.9 | Standard modified-KN range |
| `batch_size` | 32 – 128 | Larger batches smooth the gate's gradient signal |
| `max_context_len` | 24 – 48 | Set from the test-set context-length distribution, not train sequence length |

Search objective: validation top-1 accuracy (the competition metric),
with validation NLL as a tie-breaker to prefer better-calibrated models
(this matters directly for the ensembling and temperature-scaling steps).

## Design choices not obvious from the code

- **Sequence-level train/val split** (`train.py: split_sequences_by_id`):
  splitting individual (context, target) examples instead of whole bouts
  would leak adjacent positions from the same bout across train/val,
  inflating validation accuracy without reflecting true generalization.
- **Left-padding, not right-padding**: keeps "the position to predict"
  fixed at index `-1` for every example regardless of context length,
  simplifying both the pooling step and the causal-mask construction.
- **Probability averaging for ensembling, not logit averaging**
  (`predict.py: predict_example`): correct combination rule when models
  can disagree in *direction* on a genuinely multi-modal distribution —
  logit averaging lets one confidently-wrong model suppress another's
  correct minority prediction; probability averaging doesn't.
- **Context-window TTA via cropping, not augmentation-by-noise**: because
  the task is literally "how much left context should I trust," cropping
  the context and re-predicting is a TTA strategy that directly probes
  the model's own robustness question, unlike generic techniques (token
  dropout, shuffling) that don't map to a real inference-time choice here.
